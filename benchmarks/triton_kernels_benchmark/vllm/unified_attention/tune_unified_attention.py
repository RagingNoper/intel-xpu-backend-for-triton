# SPDX-License-Identifier: Apache-2.0
"""Tune XPU attention with graph replay and export device-specific JSON configs.

Run ``measure --manifest inputs.json --output measurements.json``, then
``select --measurements measurements.json --output-folder profiles``.
"""

from __future__ import annotations

# Exact integer checks reject bools in workload manifests.
# pylint: disable=unidiomatic-typecheck

import argparse
import importlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

FORMAT_VERSION = 3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def candidate_configs():
    return [{"block_m": m, "tile_size": t, "num_warps": w, "num_stages": s, "grf_mode": g}
            for m in (16, 32, 64)
            for t in (16, 32, 64, 128)
            for w in (2, 4, 8, 16)
            for s in (1, 2, 3)
            for g in ("default", "256")]


def config_id(config):
    return "fallback" if config is None else canonical(config)


# pylint: disable-next=too-many-branches
def normalize_workloads(raw):
    if not isinstance(raw, list) or not raw:
        raise ValueError("Manifest must be a nonempty JSON list")
    cases, ids = [], set()
    for index, source in enumerate(raw):
        case = dict(source)
        case.setdefault("id", f"case-{index:04d}")
        if not isinstance(case["id"], str) or case["id"] in ids:
            raise ValueError("Every workload needs a unique string id")
        ids.add(case["id"])
        case.setdefault("q_layout", "contiguous")
        if case["q_layout"] not in ("contiguous", "qkv"):
            raise ValueError("q_layout must be contiguous or qkv")
        case.setdefault("kv_layout", "contiguous")
        if case["kv_layout"] not in ("contiguous", "interleaved"):
            raise ValueError("kv_layout must be contiguous or interleaved")
        case.setdefault("dtype", "bfloat16")
        case["dtype"] = {"bf16": "bfloat16", "fp16": "float16", "fp8":
                         "float8_e4m3fn"}.get(case["dtype"], case["dtype"])
        if case["dtype"] not in ("bfloat16", "float16", "float8_e4m3fn"):
            raise ValueError("Supported query/K/V dtypes: bfloat16, float16, float8_e4m3fn")
        case.setdefault("out_dtype", "bfloat16" if case["dtype"].startswith("float8") else case["dtype"])
        if case["out_dtype"] not in ("bfloat16", "float16"):
            raise ValueError("Output dtype must be bfloat16 or float16")
        for name, default in (
            ("batch", 1),
            ("block_size", 32),
            ("head_size", 128),
            ("q_heads", 32),
            ("kv_heads", 8),
            ("num_segments", 16),
            ("seq_threshold_3d", 32),
        ):
            case.setdefault(name, default)
            minimum = 0 if name == "seq_threshold_3d" else 1
            if type(case[name]) is not int or case[name] < minimum:
                raise ValueError(f"{case['id']}: {name} must be an integer >= {minimum}")
        if case["q_heads"] % case["kv_heads"]:
            raise ValueError("q_heads must be divisible by kv_heads")
        for plural, scalar in (("query_lens", "query_len"), ("kv_lens", "kv_len")):
            if plural not in case:
                if scalar not in case:
                    raise ValueError(f"{case['id']}: provide {plural} or {scalar}")
                case[plural] = [case[scalar]] * case["batch"]
            if len(case[plural]) != case["batch"] or any(
                    type(length) is not int or length <= 0 for length in case[plural]):
                raise ValueError(f"{case['id']}: invalid {plural}")
        if any(q > k for q, k in zip(case["query_lens"], case["kv_lens"])):
            raise ValueError("Every query length must be <= its KV length")
        case.setdefault("sliding_window", 0)
        case.setdefault("softcap", 0.0)
        if type(case["sliding_window"]) is not int or case["sliding_window"] < 0:
            raise ValueError("sliding_window must be a nonnegative integer")
        if not isinstance(case["softcap"], (int, float)) or not math.isfinite(case["softcap"]) or case["softcap"] < 0:
            raise ValueError("softcap must be finite and nonnegative")
        if case["block_size"] % 16:
            raise ValueError("The initial TD workloads require block_size divisible by 16")
        if case["dtype"].startswith("float8_") and case["block_size"] % 32:
            raise ValueError("Native FP8 + TD requires block_size divisible by 32; pointer fallback is disabled")
        cases.append(case)
    return cases


def successful(option, field="samples_ms"):
    values = option.get(field, [])
    return (option.get("status") == "ok" and bool(values)
            and all(isinstance(value, (float, int)) and math.isfinite(value) and value > 0 for value in values))


def key_statistics(cases, field="samples_ms"):
    """Only compare candidates having a successful cell for every shape."""
    rows = [{option["id"]: option for option in case["options"] if successful(option, field)} for case in cases]
    if any("fallback" not in row for row in rows):
        return None
    common = set.intersection(*(set(row) for row in rows))
    # Keep initial sweep references even when they are not finalists.
    best = [
        min(
            statistics.median(option[phase])
            for option in case["options"]
            for phase in (field, "samples_ms")
            if successful(option, phase))
        for case in cases
    ]
    fallback = [statistics.median(row["fallback"][field]) for row in rows]
    results = []
    for identifier in sorted(common):
        times = [statistics.median(row[identifier][field]) for row in rows]
        ratios = [latency / minimum for latency, minimum in zip(times, best)]
        fallback_ratios = [latency / baseline for latency, baseline in zip(times, fallback)]
        regressions = [ratio - 1 for ratio in fallback_ratios]
        results.append({
            "id": identifier,
            "config": rows[0][identifier]["config"],
            "max_regret": max(ratios) - 1,
            "geomean_ratio": math.exp(statistics.mean(map(math.log, ratios))),
            "geomean_fallback_ratio": math.exp(statistics.mean(map(math.log, fallback_ratios))),
            "max_fallback_regression": max(regressions),
            "ratios": ratios,
            "fallback_regressions": regressions,
        })
    return results


def choose_key(cases, *, tolerance=0.01, field="confirmation_ms"):
    stats = key_statistics(cases, field)
    if stats is None:
        return {"accepted": False, "reason": "missing or failed fallback measurement"}
    optimum = min(item["max_regret"] for item in stats)
    tied = [item for item in stats if item["max_regret"] <= optimum + tolerance]
    winner = min(tied, key=lambda item: (item["geomean_ratio"], item["id"]))
    return {
        "accepted": True,
        "winner": winner,
        "fallback": next(item for item in stats if item["id"] == "fallback"),
        "candidates": [item for item in stats if item["id"] != "fallback"],
    }


def grouped_cases(cases):
    grouped = defaultdict(list)
    for case in cases:
        if case.get("key") is not None:
            grouped[canonical(case["key"])].append(case)
    return grouped


def finalists(cases, count=3):
    stats = key_statistics(cases)
    if stats is None:
        return ["fallback"]
    ranked = sorted(
        (item for item in stats if item["id"] != "fallback"),
        key=lambda item: (item["max_regret"], item["geomean_ratio"], item["id"]),
    )
    selected = ["fallback"] + [item["id"] for item in ranked[:count]]
    initial = choose_key(cases, field="samples_ms")
    if initial["accepted"]:
        selected.append(initial["winner"]["id"])
    for case in cases:
        measured = [option for option in case["options"] if successful(option)]
        if measured:
            selected.append(min(measured, key=lambda option: statistics.median(option["samples_ms"]))["id"])
    return list(dict.fromkeys(selected))


def load_by_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def runtime_module(path=None):
    if path:
        return load_by_path(path, "_ua_offline_profile_contract")
    return importlib.import_module("vllm.v1.attention.ops.triton_unified_attention_config")


def select_results(data, runtime, output, *, tolerance=0.01):
    if data.get("format_version") != FORMAT_VERSION or not data.get("complete"):
        raise ValueError("Selection requires a complete measurement file")
    if data.get("timing_scope") != "attention_sequence_graph_device":
        raise ValueError("Selection requires graph measurements")
    if sorted(case["id"] for case in data["cases"]) != sorted(case["id"] for case in data["manifest"]):
        raise ValueError("Measurements must include every manifest input")
    identity = data["identity"]
    partitions = defaultdict(list)
    for serialized_key, cases in grouped_cases(data["cases"]).items():
        key = runtime.AttentionKey(**json.loads(serialized_key))
        result = choose_key(cases, tolerance=tolerance)
        if not result["accepted"]:
            raise ValueError(f"Cannot select a config for {[case['id'] for case in cases]}")
        winner = result["winner"]["config"]
        config = runtime.AttentionConfig(**winner) if winner is not None else None
        partitions[key.static_key()].append((key.dynamic_key(), config))
    profiles = {}
    for static, entries in partitions.items():
        name = runtime.profile_filename(identity, static)
        destination = Path(output) / name
        runtime.write_profile(destination, identity, entries, static_key=static)
        profiles[name] = str(destination)
    return {"profiles": profiles, "accepted_keys": sum(map(len, partitions.values()))}


def torch_environment():
    # pylint: disable=import-outside-toplevel
    # Keep selection and CLI parsing independent of GPU packages.
    import torch
    import triton

    runtime = runtime_module()
    attention = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    return torch, triton, runtime, attention


def allocate_case(case, torch, device, seed):
    torch.manual_seed(seed)
    dtype, out_dtype = getattr(torch, case["dtype"]), getattr(torch, case["out_dtype"])

    def random_tensor(shape):
        storage_dtype = torch.bfloat16 if case["dtype"].startswith("float8") else dtype
        return torch.empty(shape, device=device, dtype=storage_dtype).normal_(std=0.5).to(dtype)

    q_lens, kv_lens = case["query_lens"], case["kv_lens"]
    block, dim, q_heads, kv_heads = (case[name] for name in ("block_size", "head_size", "q_heads", "kv_heads"))
    counts = [(length + block - 1) // block for length in kv_lens]
    num_blocks = case.get("num_blocks", sum(counts))
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    pages = torch.randint(num_blocks, (sum(counts), ), device=device, dtype=torch.int64)
    table = torch.zeros((case["batch"], max(counts)), device=device, dtype=torch.int32)
    offset = 0
    for sequence, count in enumerate(counts):
        table[sequence, :count] = pages[offset:offset + count]
        offset += count
    storage_heads = q_heads + 2 * kv_heads if case["q_layout"] == "qkv" else q_heads
    q_storage = random_tensor((sum(q_lens), storage_heads, dim))
    q = q_storage[:, :q_heads, :]
    if case["kv_layout"] == "interleaved":
        kv_storage = random_tensor((num_blocks, block, kv_heads, 2 * dim))
        k, v = kv_storage.split(dim, dim=-1)
    else:
        k = random_tensor((num_blocks, block, kv_heads, dim))
        v = random_tensor(k.shape)
    out = torch.empty(q.shape, device=device, dtype=out_dtype)
    cumulative = [0]
    for length in q_lens:
        cumulative.append(cumulative[-1] + length)
    segments = case["num_segments"]
    buffers = {}
    if max(q_lens) == 1:
        padded = 1 << (dim - 1).bit_length()
        buffers = {
            "softmax_segm_output": torch.empty((sum(q_lens), q_heads, segments, padded), device=device,
                                               dtype=torch.float32), "softmax_segm_max":
            torch.empty((sum(q_lens), q_heads, segments), device=device, dtype=torch.float32), "softmax_segm_expsum":
            torch.empty((sum(q_lens), q_heads, segments), device=device, dtype=torch.float32)
        }
    kwargs = {
        "q": q, "k": k, "v": v, "out": out, "cu_seqlens_q": torch.tensor(cumulative, device=device, dtype=torch.int32),
        "max_seqlen_q": max(q_lens), "seqused_k": torch.tensor(kv_lens, device=device, dtype=torch.int32),
        "max_seqlen_k": max(kv_lens), "softmax_scale": dim**-0.5, "causal": True, "window_size":
        (case["sliding_window"] - 1, 0) if case["sliding_window"] else (-1, -1), "block_table": table, "softcap":
        case["softcap"], "q_descale": None, "k_descale": None, "v_descale": None, "seq_threshold_3D":
        case["seq_threshold_3d"], "num_par_softmax_segments": segments, "use_td": True, **buffers
    }
    reference = torch.empty_like(out)
    ratio = q_heads // kv_heads
    offset = 0
    for sequence, (q_len, kv_len, count) in enumerate(zip(q_lens, kv_lens, counts)):
        keys = k[table[sequence, :count].long()].reshape(-1, kv_heads, dim)[:kv_len].float()
        values = v[table[sequence, :count].long()].reshape(-1, kv_heads, dim)[:kv_len].float()
        positions = torch.arange(kv_len, device=device)
        for head in range(kv_heads):
            for start in range(0, q_len, 64):
                end = min(q_len, start + 64)
                query = q[offset + start:offset + end, head * ratio:(head + 1) * ratio].float().transpose(0, 1)
                scores = torch.matmul(query, keys[:, head].T) * dim**-0.5
                if case["softcap"]:
                    scores = case["softcap"] * torch.tanh(scores / case["softcap"])
                query_positions = kv_len - q_len + torch.arange(start, end, device=device)
                mask = positions[None, :] <= query_positions[:, None]
                if case["sliding_window"]:
                    mask &= positions[None, :] > query_positions[:, None] - case["sliding_window"]
                probabilities = torch.softmax(scores.masked_fill(~mask[None], float("-inf")), dim=-1)
                result = torch.matmul(probabilities, values[:, head]).transpose(0, 1)
                reference[offset + start:offset + end, head * ratio:(head + 1) * ratio] = result
        offset += q_len
    return kwargs, reference


def check_output(torch, case, actual, reference):
    if case["dtype"].startswith("float8_"):
        # Match vLLM's FP8 accuracy test.
        atol, rtol = 0.15, 0.15
    elif case["out_dtype"] == "float16":
        atol, rtol = 0.003, 0.002
    else:
        atol, rtol = 0.025, 0.01
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    error = actual.float() - reference.float()
    relative_rmse = float((error.square().mean() / reference.float().square().mean().clamp_min(1e-20)).sqrt().item())
    if not math.isfinite(relative_rmse) or relative_rmse > 0.10:
        raise AssertionError(f"Relative RMS error {relative_rmse} exceeds 0.10")
    return {
        "atol": atol,
        "rtol": rtol,
        "relative_rmse": relative_rmse,
        "max_abs_error": float(error.abs().max().item()),
    }


def synchronize(torch, device):
    getattr(torch, device.type).synchronize(device)


def prepare_option(option, case, inputs, reference, env, device):
    torch, _, runtime, module = env
    config = runtime.AttentionConfig(**option["config"]) if option["config"] is not None else None
    phase = "launch"
    inputs["out"].fill_(float("nan"))
    try:
        with runtime.override_config(config):
            module.unified_attention(**inputs)
            synchronize(torch, device)
            option["key"] = asdict(runtime.last_resolution()[0])
        phase = "correctness"
        option["accuracy"] = check_output(torch, case, inputs["out"], reference)
        option["status"] = "ok"
    except Exception as error:  # pylint: disable=broad-exception-caught
        option.update(status=f"{phase}_error", error=f"{type(error).__name__}: {error}")
        synchronize(torch, device)


@contextmanager
def graph_timer(fn, torch, device, warmup_ms, rep_ms):
    """Use shorter graph batches for slow calls; time with boundary events."""
    backend = getattr(torch, device.type)
    started = time.perf_counter_ns()
    fn()
    backend.synchronize(device)
    eager_ms = max((time.perf_counter_ns() - started) / 1e6, 1e-6)
    slow = eager_ms >= 25
    if slow:
        calls = max(1, min(10, int(rep_ms / eager_ms)))
        warmup_replays = max(1, min(10000, math.ceil(warmup_ms / (calls * eager_ms))))
    else:
        for _ in range(4):
            fn()
        backend.synchronize(device)
        eager_ms = max((time.perf_counter_ns() - started) / 5e6, 1e-6)
        calls, warmup_replays = 10, 5
    graph = backend.XPUGraph()
    try:
        with backend.graph(graph):
            for _ in range(calls):
                fn()
        for _ in range(warmup_replays):
            graph.replay()
        backend.synchronize(device)
        start, end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
        start.record()
        end.record()
        backend.synchronize(device)

        def batch(repeats):
            backend.synchronize(device)
            start.record()
            for _ in range(repeats):
                graph.replay()
            end.record()
            backend.synchronize(device)
            event_ms = start.elapsed_time(end)
            if not math.isfinite(event_ms) or event_ms <= 0:
                raise RuntimeError(f"Invalid graph latency {event_ms}")
            return event_ms

        probe = 1 if slow else max(1, min(20, math.ceil(rep_ms / (calls * eager_ms))))
        estimate = batch(probe) / probe
        if not slow:
            warmup_replays = max(1, min(10000, math.ceil(warmup_ms / estimate)))
            for _ in range(warmup_replays):
                graph.replay()
            backend.synchronize(device)
        repeats = max(1, min(10000, math.ceil(rep_ms / estimate)))

        def elapsed():
            return batch(repeats) / (repeats * calls)

        yield graph, elapsed
    finally:
        graph.reset()


def benchmark_callable(fn, torch, device, warmup_ms, rep_ms, validate=None):
    """Time complete attention calls using graph replay and device events."""
    with graph_timer(fn, torch, device, warmup_ms, rep_ms) as (graph, elapsed):
        if validate is not None:
            validate(graph.replay)
        return elapsed()


def validate_replay(replay, torch, device, case, inputs, reference):
    inputs["out"].fill_(float("nan"))
    replay()
    synchronize(torch, device)
    return check_output(torch, case, inputs["out"], reference)


def time_options(options, case, inputs, env, device, rounds, warmup_ms, rep_ms, seed, field, *, reference):
    torch, _, runtime, module = env
    generator = random.Random(seed)
    for _ in range(rounds):
        order = [option for option in options if option["status"] == "ok"]
        generator.shuffle(order)
        for option in order:
            config = runtime.AttentionConfig(**option["config"]) if option["config"] is not None else None
            try:
                with runtime.override_config(config):

                    def validate(replay, result=option):
                        result["graph_accuracy"] = validate_replay(replay, torch, device, case, inputs, reference)

                    value = benchmark_callable(lambda: module.unified_attention(**inputs), torch, device, warmup_ms,
                                               rep_ms, validate)
                if not math.isfinite(value) or value <= 0:
                    raise RuntimeError(f"Invalid latency {value}")
                option.setdefault(field, []).append(value)
            except Exception as error:  # pylint: disable=broad-exception-caught
                option.update(status="timing_error", error=f"{type(error).__name__}: {error}")
                synchronize(torch, device)


def measure(args):
    env = torch_environment()
    torch, _, runtime, _ = env
    device = torch.device(args.device)
    getattr(torch, device.type).set_device(device)
    if device.type != "xpu":
        raise ValueError("The initial candidate pool and profiles target XPU")
    identity = runtime.get_profile_identity(device)
    manifest = normalize_workloads(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    data = {
        "format_version": FORMAT_VERSION,
        "complete": False,
        "identity": identity,
        "manifest": manifest,
        "settings": vars(args),
        "cases": [],
        "timing_scope": "attention_sequence_graph_device",
        "started_at": time.time(),
    }
    atomic_json(args.output, data)
    try:
        with torch.inference_mode():
            for index, case in enumerate(manifest):
                inputs, reference = allocate_case(case, torch, device, args.seed + index)
                fallback = {"id": "fallback", "config": None, "status": "pending", "samples_ms": []}
                row = {"id": case["id"], "workload": case, "key": None, "options": [fallback]}
                data["cases"].append(row)
                prepare_option(fallback, case, inputs, reference, env, device)
                if fallback["status"] != "ok":
                    raise RuntimeError(f"Fallback failed for {case['id']}: {fallback.get('error')}")
                row["key"] = fallback["key"]
                key = runtime.AttentionKey(**row["key"])  # pylint: disable=not-a-mapping
                for config in candidate_configs():
                    option = {"id": config_id(config), "config": config, "status": "pending", "samples_ms": []}
                    row["options"].append(option)
                    if not runtime.validate_config(runtime.AttentionConfig(**config), key, device.type):
                        option["status"] = "structurally_invalid"
                    else:
                        prepare_option(option, case, inputs, reference, env, device)
                time_options(
                    row["options"],
                    case,
                    inputs,
                    env,
                    device,
                    args.rounds,
                    args.warmup_ms,
                    args.rep_ms,
                    args.seed + index,
                    "samples_ms",
                    reference=reference,
                )
                atomic_json(args.output, data)
                print(f"Measured {index + 1}/{len(manifest)}: {case['id']}", flush=True)
                del inputs, reference
            for group_index, cases in enumerate(grouped_cases(data["cases"]).values()):
                selected = finalists(cases, args.finalists)
                for row in cases:
                    index = next(i for i, case in enumerate(manifest) if case["id"] == row["id"])
                    inputs, reference = allocate_case(row["workload"], torch, device, args.seed + index)
                    options = [option for option in row["options"] if option["id"] in selected]
                    for option in options:
                        prepare_option(option, row["workload"], inputs, reference, env, device)
                    time_options(
                        options,
                        row["workload"],
                        inputs,
                        env,
                        device,
                        args.confirmation_rounds,
                        args.warmup_ms,
                        args.rep_ms,
                        args.seed + 10000 + index,
                        "confirmation_ms",
                        reference=reference,
                    )
                    del inputs, reference
                atomic_json(args.output, data)
                print(f"Confirmed key {group_index + 1}", flush=True)
        data["complete"] = True
    except BaseException as error:
        data["session_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        data["finished_at"] = time.time()
        atomic_json(args.output, data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    measure_parser = modes.add_parser("measure", help="Measure candidates and confirm grouped finalists")
    measure_parser.add_argument("--manifest", required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument("--device", default="xpu:0")
    measure_parser.add_argument("--rounds", type=int, default=3)
    measure_parser.add_argument("--confirmation-rounds", type=int, default=5)
    measure_parser.add_argument("--finalists", type=int, default=3)
    measure_parser.add_argument("--warmup-ms", type=float, default=10)
    measure_parser.add_argument("--rep-ms", type=float, default=50)
    measure_parser.add_argument("--seed", type=int, default=1729)
    selection = modes.add_parser("select", help="Export one config per dynamic key")
    selection.add_argument("--measurements", required=True)
    selection.add_argument("--output-folder", required=True)
    selection.add_argument("--runtime-module", help="Config module file for CPU-only selection")
    selection.add_argument("--tolerance", type=float, default=0.01,
                           help="Worst-regret tie tolerance before geometric-mean tie-breaking")
    args = parser.parse_args(argv)
    if args.mode == "measure":
        for name in ("rounds", "warmup_ms", "rep_ms", "confirmation_rounds", "finalists"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    elif not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "select":
        result = select_results(
            json.loads(Path(args.measurements).read_text(encoding="utf-8")),
            runtime_module(args.runtime_module),
            args.output_folder,
            tolerance=args.tolerance,
        )
        print(f"Exported {result['accepted_keys']} keys")
        for path in result["profiles"].values():
            print(path)
    else:
        measure(args)


if __name__ == "__main__":
    main()
