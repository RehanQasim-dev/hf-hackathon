#!/usr/bin/env python3
"""Validate and benchmark pinned SmolVLM2 image/video inference on ET-SoC1."""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark_config_helpers import load_config
from run_llama_server_benchmark import (
    ensure_llama_cpp_build,
    is_dir,
    is_file,
    materialize_artifact,
    post_completion,
    score_common,
    sha256_file,
    terminate,
    wait_ready,
    write_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".github" / "ci" / "benchmark_config.json"
REQUEST_PATH = "/completion"


def normalize_answer(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def unsupported_vision_ops(log: str) -> set[str]:
    return set(re.findall(r"^warmup:\s+([A-Z][A-Z0-9_]+):\s+type\s*=", log, re.MULTILINE))


def log_failures(log: str, *, mode: str, request_count: int, allowed_ops: set[str]) -> list[str]:
    failures: list[str] = []
    observed_requests = log.count(f"done request: POST {REQUEST_PATH}")
    if observed_requests != request_count:
        failures.append(f"{mode}: observed {observed_requests}/{request_count} completed requests")

    if mode == "host":
        if "CLIP using CPU backend" not in log:
            failures.append("host: CPU vision backend not observed")
        return failures

    if "using device ET" not in log and "ET device 0" not in log:
        failures.append("board: ET device use not observed")
    if "CLIP using ET backend" not in log:
        failures.append("board: ET vision backend not observed")
    match = re.search(r"offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU", log)
    if not match:
        failures.append("board: LLM layer offload summary not observed")
    elif match.group(1) != match.group(2):
        failures.append(f"board: expected full LLM offload, got {match.group(1)}/{match.group(2)}")

    observed_ops = unsupported_vision_ops(log)
    unexpected = observed_ops - allowed_ops
    if unexpected:
        failures.append("board: unexpected vision fallback ops: " + ", ".join(sorted(unexpected)))
    return failures


def build_command(
    server_bin: Path,
    model_path: Path,
    mmproj_path: Path,
    cfg: dict[str, Any],
    *,
    mode: str,
) -> list[str]:
    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--host",
        str(cfg.get("host", "127.0.0.1")),
        "--port",
        str(cfg.get("port", 18107)),
        "-c",
        str(cfg.get("ctx_size", 2048)),
        "-b",
        str(cfg.get("batch_size", 256)),
        "-ub",
        str(cfg.get("ubatch_size", 128)),
        "-np",
        str(cfg.get("parallel", 1)),
        "--cache-ram",
        str(cfg.get("cache_ram_mib", 0)),
        "--no-warmup",
    ]
    if mode == "host":
        cmd.extend(["--no-mmproj-offload", "-dev", "none", "-ngl", "0"])
    else:
        cmd.extend(
            [
                "--mmproj-offload",
                "-dev",
                str(cfg.get("device", "ET")),
                "-ngl",
                str(cfg.get("gpu_layers", 99)),
            ]
        )
    cmd.extend(str(value) for value in cfg.get("extra_args", []))
    return cmd


def make_request(case: dict[str, Any], media: list[Path], cfg: dict[str, Any]) -> dict[str, Any]:
    markers = "".join("<__media__>" for _ in media)
    prompt = str(cfg["prompt_template"]).format(
        media_markers=markers,
        question=str(case["question"]),
    )
    return {
        "prompt": {
            "prompt_string": prompt,
            "multimodal_data": [base64.b64encode(path.read_bytes()).decode("ascii") for path in media],
        },
        "n_predict": int(cfg.get("max_tokens", 8)),
        "temperature": cfg.get("temperature", 0),
        "top_k": int(cfg.get("top_k", 1)),
    }


def run_mode(
    *,
    mode: str,
    cmd: list[str],
    server_cfg: dict[str, Any],
    multimodal_cfg: dict[str, Any],
    cases: list[tuple[dict[str, Any], list[Path]]],
    workdir: Path,
    run_dir: Path,
    server_bin: Path,
) -> tuple[list[dict[str, Any]], list[str], set[str]]:
    log_path = run_dir / f"server-{mode}.log"
    command_path = run_dir / f"command-{mode}.json"
    response_path = run_dir / f"responses-{mode}.json"
    command_path.write_text(json.dumps(cmd, indent=2) + "\n")

    env = os.environ.copy()
    env["TMPDIR"] = env.get("TMPDIR", "/dev/shm")
    env["LD_LIBRARY_PATH"] = os.environ.get("LLAMA_CPP_ET_LD_LIBRARY_PATH", str(server_bin.parent))
    if server_cfg.get("flash_attn") is False:
        env["LLAMA_ARG_FLASH_ATTN"] = "off"
    if mode == "board":
        env["MTMD_BACKEND_DEVICE"] = str(server_cfg.get("device", "ET"))
    else:
        env.pop("MTMD_BACKEND_DEVICE", None)

    proc: subprocess.Popen[bytes] | None = None
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        with log_path.open("wb") as log:
            log.write(("$ " + " ".join(cmd) + "\n\n").encode())
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=workdir, env=env)
            ready, note = wait_ready(proc, log_path, int(server_cfg.get("ready_timeout_s", 180)))
            if not ready:
                return [], [f"{mode}: {note}"], set()

            url = f"http://{server_cfg.get('host', '127.0.0.1')}:{server_cfg.get('port', 18107)}{REQUEST_PATH}"
            for case, media in cases:
                payload = make_request(
                    case,
                    media,
                    {
                        **multimodal_cfg,
                        "max_tokens": server_cfg.get("max_tokens", 8),
                        "temperature": server_cfg.get("temperature", 0),
                        "top_k": server_cfg.get("top_k", 1),
                    },
                )
                started = time.monotonic()
                status, response = post_completion(url, payload, int(server_cfg.get("request_timeout_s", 240)))
                elapsed_s = time.monotonic() - started
                content = str(response.get("content") or "")
                result = {
                    "case": str(case["name"]),
                    "status": status,
                    "content": content,
                    "normalized_answer": normalize_answer(content),
                    "elapsed_s": elapsed_s,
                    "frame_count": len(media),
                    "tokens_predicted": response.get("tokens_predicted"),
                    "tokens_evaluated": response.get("tokens_evaluated"),
                    "timings": response.get("timings", {}),
                }
                results.append(result)
                if status != 200:
                    failures.append(f"{mode}/{case['name']}: HTTP {status}: {response.get('error', '')}")
                if not result["normalized_answer"]:
                    failures.append(f"{mode}/{case['name']}: empty answer")
    except Exception as exc:
        failures.append(f"{mode}: server benchmark error: {exc}")
    finally:
        terminate(proc)

    response_path.write_text(json.dumps(results, indent=2) + "\n")
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    allowed_ops = {str(value) for value in server_cfg.get("allowed_vision_fallback_ops", [])}
    failures.extend(log_failures(log, mode=mode, request_count=len(cases), allowed_ops=allowed_ops))
    return results, failures, unsupported_vision_ops(log)


def correctness_failures(
    cases: list[tuple[dict[str, Any], list[Path]]],
    host_results: list[dict[str, Any]],
    board_results: list[dict[str, Any]],
    order_pair: list[str],
) -> list[str]:
    failures: list[str] = []
    host = {str(item["case"]): item for item in host_results}
    board = {str(item["case"]): item for item in board_results}
    for case, _ in cases:
        name = str(case["name"])
        accepted = {normalize_answer(str(value)) for value in case["accepted_answers"]}
        if name not in host or name not in board:
            failures.append(f"{name}: missing host or board result")
            continue
        host_answer = str(host[name]["normalized_answer"])
        board_answer = str(board[name]["normalized_answer"])
        if host_answer not in accepted:
            failures.append(f"{name}: host answer {host_answer!r} is not accepted")
        if board_answer not in accepted:
            failures.append(f"{name}: board answer {board_answer!r} is not accepted")
        if board_answer != host_answer:
            failures.append(f"{name}: board answer {board_answer!r} != host reference {host_answer!r}")

    if len(order_pair) == 2 and all(name in board for name in order_pair):
        first = str(board[order_pair[0]]["normalized_answer"])
        second = str(board[order_pair[1]]["normalized_answer"])
        if first == second:
            failures.append(f"image-order check failed: both cases returned {first!r}")
    return failures


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(
    contract: dict[str, Any],
    mcfg: dict[str, Any],
    model_path: Path,
    mmproj_path: Path,
    media_paths: dict[str, Path],
) -> None:
    multimodal = mcfg["multimodal"]
    input_contract = contract["input_contract"]
    score_cfg = mcfg.get("score", {})
    performance = contract["performance"]
    if score_cfg.get("metric") != performance["metric"] or score_cfg.get("higher_is_better") is not False:
        raise ValueError("leaderboard metric differs from the protected contract")
    if mcfg.get("canonical_variant") != "SmolVLM2-500M-Video-Instruct-Q8_0":
        raise ValueError("canonical variant differs from the protected model")
    allowed_ops = [str(value) for value in mcfg["llama_server"].get("allowed_vision_fallback_ops", [])]
    if allowed_ops != contract["correctness"]["allowed_vision_fallback_ops"]:
        raise ValueError("allowed vision fallback ops differ from the protected contract")
    if float(multimodal.get("frames_per_second", 0)) != float(input_contract["frames_per_second"]):
        raise ValueError("frames_per_second differs from the protected contract")
    for key in ("prompt_template", "cases", "order_pair"):
        if multimodal.get(key) != input_contract.get(key):
            raise ValueError(f"multimodal {key} differs from the protected contract")

    converted = contract["conversion"]["artifacts"]
    for path in (model_path, mmproj_path):
        expected = converted.get(path.name)
        if not expected:
            raise ValueError(f"{path.name} is not a protected converted artifact")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(f"{path.name} sha256 {actual} != protected {expected['sha256']}")

    fixture_contract = input_contract["fixtures"]
    for artifact_id, path in media_paths.items():
        expected = fixture_contract.get(artifact_id)
        if not expected:
            raise ValueError(f"{artifact_id} is not a protected fixture")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise ValueError(f"{artifact_id} sha256 {actual} != protected {expected['sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["models"][args.model]
    server_cfg = mcfg["llama_server"]
    multimodal_cfg = mcfg["multimodal"]
    run_dir = Path(args.results_dir)
    score_path = Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    score = score_common(args.model, str(mcfg["canonical_variant"]))

    try:
        contract_path = REPO_ROOT / str(mcfg["reference_contract"])
        contract = json.loads(contract_path.read_text())
        server_bin = materialize_artifact(mcfg, str(server_cfg["server_artifact"]))
        model_path = materialize_artifact(mcfg, str(server_cfg["model_artifact"]))
        mmproj_path = materialize_artifact(mcfg, str(server_cfg["mmproj_artifact"]))
        workdir = materialize_artifact(mcfg, str(server_cfg["workdir_artifact"]))
        media_paths = {
            str(artifact): materialize_artifact(mcfg, str(artifact))
            for case in multimodal_cfg["cases"]
            for artifact in case["media_artifacts"]
        }
        cases = [
            (
                case,
                [media_paths[str(artifact)] for artifact in case["media_artifacts"]],
            )
            for case in multimodal_cfg["cases"]
        ]
        validate_contract(contract, mcfg, model_path, mmproj_path, media_paths)
        ensure_llama_cpp_build(mcfg, server_cfg, server_bin, None, workdir)
    except Exception as exc:
        note = f"artifact setup failed: {exc}"
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0

    missing = [str(path) for path in [server_bin, model_path, mmproj_path] if not is_file(path)]
    missing.extend(str(path) for _, media in cases for path in media if not is_file(path))
    if not is_dir(workdir):
        missing.append(str(workdir))
    if missing:
        note = "missing required files: " + ", ".join(missing)
        score.update({"status": "fail", "note": note, "valid_note": note})
        write_score(score_path, score)
        return 0

    board_lock = Path(os.environ.get("BOARD_LOCK", "/var/lock/etsoc-shire0.lock"))
    board_lock.parent.mkdir(parents=True, exist_ok=True)
    with board_lock.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        host_results, host_failures, _ = run_mode(
            mode="host",
            cmd=build_command(server_bin, model_path, mmproj_path, server_cfg, mode="host"),
            server_cfg=server_cfg,
            multimodal_cfg=multimodal_cfg,
            cases=cases,
            workdir=workdir,
            run_dir=run_dir,
            server_bin=server_bin,
        )
        board_results, board_failures, fallback_ops = run_mode(
            mode="board",
            cmd=build_command(server_bin, model_path, mmproj_path, server_cfg, mode="board"),
            server_cfg=server_cfg,
            multimodal_cfg=multimodal_cfg,
            cases=cases,
            workdir=workdir,
            run_dir=run_dir,
            server_bin=server_bin,
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    failures = host_failures + board_failures
    failures.extend(
        correctness_failures(
            cases,
            host_results,
            board_results,
            [str(value) for value in multimodal_cfg.get("order_pair", [])],
        )
    )
    total_board_s = sum(float(item["elapsed_s"]) for item in board_results)
    total_frames = sum(int(item["frame_count"]) for item in board_results)
    fps = float(multimodal_cfg.get("frames_per_second", 1.0))
    represented_s = total_frames / fps if fps > 0 else 0.0
    real_time_factor = total_board_s / represented_s if represented_s > 0 else None
    accuracy = (
        sum(
            str(item["normalized_answer"])
            in {normalize_answer(str(value)) for value in case["accepted_answers"]}
            for case, item in zip((case for case, _ in cases), board_results)
        )
        / len(cases)
        if len(board_results) == len(cases) and cases
        else 0.0
    )
    host_agreement = (
        sum(
            host["normalized_answer"] == board["normalized_answer"]
            for host, board in zip(host_results, board_results)
        )
        / len(cases)
        if len(host_results) == len(board_results) == len(cases) and cases
        else 0.0
    )
    predicted_tokens = sum(int(item.get("tokens_predicted") or 0) for item in board_results)
    predicted_ms = sum(float(item.get("timings", {}).get("predicted_ms") or 0.0) for item in board_results)
    decode_tps = predicted_tokens / (predicted_ms / 1000.0) if predicted_ms > 0 else None
    note = "; ".join(failures) if failures else (
        f"{len(cases)} public image cases matched the pinned CPU reference; "
        f"vision fallback ops: {', '.join(sorted(fallback_ops)) or 'none'}"
    )
    score.update(
        {
            "status": "fail" if failures else "pass",
            "passed": not failures,
            "real_time_factor": real_time_factor,
            "mean_end_to_end_s": total_board_s / len(board_results) if board_results else None,
            "elapsed_s": total_board_s,
            "represented_duration_s": represented_s,
            "vision_frames": total_frames,
            "task_accuracy": accuracy,
            "host_agreement": host_agreement,
            "tokens_per_second": decode_tps,
            "vision_fallback_ops": sorted(fallback_ops),
            "validation_contract_sha256": contract_sha256(contract_path),
            "valid_dump": not failures,
            "valid_note": note,
            "note": note,
        }
    )
    write_score(score_path, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
