from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path
from statistics import median
from typing import Any

import httpx

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deploy.kaggle.model_mounts import discover_kaggle_model_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen benchmark qualification benchmark protocol.")
    parser.add_argument("--profile", choices=("cpu", "cuda-single", "cuda-replica"), required=True)
    parser.add_argument("--protocol", type=Path, default=Path("benchmarks/protocol-v1.json"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser.parse_args()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def gpu_rows() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if binary is None and Path("/opt/bin/nvidia-smi").is_file():
        binary = "/opt/bin/nvidia-smi"
    if binary is None:
        return []
    result = subprocess.run(
        [binary, "--query-compute-apps=pid,gpu_uuid,used_memory",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    rows: list[dict[str, Any]] = []
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append({
                "pid": int(parts[0]),
                "gpu_uuid": parts[1],
                "used_memory_mib": int(parts[2]),
            })
        except ValueError:
            continue
    return sorted(rows, key=lambda row: (row["gpu_uuid"], row["pid"]))


def memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def cgroup_limit() -> int | None:
    path = Path("/sys/fs/cgroup/memory.max")
    if not path.is_file():
        return None
    raw = path.read_text().strip()
    return None if raw == "max" else int(raw)


def proc_rss(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def child_pids(parent_pid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="], check=True, capture_output=True, text=True
    )
    return sorted(
        int(pid)
        for line in result.stdout.splitlines()
        for pid, ppid in [line.split()]
        if int(ppid) == parent_pid
    )


def wait_ready(port: int, timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1.0)
            if response.status_code == 200:
                return response.json(), time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise RuntimeError("server did not become ready")


def wav_metrics(payload: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        channels = wav.getnchannels()
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sample_rate": rate,
        "channels": channels,
        "frames": frames,
        "audio_seconds": frames / rate,
    }


def one_shot(
    *, port: int, token: str, request_id: str, text: str, timeout: float
) -> dict[str, Any]:
    started = time.perf_counter()
    response = httpx.post(
        f"http://127.0.0.1:{port}/v1/tts",
        headers={"Authorization": f"Bearer {token}", "X-Request-ID": request_id},
        json={"text": text},
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "request_id": request_id,
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
    }
    if response.status_code == 200:
        result.update(wav_metrics(response.content))
        result["rtf"] = elapsed / result["audio_seconds"]
    else:
        result["error"] = response.text[:500]
    return result


def stream(
    *, port: int, token: str, request_id: str, text: str, timeout: float
) -> dict[str, Any]:
    started = time.perf_counter()
    first: float | None = None
    total = 0
    chunks = 0
    status = None
    headers: dict[str, str] = {}
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{port}/v1/tts/stream",
        headers={"Authorization": f"Bearer {token}", "X-Request-ID": request_id},
        json={"text": text},
        timeout=timeout,
    ) as response:
        status = response.status_code
        headers = dict(response.headers)
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            if first is None:
                first = time.perf_counter() - started
            chunks += 1
            total += len(chunk)
    completion = time.perf_counter() - started
    sample_rate = int(headers.get("x-audio-sample-rate", "48000"))
    channels = int(headers.get("x-audio-channels", "1"))
    audio_seconds = total / 2 / sample_rate / channels if status == 200 else None
    return {
        "request_id": request_id,
        "status_code": status,
        "first_body_seconds": first,
        "completion_seconds": completion,
        "chunks": chunks,
        "pcm_bytes": total,
        "sample_rate": sample_rate,
        "channels": channels,
        "audio_seconds": audio_seconds,
        "rtf": completion / audio_seconds if audio_seconds else None,
    }


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(item["elapsed_seconds"]) for item in values]
    rtf = [float(item["rtf"]) for item in values]
    audio = [float(item["audio_seconds"]) for item in values]
    return {
        "repetitions": len(values),
        "elapsed_seconds": {
            "min": min(elapsed),
            "median": median(elapsed),
            "max": max(elapsed),
        },
        "rtf": {"min": min(rtf), "median": median(rtf), "max": max(rtf)},
        "audio_seconds": {
            "min": min(audio),
            "median": median(audio),
            "max": max(audio),
        },
    }


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    inputs = protocol["inputs"]
    policy = protocol["warmup_policy"]
    args.workspace.mkdir(parents=True, exist_ok=True)
    evidence_path = args.workspace / f"benchmark-{args.profile}.json"
    log_path = args.workspace / f"benchmark-{args.profile}-server.log"

    candidates = discover_kaggle_model_candidates()
    if len(candidates) != 1:
        raise RuntimeError(f"expected one Kaggle model candidate, got {len(candidates)}")
    model = candidates[0]

    baseline_gpu = gpu_rows()
    baseline_events = memory_events()
    if port_open(args.port):
        raise RuntimeError(f"port {args.port} is already open")

    token = secrets.token_urlsafe(48)
    env = os.environ.copy()
    env.update({
        "VOXCPM_PROFILE": args.profile,
        "VOXCPM_MODEL_PATH": str(model.path),
        "VOXCPM_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VOXCPM_HOST": "127.0.0.1",
        "VOXCPM_PORT": str(args.port),
        "VOXCPM_REQUIRE_AUTH": "1",
        "VOXCPM_API_TOKEN": token,
        "VOXCPM_MAX_QUEUE_SIZE": "2",
        "VOXCPM_STREAM_IPC_MAX_CHUNKS": "4",
        "HF_ENDPOINT": "http://127.0.0.1:9",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    })
    for key in ("VOXCPM_DEVICE", "VOXCPM_GPU_DEVICES", "VOXCPM_WORKERS"):
        env.pop(key, None)

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "source_sha": git_sha(),
        "profile": args.profile,
        "model": {
            "model_id": model.model_id,
            "revision": model.revision,
            "metadata": dict(model.metadata),
        },
        "host": {
            "python": sys.version.split()[0],
            "cgroup_memory_limit_bytes": cgroup_limit(),
        },
        "baseline": {
            "gpu_processes": baseline_gpu,
            "memory_events": baseline_events,
        },
        "cold_start": None,
        "warmup": [],
        "warm_runs": {},
        "streaming": None,
        "concurrency": None,
        "resources": {},
        "cleanup": None,
    }

    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "scripts.serve"],
        cwd=_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        ready, startup = wait_ready(args.port, args.startup_timeout)
        evidence["cold_start"] = {
            "startup_seconds": startup,
            "readyz": ready,
            "server_pid": process.pid,
            "server_rss_bytes": proc_rss(process.pid),
            "child_pids": child_pids(process.pid),
            "gpu_processes": gpu_rows(),
        }

        for i in range(int(policy["warmup_requests"])):
            result = one_shot(
                port=args.port, token=token, request_id=f"benchmark-warmup-{i}",
                text=inputs["B2"]["text"], timeout=args.request_timeout,
            )
            if result["status_code"] != 200:
                raise RuntimeError("warmup failed")
            evidence["warmup"].append(result)

        repetitions = int(policy["measured_warm_repetitions"])
        for benchmark_id in ("B1", "B2", "B3"):
            runs = []
            for i in range(repetitions):
                result = one_shot(
                    port=args.port, token=token,
                    request_id=f"benchmark-{benchmark_id.lower()}-{i}",
                    text=inputs[benchmark_id]["text"], timeout=args.request_timeout,
                )
                if result["status_code"] != 200:
                    raise RuntimeError(f"{benchmark_id} failed")
                runs.append(result)
            evidence["warm_runs"][benchmark_id] = {
                "input": inputs[benchmark_id],
                "runs": runs,
                "summary": summarize(runs),
            }

        if args.profile == "cuda-single":
            result = stream(
                port=args.port, token=token, request_id="benchmark-stream",
                text=inputs["B7"]["text"], timeout=args.request_timeout,
            )
            if result["status_code"] != 200:
                raise RuntimeError("B7 streaming failed")
            evidence["streaming"] = result

        b8 = inputs["B8"]["requests"]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    one_shot,
                    port=args.port,
                    token=token,
                    request_id=item["request_id"],
                    text=item["text"],
                    timeout=args.request_timeout,
                )
                for item in b8
            ]
            concurrent_results = [future.result() for future in futures]
        makespan = time.perf_counter() - started
        if any(item["status_code"] != 200 for item in concurrent_results):
            raise RuntimeError("B8 concurrency failed")
        total_audio = sum(float(item["audio_seconds"]) for item in concurrent_results)
        evidence["concurrency"] = {
            "requests": concurrent_results,
            "makespan_seconds": makespan,
            "completed_requests": len(concurrent_results),
            "requests_per_second": len(concurrent_results) / makespan,
            "audio_seconds_per_second": total_audio / makespan,
        }

        evidence["resources"] = {
            "server_rss_bytes": proc_rss(process.pid),
            "child_pids": child_pids(process.pid),
            "gpu_processes": gpu_rows(),
            "memory_events": memory_events(),
        }
    finally:
        stop(process)
        log.close()
        time.sleep(1)
        post_gpu = gpu_rows()
        evidence["cleanup"] = {
            "gpu_processes": post_gpu,
            "gpu_processes_match_baseline": post_gpu == baseline_gpu,
            "port_open": port_open(args.port),
            "memory_events": memory_events(),
        }
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if evidence["cleanup"]["port_open"]:
        raise RuntimeError("port leak after benchmark")
    if not evidence["cleanup"]["gpu_processes_match_baseline"]:
        raise RuntimeError("GPU process state did not return to baseline")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"BENCHMARK_{args.profile.upper().replace('-', '_')}_BENCHMARK=PASS")
    print(f"BENCHMARK_EVIDENCE={evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
