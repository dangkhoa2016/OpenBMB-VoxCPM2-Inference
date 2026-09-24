from __future__ import annotations

import argparse
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
from typing import Any

import httpx

from deploy.kaggle.model_mounts import discover_kaggle_model_candidates
from voxcpm_runtime.config import RuntimeConfig
from voxcpm_runtime.profiles import materialize_execution_profile


DEFAULT_TEXT = "Xin chào."
DEFAULT_PORT = 8090
_SUPPORTED_PROFILES = ("cpu", "cuda-single", "cuda-replica", "auto")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kaggle demo qualification Kaggle production demo.")
    parser.add_argument("--profile", choices=_SUPPORTED_PROFILES, default="auto")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    return parser.parse_args()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _gpu_process_rows() -> list[dict[str, Any]]:
    binary = shutil.which("nvidia-smi")
    if binary is None and Path("/opt/bin/nvidia-smi").is_file():
        binary = "/opt/bin/nvidia-smi"
    if binary is None:
        return []
    result = subprocess.run(
        [
            binary,
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            memory_mib = int(fields[2])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "gpu_uuid": fields[1],
                "used_memory_mib": memory_mib,
            }
        )
    return sorted(rows, key=lambda row: (row["gpu_uuid"], row["pid"]))


def _memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split()
        values[key] = int(raw)
    return values


def _pid_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _child_pids(parent_pid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=True,
        capture_output=True,
        text=True,
    )
    children: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, ppid = map(int, parts)
        if ppid == parent_pid:
            children.append(pid)
    return sorted(children)


def _wait_ready(port: int, timeout: float) -> tuple[dict[str, Any], float]:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1.0)
            if response.status_code == 200:
                return response.json(), time.monotonic() - started
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as error:
            last_error = type(error).__name__
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready: {last_error}")


def _request_tts(
    *,
    port: int,
    token: str,
    text: str,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    response = httpx.post(
        f"http://127.0.0.1:{port}/v1/tts",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Request-ID": "kaggle-production-demo",
        },
        json={"text": text},
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "status_code": response.status_code,
        "elapsed_seconds": round(elapsed, 3),
        "content_bytes": len(response.content),
        "request_id": response.headers.get("x-request-id"),
    }
    if response.status_code != 200:
        result["error_body"] = response.text[:500]
        return result
    with wave.open(io.BytesIO(response.content), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        result.update(
            {
                "sample_rate": rate,
                "channels": wav.getnchannels(),
                "frames": frames,
                "duration_seconds": round(frames / rate, 3),
                "sha256": hashlib.sha256(response.content).hexdigest(),
            }
        )
    return result


def _stop_process(process: subprocess.Popen[str], timeout: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def main() -> int:
    args = _parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    evidence_path = args.workspace / f"kaggle-{args.profile}-evidence.json"
    server_log_path = args.workspace / f"kaggle-{args.profile}-server.log"

    candidates = discover_kaggle_model_candidates()
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one qualified Kaggle model candidate, got {len(candidates)}")
    model = candidates[0]

    baseline_gpu = _gpu_process_rows()
    if _port_open(args.port):
        raise RuntimeError(f"port {args.port} is already in use")

    resolved = materialize_execution_profile(RuntimeConfig(profile=args.profile))
    token = secrets.token_urlsafe(48)
    env = os.environ.copy()
    env.update(
        {
            "VOXCPM_PROFILE": args.profile,
            "VOXCPM_MODEL_PATH": str(model.path),
            "VOXCPM_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VOXCPM_HOST": "127.0.0.1",
            "VOXCPM_PORT": str(args.port),
            "VOXCPM_REQUIRE_AUTH": "1",
            "VOXCPM_API_TOKEN": token,
            "VOXCPM_MAX_QUEUE_SIZE": "1",
            "VOXCPM_STREAM_IPC_MAX_CHUNKS": "4",
            "HF_ENDPOINT": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for key in ("VOXCPM_DEVICE", "VOXCPM_GPU_DEVICES", "VOXCPM_WORKERS"):
        env.pop(key, None)

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "qualification": "kaggle-production-demo",
        "project": "OpenBMB-VoxCPM2-Inference",
        "git_sha": _git_sha(),
        "profile": args.profile,
        "request_text": args.text,
        "model": {
            "model_id": model.model_id,
            "revision": model.revision,
            "metadata": dict(model.metadata),
        },
        "resolved_profile": {
            "device": resolved.device.value,
            "gpu_devices": list(resolved.gpu_devices or ()),
            "workers": resolved.workers,
        },
        "baseline": {
            "gpu_processes": baseline_gpu,
            "port_open": False,
            "memory_events": _memory_events(),
        },
        "runtime_network": {
            "offline_flags": True,
            "remote_endpoint_blackholed": True,
        },
    }

    server_log = server_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "scripts/serve.py"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        ready, startup_seconds = _wait_ready(args.port, args.startup_timeout)
        evidence["startup"] = {
            "elapsed_seconds": round(startup_seconds, 3),
            "readyz": ready,
            "server_pid": process.pid,
            "server_rss_bytes": _pid_rss_bytes(process.pid),
            "child_pids": _child_pids(process.pid),
            "gpu_processes": _gpu_process_rows(),
        }

        result = _request_tts(
            port=args.port,
            token=token,
            text=args.text,
            timeout=args.request_timeout,
        )
        evidence["request"] = result
        if result["status_code"] != 200:
            raise RuntimeError(f"TTS request failed with HTTP {result['status_code']}")
        if result.get("sample_rate") != 48_000 or result.get("channels") != 1:
            raise RuntimeError("unexpected WAV format")

        evidence["post_request"] = {
            "gpu_processes": _gpu_process_rows(),
            "server_rss_bytes": _pid_rss_bytes(process.pid),
            "child_pids": _child_pids(process.pid),
            "memory_events": _memory_events(),
        }
    finally:
        _stop_process(process)
        server_log.close()
        time.sleep(1.0)
        post_gpu = _gpu_process_rows()
        port_open_after = _port_open(args.port)
        evidence["cleanup"] = {
            "server_exit_code": process.returncode,
            "gpu_processes": post_gpu,
            "gpu_processes_match_baseline": post_gpu == baseline_gpu,
            "port_open": port_open_after,
            "memory_events": _memory_events(),
        }
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if not evidence["cleanup"]["gpu_processes_match_baseline"]:
        raise RuntimeError("GPU compute-process state did not return to baseline")
    if evidence["cleanup"]["port_open"]:
        raise RuntimeError("API port remained open after shutdown")

    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"KAGGLE_PROFILE_{args.profile.upper().replace('-', '_')}=PASS")
    print(f"KAGGLE_EVIDENCE={evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
