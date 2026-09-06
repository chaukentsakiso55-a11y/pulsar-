from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / "data" / "pulsar-server.pid"
LOG_FILE = ROOT / "logs" / "pulsar-server.log"


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(quiet: bool = False) -> None:
    pid = _read_pid()
    if not pid:
        if not quiet:
            print("Pulsar server is not recorded as running.")
        return
    if not _pid_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        if not quiet:
            print("Removed stale Pulsar server PID file.")
        return

    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    if not quiet:
        print(f"Stopped Pulsar server PID {pid}.")


def start(mode: str) -> None:
    existing = _read_pid()
    if existing and _pid_alive(existing):
        print(f"Pulsar server is already running as PID {existing}. Stop it first.")
        raise SystemExit(1)
    if existing:
        PID_FILE.unlink(missing_ok=True)

    host = "127.0.0.1" if mode == "local" else "0.0.0.0"
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "pulsar.main:app",
        "--host",
        host,
        "--port",
        "8000",
    ]
    kwargs: dict = {"cwd": ROOT}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    else:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log = LOG_FILE.open("ab")
        kwargs["stdout"] = log
        kwargs["stderr"] = subprocess.STDOUT
        kwargs["start_new_session"] = True

    process = subprocess.Popen(cmd, **kwargs)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    time.sleep(0.8)
    if process.poll() is not None:
        PID_FILE.unlink(missing_ok=True)
        raise SystemExit("Pulsar server exited during startup.")
    print(f"Pulsar server started as PID {process.pid} on http://{host}:8000")


def status() -> None:
    pid = _read_pid()
    if pid and _pid_alive(pid):
        print(f"running:{pid}")
        return
    if pid:
        PID_FILE.unlink(missing_ok=True)
    print("stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pulsar AI server process controller")
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("mode", choices=["local", "server"])
    sub.add_parser("stop")
    sub.add_parser("status")
    args = parser.parse_args()

    if args.command == "start":
        start(args.mode)
    elif args.command == "stop":
        stop()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
