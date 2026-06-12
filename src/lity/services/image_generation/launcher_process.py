from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


def process_is_running(process: Any) -> bool:
    try:
        return process.poll() is None
    except Exception:
        return False


def terminate_process(process: Any) -> None:
    if os.name != "nt" and getattr(process, "pid", None):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            return
        except Exception:
            pass
    process.terminate()


def kill_process(process: Any) -> None:
    if os.name != "nt" and getattr(process, "pid", None):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            return
        except Exception:
            pass
    process.kill()


def pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_pid(pid: int) -> None:
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            return
        except Exception:
            pass
    os.kill(pid, signal.SIGTERM)


def kill_pid(pid: int) -> None:
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except Exception:
            pass
    os.kill(pid, signal.SIGKILL)


def wait_for_pid_exit(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if reap_exited_child(pid):
            return
        if not pid_is_running(pid):
            return
        time.sleep(0.05)
    raise subprocess.TimeoutExpired(str(pid), timeout)


def reap_exited_child(pid: int) -> bool:
    if os.name == "nt":
        return False
    try:
        finished_pid, _status = os.waitpid(pid, os.WNOHANG)
        return finished_pid == pid
    except ChildProcessError:
        return False
    except Exception:
        return False
