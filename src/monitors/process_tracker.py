# src/monitors/process_tracker.py

import os
import psutil


def get_processes_in_group(pgid: int):
    """
    Returns a list of psutil.Process objects that belong to the given process group ID (PGID).
    """

    processes = []

    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if os.getpgid(proc.info["pid"]) == pgid:
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            # Process might have terminated or we don't have permission
            continue

    return processes


def get_process_tree(root_pid: int):
    """
    Returns the root process and all its children (recursive).
    Useful if PGID is unreliable or for debugging.
    """

    try:
        root = psutil.Process(root_pid)
    except psutil.NoSuchProcess:
        return []

    processes = [root]

    try:
        children = root.children(recursive=True)
        processes.extend(children)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return processes


def is_process_alive(pid: int) -> bool:
    """
    Check if a process is alive.
    """

    return psutil.pid_exists(pid)
