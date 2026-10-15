"""Parallel query execution with a watchdog that kills stuck workers.

SQLite's progress handler aborts most long queries cooperatively (see runner.py), but a
single very long VDBE step (a huge sort, for instance) can overrun it. Each worker
therefore publishes the start time of its current task and the parent terminates and
replaces any worker that exceeds ``timeout_s + grace_s``; the task is recorded as a timeout.
Timestamps use ``time.monotonic()`` (system-wide on Linux and macOS) so that a laptop going
to sleep does not make the parent kill healthy workers.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Iterable

from tqdm import tqdm

from mexql.execution.runner import ExecResult, execute_query


@dataclass(frozen=True)
class ExecTask:
    key: str
    db_path: str
    sql: str


def _worker(worker_id: int, task_q, result_q, start_times, current_idx, timeout_s: float, max_rows: int) -> None:
    while True:
        item = task_q.get()
        if item is None:
            break
        idx, db_path, sql = item
        start_times[worker_id] = time.monotonic()
        current_idx[worker_id] = idx
        try:
            res = execute_query(db_path, sql, timeout_s=timeout_s, max_rows=max_rows)
        except Exception as exc:  # pragma: no cover - defensive
            res = ExecResult("sql_error", error=f"worker exception: {exc!r}")
        current_idx[worker_id] = -1
        result_q.put((idx, res))


def run_parallel(tasks: Iterable[ExecTask], timeout_s: float = 30.0, max_rows: int = 500_000,
                 num_workers: int = 4, grace_s: float = 5.0, desc: str = "executing") -> dict[str, ExecResult]:
    """Execute tasks in worker processes; returns results keyed by ``ExecTask.key``."""
    tasks = list(tasks)
    if not tasks:
        return {}
    num_workers = max(1, min(num_workers, len(tasks)))
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    start_times = ctx.Array("d", [0.0] * num_workers)
    current_idx = ctx.Array("l", [-1] * num_workers)

    for idx, t in enumerate(tasks):
        task_q.put((idx, t.db_path, t.sql))
    for _ in range(num_workers):
        task_q.put(None)

    def spawn(worker_id: int):
        p = ctx.Process(target=_worker, args=(worker_id, task_q, result_q, start_times, current_idx, timeout_s, max_rows),
                        daemon=True)
        p.start()
        return p

    workers = [spawn(i) for i in range(num_workers)]
    results: dict[int, ExecResult] = {}
    bar = tqdm(total=len(tasks), desc=desc, unit="q", leave=False)
    try:
        while len(results) < len(tasks):
            try:
                idx, res = result_q.get(timeout=0.25)
                if idx not in results:
                    results[idx] = res
                    bar.update(1)
                continue
            except queue.Empty:
                pass
            now = time.monotonic()
            for i, proc in enumerate(workers):
                idx = current_idx[i]
                if idx < 0:  # idle, or exited normally after its sentinel
                    continue
                stuck = now - start_times[i] > timeout_s + grace_s
                if stuck or not proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2.0)
                    if idx not in results:
                        reason = "killed: exceeded timeout + grace" if stuck else "worker died"
                        results[idx] = ExecResult("timeout" if stuck else "sql_error", error=reason)
                        bar.update(1)
                    current_idx[i] = -1
                    workers[i] = spawn(i)
    finally:
        bar.close()
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for proc in workers:
            proc.join(timeout=2.0)
        result_q.close()
        task_q.close()
    return {tasks[idx].key: res for idx, res in results.items()}
