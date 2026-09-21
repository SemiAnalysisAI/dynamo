#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, reconnectable Slurm control. No workload executes on the login host."""

import argparse
import contextlib
import fcntl
import getpass
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path

from barite_common import (
    atomic_json,
    read_json,
    validate_run_key,
    validate_sha,
    verify_controller,
)

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def parse_job_id(line):
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", line.strip())
    if not match:
        raise ValueError("malformed sbatch job receipt")
    return match[1]


def parse_record(text):
    return dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", text))


def identity_matches(record, request, job_id=None):
    return (
        record.get("JobName") == "dynamo-" + request["run_key"]
        and record.get("Comment") == request["run_key"]
        and record.get("UserId", "").split("(")[0] == getpass.getuser()
        and (job_id is None or record.get("JobId") == str(job_id))
    )


def command(args, timeout=20):
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def load_request(run):
    req = read_json(run / "request.json")
    validate_run_key(req["run_key"])
    for key in ("source_sha", "controller_sha"):
        validate_sha(req[key])
    for key in ("archive_sha256", "controller_bundle_sha256"):
        validate_sha(req[key], 64)
    if req.get("reuse_image_sha"):
        validate_sha(req["reuse_image_sha"], 64)
    expected = {
        "schema_version": 1,
        "suite": "aggregate",
        "gpus": 1,
        "cpus": 16,
        "mem_gib": 64,
        "time_limit_minutes": 240,
        "queue_timeout_seconds": 1800,
        "controller_timeout_seconds": 17100,
    }
    if req.get("partition") not in ("auto", "compute-0", "compute-1"):
        raise ValueError("unsupported partition policy")
    if any(req.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported or unbounded allocation request")
    if (
        run.name != req["run_key"]
        or run.parent != Path.home() / "dynamo-rocm-ci" / "runs"
    ):
        raise ValueError("run directory must be the canonical per-user run path")
    if req.get("expected_uid") is not None and os.getuid() != req["expected_uid"]:
        raise ValueError("remote account UID mismatch")
    verify_controller(run)
    return req


@contextlib.contextmanager
def lock(run):
    with (run / "controller.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def inspect_job(run, req, job_id, timeout=20):
    result = command(["scontrol", "show", "job", "-o", str(job_id)], timeout=timeout)
    if result.returncode:
        return None
    record = parse_record(result.stdout)
    if not identity_matches(record, req, job_id):
        raise ValueError("scheduler identity does not match run; refusing to act")
    state = record.get("JobState", "").split("+")[0]
    evidence = {
        "job_id": str(job_id),
        "run_key": req["run_key"],
        "state": state,
        "exit_code": record.get("ExitCode"),
        "restarts": int(record.get("Restarts", "-1")),
        "observed_at": time.time(),
        "record": record,
    }
    atomic_json(run / "scheduler-latest.json", evidence)
    if state in TERMINAL:
        atomic_json(run / "terminal.json", evidence)
    return evidence


def recover(run, req):
    result = command(
        [
            "squeue",
            "--noheader",
            "--user",
            getpass.getuser(),
            "--name",
            "dynamo-" + req["run_key"],
            "--format=%A",
        ]
    )
    if result.returncode:
        raise RuntimeError("cannot query scheduler for submission recovery")
    candidates = []
    for job_id in set(result.stdout.split()):
        parse_job_id(job_id)
        if inspect_job(run, req, job_id):
            candidates.append(job_id)
    if len(candidates) > 1:
        raise RuntimeError("multiple matching jobs; refusing ambiguous recovery")
    if candidates:
        receipt = {
            "job_id": candidates[0],
            "run_key": req["run_key"],
            "source_sha": req["source_sha"],
            "recovered": True,
        }
        atomic_json(run / "receipt.json", receipt)
        return receipt
    return None


def cancel(run, req, job_id):
    evidence = inspect_job(run, req, job_id)
    if evidence is None:
        raise RuntimeError("cannot verify job identity for cancellation")
    if evidence["state"] not in TERMINAL:
        result = command(["scancel", str(job_id)])
        atomic_json(
            run / "cancel.json",
            {"job_id": str(job_id), "returncode": result.returncode, "at": time.time()},
        )
        if result.returncode:
            raise RuntimeError("scancel failed")


def snapshot(run):
    result = {"run_key": run.name, "active": False, "phase": "staged"}
    for name in ("receipt", "wait-result", "terminal", "controller-result", "monitor"):
        path = run / (name + ".json")
        if path.exists():
            result[name.replace("-", "_")] = read_json(path)
    if "controller_result" in result:
        result["phase"] = result["controller_result"]["status"]
    elif "terminal" in result:
        result.update(active=True, phase="awaiting-waiter")
    elif "monitor" in result or "receipt" in result:
        result.update(active=True, phase="submitted")
    heartbeat = run / "heartbeat.json"
    last_heartbeat = (
        read_json(heartbeat)["at"]
        if heartbeat.exists()
        else result.get("monitor", {}).get("started_at", time.time())
    )
    if result["active"] and time.time() - last_heartbeat > 90:
        result.update(
            active=False,
            phase="indeterminate",
            reason="monitor heartbeat expired; resume to recover",
        )
    return result


def idle_gpu_node(record):
    """Recognize healthy idle MI300X-pool nodes with enough requested resources."""
    states = set(record.get("State", "").split("+"))
    if "IDLE" not in states or states - {"IDLE", "DYNAMIC_NORM"}:
        return False
    if "compute-1" not in record.get("Partitions", "").split(","):
        return False
    try:
        cpus = int(record["CPUEfctv"]) - int(record["CPUAlloc"])
        memory = int(record["RealMemory"]) - int(record["AllocMem"])
    except (KeyError, ValueError) as error:
        raise ValueError("incomplete idle-node resource evidence") from error
    gpu = re.search(
        r"(?:^|,)gpu(?::[^:,()]+)?:([1-9][0-9]*)(?:\(|,|$)", record.get("Gres", "")
    )
    return cpus >= 16 and memory >= 65536 and gpu is not None


def select_partition(run, req):
    """Choose from current scheduler evidence; never treat a query failure as idle."""
    policy = req["partition"]
    evidence = {"requested_policy": policy, "observed_at": time.time(), "queries": []}

    def query(args):
        result = command(args)
        evidence["queries"].append(
            {
                "command": args,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        atomic_json(run / "partition-selection.json", evidence)
        if result.returncode:
            raise RuntimeError("partition selection scheduler query failed")
        return result.stdout

    def partition(name):
        record = parse_record(query(["scontrol", "show", "partition", "-o", name]))
        if record.get("PartitionName") != name or record.get("State") not in {
            "UP",
            "DOWN",
            "DRAIN",
            "INACTIVE",
        }:
            raise ValueError("invalid partition selection evidence")
        return record

    preferred = "compute-1" if policy == "auto" else policy
    chosen = preferred
    state = partition(preferred)
    if policy == "auto":
        candidates = []
        if state["State"] == "UP":
            output = query(["scontrol", "show", "nodes", "-o"])
            records = [
                parse_record(line) for line in output.splitlines() if line.strip()
            ]
            if not records or any(
                not {"NodeName", "State", "Partitions", "Gres"}.issubset(record)
                for record in records
            ):
                raise ValueError("invalid node selection evidence")
            candidates = [
                record["NodeName"] for record in records if idle_gpu_node(record)
            ]
        evidence["eligible_idle_nodes"] = sorted(candidates)
        if not candidates:
            chosen = "compute-0"
            state = partition(chosen)
            evidence["reason"] = (
                "no eligible idle GPU node in available compute-1 partition"
            )
        else:
            evidence["reason"] = "compute-1 has eligible idle GPU nodes"
    else:
        evidence["reason"] = "explicit partition override"
    if state["State"] != "UP":
        raise RuntimeError("selected partition is not available")
    evidence["chosen_partition"] = chosen
    atomic_json(run / "partition-selection.json", evidence)
    return chosen


WAIT_FINISHED_REASON = "wait finished; explicit terminal evidence required"


def settle_terminal(run, req, job_id, deadline):
    """The waiter may exit while Slurm still exposes a completing transition."""
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        atomic_json(run / "heartbeat.json", {"at": time.time()})
        evidence = inspect_job(run, req, job_id, timeout=min(20, remaining))
        if evidence and evidence["state"] in TERMINAL:
            return evidence
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return None


def reconcile_terminal(run, req):
    """Upgrade only a missing-terminal outcome when later evidence arrives."""
    paths = [
        run / (name + ".json")
        for name in ("controller-result", "terminal", "wait-result", "receipt")
    ]
    if not all(path.exists() for path in paths):
        return
    result, terminal, waiter, receipt = (read_json(path) for path in paths)
    if result != {"status": "indeterminate", "reason": WAIT_FINISHED_REASON}:
        return
    job_id = receipt.get("job_id")
    if (
        receipt.get("run_key") != req["run_key"]
        or receipt.get("source_sha") != req["source_sha"]
        or terminal.get("run_key") != req["run_key"]
        or terminal.get("state") not in TERMINAL
        or not job_id
        or terminal.get("job_id") != job_id
        or waiter.get("job_id") != job_id
    ):
        return
    atomic_json(paths[0], {"status": "terminal", "reason": WAIT_FINISHED_REASON})


def monitor(run):
    req = load_request(run)
    with lock(run):
        if (run / "submission-intent.json").exists():
            return
        started = time.time()
        atomic_json(run / "submission-intent.json", {"started_at": started})
    args = [
        "stdbuf",
        "-oL",
        "-eL",
        "sbatch",
        "--parsable",
        "--wait",
        "--no-requeue",
        "--nodes=1",
        "--ntasks=1",
        "--gpus=1",
        "--deadline=now+270minutes",
        "--cpus-per-task=16",
        "--mem=64G",
        "--time=04:00:00",
        "--signal=B:USR1@120",
        "--job-name=dynamo-" + req["run_key"],
        "--comment=" + req["run_key"],
        "--output=" + str(run / "slurm.log"),
        "--error=" + str(run / "slurm.log"),
        "--chdir=" + str(run),
        str(run / "controller" / "barite.sbatch"),
        str(run),
    ]
    selector = selectors.DefaultSelector()
    process = None
    job_id = None
    missing_record = False
    terminal = None
    try:
        args.insert(4, "--partition=" + select_partition(run, req))
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        buffers = {"stdout": b"", "stderr": b""}
        last_check = 0
        last_scheduler_check = 0
        while selector.get_map() or process.poll() is None:
            for key, _ in selector.select(timeout=1):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                with (run / ("sbatch-" + name + ".log")).open("ab", buffering=0) as log:
                    log.write(chunk)
                    os.fsync(log.fileno())
                buffers[name] += chunk
                if name == "stderr" and b"invalid job id" in buffers[name].lower():
                    missing_record = True
                while b"\n" in buffers[name]:
                    line, buffers[name] = buffers[name].split(b"\n", 1)
                    if name == "stdout" and job_id is None:
                        job_id = parse_job_id(line.decode())
                        atomic_json(
                            run / "receipt.json",
                            {
                                "job_id": job_id,
                                "run_key": req["run_key"],
                                "source_sha": req["source_sha"],
                                "submitted_at": started,
                            },
                        )
            now = time.time()
            if now - last_check >= 5:
                last_check = now
                atomic_json(run / "heartbeat.json", {"at": now})
                if job_id and now - last_scheduler_check >= 20:
                    last_scheduler_check = now
                    evidence = inspect_job(run, req, job_id)
                    if evidence and evidence["state"] in TERMINAL:
                        terminal = evidence
                    if (
                        evidence
                        and evidence["state"] == "PENDING"
                        and now - started > req["queue_timeout_seconds"]
                    ):
                        cancel(run, req, job_id)
                if job_id is None and now - started > 60:
                    recovered = recover(run, req)
                    if recovered:
                        job_id = recovered["job_id"]
                if now - started > req["controller_timeout_seconds"]:
                    raise TimeoutError("controller lifetime exhausted")
        code = process.wait(timeout=10)
        # stderr may end without a newline, and Slurm versions vary the diagnostic.
        stderr = (
            (run / "sbatch-stderr.log").read_text()
            if (run / "sbatch-stderr.log").exists()
            else ""
        )
        missing_record = missing_record or any(
            term in stderr.lower()
            for term in (
                "invalid job id",
                "job/step already completing or completed",
                "slurm_load_jobs error",
                "job no longer found",
                "exit code not found",
            )
        )
        atomic_json(
            run / "wait-result.json",
            {"returncode": code, "missing_record": missing_record, "job_id": job_id},
        )
        if job_id and terminal is None:
            remaining = req["controller_timeout_seconds"] - (time.time() - started)
            terminal = settle_terminal(
                run, req, job_id, time.monotonic() + max(0, min(60, remaining))
            )
        status = (
            "terminal"
            if terminal and terminal["state"] in TERMINAL
            else "indeterminate"
        )
        atomic_json(
            run / "controller-result.json",
            {
                "status": status,
                "reason": WAIT_FINISHED_REASON,
            },
        )
    except Exception as error:  # noqa: BLE001 - persist unexpected daemon failures and release its allocation
        diagnostics = []
        if job_id is None:
            try:
                recovered = recover(run, req)
                job_id = recovered["job_id"] if recovered else None
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as recovery_error:
                diagnostics.append("recovery failed: " + str(recovery_error))
        if job_id:
            try:
                cancel(run, req, job_id)
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as cancellation_error:
                diagnostics.append("cancellation failed: " + str(cancellation_error))
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        atomic_json(
            run / "controller-result.json",
            {
                "status": "indeterminate",
                "reason": str(error),
                "diagnostics": diagnostics,
            },
        )
    finally:
        selector.close()
        if process:
            process.stdout.close()
            process.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("start", "status", "resume", "finalize", "_monitor")
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cancel-if-active", action="store_true")
    parser.add_argument("--deadline-seconds", type=int, default=240)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    req = load_request(run)
    if args.action == "_monitor":
        monitor(run)
        return
    if not 1 <= args.deadline_seconds <= 240:
        raise ValueError("finalizer deadline must be 1..240 seconds")
    with lock(run):
        if args.action == "start" and not (run / "monitor.json").exists():
            with (run / "monitor.log").open("ab") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_monitor",
                        "--run-dir",
                        str(run),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    close_fds=True,
                )
            atomic_json(
                run / "monitor.json", {"pid": child.pid, "started_at": time.time()}
            )
    if args.action in ("resume", "finalize"):
        receipt = (
            read_json(run / "receipt.json")
            if (run / "receipt.json").exists()
            else recover(run, req)
        )
        if receipt:
            if args.action == "finalize" and args.cancel_if_active:
                cancel(run, req, receipt["job_id"])
            deadline = time.monotonic() + args.deadline_seconds
            while True:
                evidence = inspect_job(run, req, receipt["job_id"])
                if (
                    args.action != "finalize"
                    or evidence is None
                    or evidence["state"] in TERMINAL
                    or time.monotonic() + 5 >= deadline
                ):
                    break
                time.sleep(2)
        elif (run / "submission-intent.json").exists():
            atomic_json(
                run / "controller-result.json",
                {
                    "status": "indeterminate",
                    "reason": "submission receipt unavailable; never resubmitting",
                },
            )
    reconcile_terminal(run, req)
    print(json.dumps(snapshot(run), sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            json.dumps({"error": str(exc), "active": False, "phase": "indeterminate"})
        )
        sys.exit(1)
