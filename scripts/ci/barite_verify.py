#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed artifact verification for the trusted Barite controller."""

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import ipaddress
import json
import os
import shutil
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath

from barite_common import atomic_json, read_json, sha256_file, sha256_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(request):
    return {
        key: request[key]
        for key in (
            "source_sha",
            "archive_sha256",
            "controller_sha",
            "controller_bundle_sha256",
        )
    }


def match_identity(manifest, request):
    for key, value in identity(request).items():
        require(manifest.get(key) == value, f"{key} mismatch")


def extract_source(run_dir, destination):
    """Validate all archive entries before extracting into a new directory."""
    request = read_json(run_dir / "request.json")
    manifest = read_json(run_dir / "source-manifest.json")
    for key in ("source_sha", "archive_sha256"):
        require(manifest.get(key) == request[key], f"source {key} mismatch")
    archive = run_dir / "source.tar"
    require(sha256_file(archive) == request["archive_sha256"], "archive hash mismatch")
    require(not destination.exists(), "extraction destination already exists")
    with tarfile.open(archive, "r:") as handle:
        members = handle.getmembers()
        seen = set()
        links = set()
        for member in members:
            path = PurePosixPath(member.name)
            require(
                not path.is_absolute() and ".." not in path.parts and path.parts,
                "unsafe archive path",
            )
            require(member.name not in seen, "duplicate archive path")
            seen.add(member.name)
            require(
                member.isfile() or member.isdir() or member.issym(),
                "unsupported archive entry",
            )
            require(not member.mode & 0o7000, "unsafe archive permissions")
            if member.issym():
                target = PurePosixPath(member.linkname)
                require(not target.is_absolute(), "absolute symlink")
                depth = len(path.parent.parts)
                for part in target.parts:
                    depth += -1 if part == ".." else 1
                    require(depth >= 0, "escaping symlink")
                links.add(path)
        expected = {entry["path"]: entry for entry in manifest["files"]}
        require(
            len(expected) == len(manifest["files"]), "duplicate source manifest entries"
        )
        actual = {}
        for member in members:
            if member.isdir():
                continue
            if member.issym():
                content = member.linkname.encode()
                digest = hashlib.sha256(content).hexdigest()
                size = len(content)
            else:
                with handle.extractfile(member) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                size = member.size
            entry = {
                "path": member.name,
                "type": "symlink" if member.issym() else "file",
                "sha256": digest,
                "size": size,
                "mode": member.mode & 0o777,
            }
            if member.issym():
                entry["linkname"] = member.linkname
            actual[member.name] = entry
        require(actual == expected, "source manifest file mismatch")
        for member in members:
            require(
                not any(
                    parent in links for parent in PurePosixPath(member.name).parents
                ),
                "entry beneath symlink",
            )
        destination.mkdir(parents=True)
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                path.mkdir(exist_ok=True)
            elif member.issym():
                path.symlink_to(member.linkname)
            else:
                with handle.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(member.mode & 0o777)
        for link in links:
            require(
                (destination / link).resolve().is_relative_to(destination.resolve()),
                "escaping symlink chain",
            )


def wheel_record(path):
    with zipfile.ZipFile(path) as archive:
        records = [
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        ]
        require(len(records) == 1, "wheel requires one RECORD")
        rows = csv.reader(io.StringIO(archive.read(records[0]).decode()))
        entries = {}
        for name, digest, size in rows:
            if digest:
                algorithm, encoded = digest.split("=", 1)
                require(algorithm == "sha256", "wheel requires sha256 RECORD")
                content = archive.read(name)
                actual = (
                    base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                    .decode()
                    .rstrip("=")
                )
                require(
                    actual == encoded and len(content) == int(size),
                    f"wheel RECORD mismatch: {name}",
                )
                entries[name] = hashlib.sha256(content).hexdigest()
        require(entries, "empty wheel RECORD")
        return entries


def build_manifest(request, spec):
    for key in ("source_sha", "archive_sha256"):
        require(spec.get(key) == request[key], f"build {key} mismatch")
    trusted_path = Path(__file__).resolve().parent / "rocm" / "contract.json"
    trusted = read_json(trusted_path)
    require(
        spec.get("contract_sha256") == sha256_file(trusted_path),
        "build contract mismatch",
    )
    require(
        spec.get("base_uri") == trusted["base_uri"]
        and spec.get("model") == trusted["model"],
        "build base/model mismatch",
    )
    wheels = []
    for item in spec["wheels"]:
        path = Path(item["path"])
        wheels.append(
            {
                **item,
                "filename": path.name,
                "sha256": sha256_file(path),
                "record": wheel_record(path),
            }
        )
    require(wheels, "no built wheels")
    system_packages = {
        name: system_origin(name, Path(sys.prefix).resolve())
        for name in spec.get("system_modules", ["torch", "vllm"])
    }
    return {
        **spec,
        **identity(request),
        "schema_version": 1,
        "status": "passed",
        "wheels": wheels,
        "system_packages": system_packages,
    }


def verify_build(manifest, request):
    match_identity(manifest, request)
    require(
        manifest.get("status") == "passed" and manifest.get("wheels"),
        "build did not pass",
    )


def image_manifest(request, image, build):
    verify_build(build, request)
    return {
        **identity(request),
        "schema_version": 1,
        "status": "passed",
        "sqsh_sha256": sha256_file(image),
        "build": build,
    }


def verify_image(request, image, manifest):
    match_identity(manifest, request)
    require(manifest.get("status") == "passed", "image not published successfully")
    actual = sha256_file(image)
    require(actual == manifest["sqsh_sha256"], "image hash mismatch")
    require(
        not request.get("reuse_image_sha") or actual == request["reuse_image_sha"],
        "reuse image mismatch",
    )
    verify_build(manifest["build"], request)
    return manifest


def system_origin(name, prefix):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    require(
        not path.is_relative_to(prefix), f"base package shadowed in overlay: {name}"
    )
    distribution = importlib.metadata.distribution(name)
    root = Path(distribution.locate_file("")).resolve()
    require(path.is_relative_to(root), f"base module origin mismatch: {name}")
    require(
        bool({"site-packages", "dist-packages"}.intersection(path.parts))
        and path.is_relative_to(Path(sys.base_prefix).resolve())
        and not path.is_relative_to(Path.home() / ".local"),
        f"untrusted base origin: {name}",
    )
    relative = path.relative_to(root).as_posix()
    entries = {str(item): item for item in distribution.files or []}
    require(
        relative in entries and entries[relative].hash is not None,
        f"missing base RECORD: {name}",
    )
    record_hash = entries[relative].hash
    require(record_hash.mode == "sha256", f"unsupported base RECORD hash: {name}")
    digest = sha256_file(path)
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
    require(encoded == record_hash.value, f"base RECORD mismatch: {name}")
    return {"path": str(path), "sha256": digest, "version": distribution.version}


def provenance(request, build):
    """Import with the test interpreter and match actual bytes to wheel RECORD."""
    verify_build(build, request)
    prefix = Path(sys.prefix).resolve()
    require(prefix != Path(sys.base_prefix).resolve(), "overlay venv required")
    modules = {}
    for wheel in build["wheels"]:
        distribution = importlib.metadata.distribution(wheel["distribution"])
        root = Path(distribution.locate_file("")).resolve()
        require(root.is_relative_to(prefix), "candidate distribution outside overlay")
        for name in wheel["modules"]:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            require(
                path.is_relative_to(root),
                f"module outside installed distribution: {name}",
            )
            relative = path.relative_to(root).as_posix()
            require(
                wheel["record"].get(relative) == sha256_file(path),
                f"module RECORD mismatch: {name}",
            )
            modules[name] = {"path": str(path), "sha256": sha256_file(path)}
    for name, expected in build["system_packages"].items():
        actual = system_origin(name, prefix)
        require(actual == expected, f"base package changed: {name}")
        modules[name] = actual
    return {
        **identity(request),
        "run_key": request["run_key"],
        "status": "passed",
        "build_manifest_sha256": sha256_json(build),
        "executable": sys.executable,
        "prefix": str(prefix),
        "modules": modules,
        "created_at": time.time(),
    }


def verify_junit(path, expected, started_at):
    require(path.stat().st_mtime >= started_at, "stale JUnit")
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    actual = [f"{case.get('classname')}::{case.get('name')}" for case in cases]
    require(
        expected and len(set(expected)) == len(expected),
        "empty or duplicate expected cases",
    )
    require(sorted(actual) == sorted(expected), f"unexpected JUnit cases: {actual}")
    require(
        not any(
            list(case.iter(tag))
            for case in cases
            for tag in ("failure", "error", "skipped")
        ),
        "failed/skipped/xfailed JUnit",
    )
    for suite in root.iter("testsuite"):
        for key in ("failures", "errors", "skipped"):
            require(int(suite.get(key, "0")) == 0, f"JUnit {key}")
    return {"sha256": sha256_file(path), "cases": actual}


def verify_listener_evidence(evidence, request, suite, job_id, started_at):
    """Validate the small, role-scoped listener snapshot, also on the client."""
    roles = {"nats", "etcd_client", "etcd_peer", "frontend", "system"}
    require(evidence["status"] == "passed", "listener capture incomplete")
    require(
        evidence["run_key"] == request["run_key"]
        and evidence["source_sha"] == request["source_sha"]
        and evidence["suite"] == suite
        and str(evidence["job_id"]) == str(job_id),
        "listener evidence identity mismatch",
    )
    ports = evidence["expected_ports"]
    require(
        set(ports) == roles and len(set(ports.values())) == len(roles),
        "listener roles/ports incomplete",
    )
    require(
        all(isinstance(port, int) and 0 < port < 65536 for port in ports.values()),
        "invalid listener port",
    )
    require(
        {entry["role"] for entry in evidence["listeners"]} == roles,
        "required listener was not observed",
    )
    for entry in evidence["listeners"]:
        require(entry["port"] == ports[entry["role"]], "listener port mismatch")
        require(
            ipaddress.ip_address(entry["address"]).is_loopback,
            f"non-loopback {entry['role']} listener: {entry['address']}",
        )
        require(
            entry["pid"] > 0
            and entry["root_pid"] > 0
            and 0 < entry["process_created_at"] <= entry["captured_at"]
            and entry["captured_at"] >= started_at,
            "stale or invalid listener process",
        )
    anchors = [
        entry
        for entry in evidence["listeners"]
        if entry["role"] in ("frontend", "system")
    ]
    expanded = evidence["service_listeners"]
    require(
        {entry["pid"] for entry in expanded} == {entry["pid"] for entry in anchors},
        "Dynamo service listener inventory incomplete",
    )
    for entry in expanded:
        owners = [anchor for anchor in anchors if anchor["pid"] == entry["pid"]]
        require(
            entry["service_roles"] == sorted({anchor["role"] for anchor in owners}),
            "Dynamo service listener owner mismatch",
        )
        require(
            all(
                entry["process_created_at"] == anchor["process_created_at"]
                and entry["root_pid"] == anchor["root_pid"]
                for anchor in owners
            ),
            "Dynamo service listener process identity mismatch",
        )
        require(
            entry["captured_at"] >= started_at
            and entry["captured_at"] >= entry["process_created_at"],
            "stale Dynamo service listener",
        )
        require(
            isinstance(entry["port"], int) and 0 < entry["port"] < 65536,
            "invalid Dynamo service listener port",
        )
        require(
            ipaddress.ip_address(entry["address"]).is_loopback,
            f"non-loopback Dynamo service listener: {entry['address']}:{entry['port']}",
        )
    for anchor in anchors:
        require(
            any(
                entry["pid"] == anchor["pid"]
                and entry["port"] == anchor["port"]
                and entry["address"] == anchor["address"]
                for entry in expanded
            ),
            "named endpoint missing from Dynamo service listener inventory",
        )
    return evidence


def workload(request, runtime_dir, job_id=None):
    contract = read_json(runtime_dir / "contract.json")
    trusted_path = Path(__file__).resolve().parent / "rocm" / "contract.json"
    trusted = read_json(trusted_path)
    expected_suites = {}
    for name, nodes in trusted["suites"].items():
        expected_suites[name] = []
        for node in nodes:
            path, case = node.split("::", 1)
            expected_suites[name].append(
                path.removesuffix(".py").replace("/", ".") + "::" + case
            )
    require(
        contract["suites"] == expected_suites,
        "test contract differs from trusted selection",
    )
    require(
        contract["run_key"] == request["run_key"]
        and contract["source_sha"] == request["source_sha"],
        "stale test contract",
    )
    require(
        set(contract["suites"]) == {"imports", "frontend", "aggregate"},
        "missing required suites",
    )
    evidence = {}
    for name in ("hip", "provenance"):
        evidence[name] = read_json(runtime_dir / f"{name}.json")
        require(
            evidence[name].get("status") == "passed"
            and evidence[name].get("run_key") == request["run_key"],
            f"invalid {name} evidence",
        )
    model = read_json(runtime_dir / "model-manifest.json")
    model_content = (
        json.dumps(
            {"model": model["model"], "files": model["files"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    model_hash = hashlib.sha256(model_content.encode()).hexdigest()
    require(model["model"] == trusted["model"], "model revision mismatch")
    require(
        model_hash
        == model["model_manifest_sha256"]
        == contract["model_manifest_sha256"],
        "model manifest mismatch",
    )
    hip = evidence["hip"]
    require(
        str(hip.get("architecture", "")).startswith("gfx942"),
        "unexpected GPU architecture",
    )
    require(hip.get("source_sha") == request["source_sha"], "HIP source mismatch")
    require(
        hip.get("device_count") == 1
        and hip.get("hip")
        and hip.get("rocr_visible_devices"),
        "missing allocated HIP device",
    )
    require(hip.get("result") == [[2.0, 2.0], [2.0, 2.0]], "HIP arithmetic failed")
    require(
        (runtime_dir / "hip.json").stat().st_mtime >= contract["started_at"],
        "stale HIP evidence",
    )
    match_identity(evidence["provenance"], request)
    require(
        evidence["provenance"]["created_at"] >= contract["started_at"],
        "stale provenance",
    )
    image = read_json(runtime_dir / "image-manifest.json")
    match_identity(image, request)
    require(image.get("status") == "passed", "image was not published successfully")
    verify_build(image["build"], request)
    require(
        evidence["provenance"].get("build_manifest_sha256")
        == sha256_json(image["build"]),
        "runtime build differs from published image",
    )
    require(
        image["build"]["contract_sha256"] == sha256_file(trusted_path),
        "image test contract mismatch",
    )
    listener_evidence = {}
    current_job = str(job_id) if job_id is not None else os.environ["SLURM_JOB_ID"]
    for suite in ("frontend", "aggregate"):
        path = runtime_dir / f"{suite}-listeners.json"
        observed = verify_listener_evidence(
            read_json(path), request, suite, current_job, contract["started_at"]
        )
        listener_evidence[suite] = {"sha256": sha256_file(path), **observed}
    collections = {}
    required_plugins = {entry[0] for entry in image["build"]["pytest_plugins"]}
    for name, nodes in trusted["suites"].items():
        path = runtime_dir / f"{name}-collection.json"
        collected = read_json(path)
        require(collected["nodeids"] == nodes, f"unexpected {name} collection")
        require(
            collected["plugins"] and required_plugins.issubset(collected["plugins"]),
            f"missing {name} pytest plugins",
        )
        require(
            path.stat().st_mtime >= contract["started_at"], f"stale {name} collection"
        )
        collections[name] = {"sha256": sha256_file(path), **collected}
    junit = {
        name: verify_junit(
            runtime_dir / "test-results" / f"{name}.xml", cases, contract["started_at"]
        )
        for name, cases in contract["suites"].items()
    }
    return {
        **identity(request),
        "schema_version": 1,
        "run_key": request["run_key"],
        "job_id": str(job_id) if job_id is not None else os.environ["SLURM_JOB_ID"],
        "status": "passed",
        "sqsh_sha256": image["sqsh_sha256"],
        "contract_sha256": sha256_file(trusted_path),
        "model_manifest_sha256": model_hash,
        "junit": junit,
        "collections": collections,
        "listeners": listener_evidence,
        "evidence": evidence,
        "completed_at": time.time(),
    }


def verify_collected(request, runtime_dir):
    """Revalidate the downloaded evidence without importing installed GPU packages."""
    completed = read_json(runtime_dir / "completed.json")
    expected = workload(request, runtime_dir, job_id=completed["job_id"])
    for key, value in expected.items():
        if key != "completed_at":
            require(completed.get(key) == value, f"collected workload mismatch: {key}")
    require(
        completed["completed_at"] >= expected["evidence"]["provenance"]["created_at"],
        "completion predates provenance",
    )
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "extract",
            "build-manifest",
            "verify-build",
            "image-manifest",
            "verify-image",
            "provenance",
            "workload",
            "verify-collected",
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    for option in (
        "destination",
        "build-spec",
        "build-manifest",
        "image",
        "manifest",
        "runtime-dir",
        "output",
    ):
        parser.add_argument(f"--{option}", type=Path)
    args = parser.parse_args()
    request = read_json(args.run_dir / "request.json")
    result = None
    if args.command == "extract":
        extract_source(args.run_dir, args.destination)
    elif args.command == "build-manifest":
        result = build_manifest(request, read_json(args.build_spec))
    elif args.command == "verify-build":
        verify_build(read_json(args.build_manifest), request)
    elif args.command == "image-manifest":
        result = image_manifest(request, args.image, read_json(args.build_manifest))
    elif args.command == "verify-image":
        result = verify_image(request, args.image, read_json(args.manifest))
    elif args.command == "provenance":
        result = provenance(request, read_json(args.build_manifest))
    elif args.command == "workload":
        result = workload(request, args.runtime_dir or args.run_dir)
    elif args.command == "verify-collected":
        result = verify_collected(request, args.runtime_dir or args.run_dir)
    if result is not None and args.output:
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
