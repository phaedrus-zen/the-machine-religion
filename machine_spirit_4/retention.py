"""Manifest-driven retention for TMR/MS3/MS4 persisted data.

The scheduler calls :func:`run_retention_cycle`; operators can run this module
directly for a metadata-only dry run.  Payload bytes are never read for normal
retention decisions, private psyche stores are never auto-pruned, and any
manifest or path-safety error disables the whole mutation pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable


MANIFEST_SCHEMA = "HiveMindRetentionManifest.v1"
REPORT_SCHEMA = "HiveMindRetentionReport.v1"
RECEIPT_SCHEMA = "HiveMindRetentionReceipt.v1"
DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("retention_manifest.json")
PRIVATE_CLASS = "private_psyche"
DATA_CLASSES = frozenset({PRIVATE_CLASS, "operational", "derived"})
ROTATIONS = frozenset(
    {"none", "prune_files", "rename_recreate", "writer_atomic_replace", "sqlite_vacuum"}
)
AUTHORIZATION_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
SECONDS_PER_DAY = 86_400

log = logging.getLogger("ms4.retention")

AuthorizationFn = Callable[[str, str], str | None]
QuiescenceFn = Callable[[str], bool]

PATH_OVERRIDE_STORES = {
    "MS4_AUDIT_LOG": "ms4_audit_log",
    "MS4_VOICE_TURNS_LOG": "ms4_voice_turn_ring",
    "MS4_REFLEX_DIR": "ms4_canned_audio_cache",
    "MS4_VOICE_CANDIDATE_REVIEW_STORE": "ms4_voice_candidate_review",
    "MS4_QM_INDEX_DIR": "ms4_quartermaster_cache",
    "MS4_MCP_IMPORTS_PATH": "ms4_runtime_registry_state",
    "MS4_HERMES_DIR": "hermes_updater_runtime",
}


class ManifestError(ValueError):
    """The manifest cannot safely authorize a retention mutation."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    candidate = PurePath(normalized)
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise ManifestError(f"{field} must remain beneath the retention root")
    return normalized


def _optional_nonnegative_int(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if not _is_int(value) or value < 0:
        raise ManifestError(f"{field} must be null or a non-negative integer")
    return value


def _validate_store(raw: Any, seen: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManifestError("each store must be an object")
    store_id = raw.get("id")
    if not isinstance(store_id, str) or not re.fullmatch(r"[a-z0-9_.-]{1,80}", store_id):
        raise ManifestError("store id is invalid")
    if store_id in seen:
        raise ManifestError("store ids must be unique")
    seen.add(store_id)
    data_class = raw.get("data_class")
    if data_class not in DATA_CLASSES:
        raise ManifestError(f"store {store_id} has an invalid data class")
    for field in ("sensitivity", "enforcement"):
        if not isinstance(raw.get(field), str) or not re.fullmatch(
            r"[a-z0-9_]{1,80}", raw[field]
        ):
            raise ManifestError(f"store {store_id} must declare {field}")
    paths = raw.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ManifestError(f"store {store_id} must declare paths")
    raw["paths"] = [
        _safe_relative(value, field=f"stores.{store_id}.paths") for value in paths
    ]
    active_paths = raw.get("active_paths", [])
    if not isinstance(active_paths, list):
        raise ManifestError(f"store {store_id} active_paths must be a list")
    raw["active_paths"] = [
        _safe_relative(value, field=f"stores.{store_id}.active_paths")
        for value in active_paths
    ]
    for field in ("writer", "growth_driver"):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            raise ManifestError(f"store {store_id} must declare {field}")
    access = raw.get("access")
    if not isinstance(access, dict) or set(access) != {"read", "export", "delete", "at_rest"}:
        raise ManifestError(f"store {store_id} access contract is incomplete")
    if not all(isinstance(value, str) and value for value in access.values()):
        raise ManifestError(f"store {store_id} access values must be non-empty strings")
    policy = raw.get("policy")
    if not isinstance(policy, dict):
        raise ManifestError(f"store {store_id} policy is missing")
    for field in (
        "max_age_days",
        "max_bytes",
        "max_files",
        "rotate_at_bytes",
        "compact_at_bytes",
        "protect_newest_files",
    ):
        policy[field] = _optional_nonnegative_int(
            policy.get(field), field=f"stores.{store_id}.policy.{field}"
        )
    if policy["max_bytes"] is None and policy["max_files"] is None:
        raise ManifestError(f"store {store_id} needs a byte or file-count budget")
    rotation = policy.get("rotation")
    if rotation not in ROTATIONS:
        raise ManifestError(f"store {store_id} rotation is invalid")
    if not isinstance(policy.get("auto_prune"), bool):
        raise ManifestError(f"store {store_id} auto_prune must be boolean")
    if data_class == PRIVATE_CLASS and policy["auto_prune"]:
        raise ManifestError(f"private store {store_id} cannot auto-prune")
    if rotation == "rename_recreate":
        if not raw["active_paths"] or not policy["rotate_at_bytes"]:
            raise ManifestError(f"store {store_id} rotation lacks an active path or threshold")
    if rotation == "sqlite_vacuum" and policy["compact_at_bytes"] is None:
        raise ManifestError(f"store {store_id} SQLite compaction lacks a threshold")
    if rotation == "sqlite_vacuum" and policy["auto_prune"]:
        raise ManifestError(f"store {store_id} SQLite files cannot be auto-pruned")
    compaction = policy.get("compaction")
    if compaction not in {"none", "writer_bounded", "sqlite_vacuum", "atomic_replace"}:
        raise ManifestError(f"store {store_id} compaction is invalid")
    hold_suffix = policy.get("hold_suffix")
    if not isinstance(hold_suffix, str) or not hold_suffix.startswith("."):
        raise ManifestError(f"store {store_id} hold suffix is invalid")
    return raw


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load and fully validate a retention manifest before any data scan."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ManifestError("manifest is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError("manifest schema is invalid")
    schedule = payload.get("schedule_seconds")
    if not _is_int(schedule) or not 30 <= schedule <= SECONDS_PER_DAY:
        raise ManifestError("schedule_seconds must be between 30 and 86400")
    minimum_free = payload.get("minimum_free_bytes")
    if not _is_int(minimum_free) or minimum_free < 0:
        raise ManifestError("minimum_free_bytes must be non-negative")
    for field in ("budget_authority", "disk_pressure_behavior"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ManifestError(f"{field} must be declared")
    payload["telemetry_path"] = _safe_relative(
        payload.get("telemetry_path"), field="telemetry_path"
    )
    payload["receipt_path"] = _safe_relative(
        payload.get("receipt_path"), field="receipt_path"
    )
    if not isinstance(payload.get("authorization_provider"), str):
        raise ManifestError("authorization_provider must be declared")
    stores = payload.get("stores")
    if not isinstance(stores, list) or not stores:
        raise ManifestError("stores must be a non-empty list")
    seen: set[str] = set()
    payload["stores"] = [_validate_store(store, seen) for store in stores]
    return payload


def _beneath(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _expand(root: Path, patterns: list[str], hold_suffix: str) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.name.endswith(hold_suffix) or path.is_symlink() or not path.is_file():
                continue
            if not _beneath(root, path):
                raise ManifestError("a matched path escaped the retention root")
            try:
                stat = path.stat()
            except OSError as exc:
                raise ManifestError("a matched path could not be inspected") from exc
            relative = path.relative_to(root).as_posix()
            records[relative] = {
                "path": path,
                "relative": relative,
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "mtime_ns": int(stat.st_mtime_ns),
                "mode": int(stat.st_mode & 0o777),
                "device": int(stat.st_dev),
                "inode": int(stat.st_ino),
                "root": root,
                "hold_path": Path(f"{path}{hold_suffix}"),
                "held": Path(f"{path}{hold_suffix}").is_file(),
                "active": False,
            }
    return list(records.values())


def _active_set(root: Path, patterns: list[str]) -> set[str]:
    active: set[str] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_symlink() or not path.is_file() or not _beneath(root, path):
                continue
            active.add(path.relative_to(root).as_posix())
    return active


def _mark_active(root: Path, store: dict[str, Any], records: list[dict[str, Any]]) -> None:
    active = _active_set(root, store["active_paths"])
    for record in records:
        record["active"] = record["relative"] in active
    protect = int(store["policy"].get("protect_newest_files") or 0)
    if protect:
        newest = sorted(
            records,
            key=lambda item: (item["mtime_ns"], item["relative"]),
            reverse=True,
        )[:protect]
        for record in newest:
            record["active"] = True


def _rotation_target(path: Path, now: float) -> Path:
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        candidate = path.with_name(f"{path.name}.retained.{stamp}.{uuid.uuid4().hex[:8]}")
        if not candidate.exists():
            return candidate


def _plan_store(
    root: Path,
    store: dict[str, Any],
    now: float,
    *,
    quiescent: bool,
) -> dict[str, Any]:
    policy = store["policy"]
    records = _expand(root, store["paths"], policy["hold_suffix"])
    _mark_active(root, store, records)
    operations: list[dict[str, Any]] = []
    simulated = [dict(record) for record in records]

    if policy["rotation"] == "rename_recreate" and quiescent:
        threshold = int(policy["rotate_at_bytes"])
        for record in list(simulated):
            if not record["active"] or record["held"] or record["size"] <= threshold:
                continue
            target = _rotation_target(record["path"], now)
            operations.append({"kind": "rotate", "record": record, "target": target})
            simulated.remove(record)
            simulated.extend(
                [
                    {
                        **record,
                        "path": target,
                        "relative": target.relative_to(root).as_posix(),
                        "hold_path": Path(f"{target}{policy['hold_suffix']}"),
                        "active": False,
                    },
                    {**record, "size": 0, "mtime": now, "mtime_ns": int(now * 1e9)},
                ]
            )

    if policy["rotation"] == "sqlite_vacuum":
        threshold = int(policy["compact_at_bytes"])
        for record in simulated:
            if (
                record["path"].suffix == ".sqlite3"
                and (not record["active"] or quiescent)
                and not record["held"]
                and record["size"] >= threshold
            ):
                operations.append({"kind": "compact", "record": record})

    if policy["auto_prune"]:
        def removable(item: dict[str, Any]) -> bool:
            return not item["active"] and not item["held"]

        selected: set[str] = set()
        max_age = policy["max_age_days"]
        if max_age is not None:
            cutoff = now - (max_age * SECONDS_PER_DAY)
            selected.update(
                record["relative"]
                for record in simulated
                if removable(record) and record["mtime"] < cutoff
            )

        def remaining() -> list[dict[str, Any]]:
            return [record for record in simulated if record["relative"] not in selected]

        candidates = sorted(
            (record for record in simulated if removable(record)),
            key=lambda item: (item["mtime_ns"], item["relative"]),
        )
        max_files = policy["max_files"]
        max_bytes = policy["max_bytes"]
        for candidate in candidates:
            current = remaining()
            over_count = max_files is not None and len(current) > max_files
            over_bytes = max_bytes is not None and sum(item["size"] for item in current) > max_bytes
            if not over_count and not over_bytes:
                break
            selected.add(candidate["relative"])
        by_relative = {record["relative"]: record for record in simulated}
        operations.extend(
            {"kind": "prune", "record": by_relative[relative]}
            for relative in sorted(
                selected,
                key=lambda value: (by_relative[value]["mtime_ns"], value),
            )
        )
    return {
        "store": store,
        "records": records,
        "operations": operations,
        "quiescent": quiescent,
    }


def _same_generation(record: dict[str, Any]) -> bool:
    path: Path = record["path"]
    if path.is_symlink() or not _beneath(record["root"], path):
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    return (
        int(stat.st_dev) == record["device"]
        and int(stat.st_ino) == record["inode"]
        and int(stat.st_size) == record["size"]
        and int(stat.st_mtime_ns) == record["mtime_ns"]
    )


def _hold_present(record: dict[str, Any]) -> bool:
    try:
        return bool(record["hold_path"].is_file())
    except OSError:
        return True


def _rotate(operation: dict[str, Any]) -> tuple[bool, str]:
    record = operation["record"]
    path: Path = record["path"]
    target: Path = operation["target"]
    if not _same_generation(record) or _hold_present(record) or target.exists():
        return False, "generation_changed"
    try:
        path.rename(target)
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                int(record.get("mode") or 0o600),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            if not path.exists():
                target.rename(path)
            raise
        return True, "rotated"
    except Exception:
        return False, "rotation_failed"


def _prune(operation: dict[str, Any]) -> tuple[bool, str]:
    record = operation["record"]
    path: Path = record["path"]
    if _hold_present(record):
        return False, "evidence_hold"
    if not _same_generation(record):
        return False, "generation_changed"
    tombstone = path.with_name(f".{path.name}.retention-delete.{uuid.uuid4().hex}")
    try:
        path.rename(tombstone)
        try:
            if _hold_present(record):
                tombstone.rename(path)
                return False, "evidence_hold"
            tombstone.unlink()
        except Exception:
            if not path.exists() and tombstone.exists():
                tombstone.rename(path)
            raise
        return True, "pruned"
    except Exception:
        return False, "prune_failed"


def _compact(operation: dict[str, Any]) -> tuple[bool, str]:
    record = operation["record"]
    if _hold_present(record):
        return False, "evidence_hold"
    if not _same_generation(record):
        return False, "generation_changed"
    try:
        with sqlite3.connect(record["path"], timeout=0.1) as connection:
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            free_pages = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
            if page_count <= 0 or free_pages <= 0 or free_pages / page_count < 0.2:
                return True, "compaction_not_needed"
            connection.execute("VACUUM")
        return True, "compacted"
    except Exception:
        return False, "compaction_failed"


def _read_previous_telemetry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _store_report(
    root: Path,
    plan: dict[str, Any],
    *,
    previous: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    store = plan["store"]
    policy = store["policy"]
    records = _expand(root, store["paths"], policy["hold_suffix"])
    _mark_active(root, store, records)
    current_bytes = sum(record["size"] for record in records)
    current_files = len(records)
    max_bytes = policy["max_bytes"]
    max_files = policy["max_files"]
    max_age_days = policy["max_age_days"]
    oldest_age_days = (
        max(0.0, (now - min(record["mtime"] for record in records)) / SECONDS_PER_DAY)
        if records
        else None
    )
    age_over_budget = bool(
        max_age_days is not None
        and oldest_age_days is not None
        and oldest_age_days > max_age_days
    )
    over_budget = bool(
        (max_bytes is not None and current_bytes > max_bytes)
        or (max_files is not None and current_files > max_files)
        or age_over_budget
    )
    rotation_pending = bool(
        policy["rotation"] == "rename_recreate"
        and not plan["quiescent"]
        and any(
            record["active"]
            and not record["held"]
            and record["size"] > int(policy["rotate_at_bytes"])
            for record in records
        )
    )
    compaction_pending = bool(
        policy["rotation"] == "sqlite_vacuum"
        and not plan["quiescent"]
        and any(
            record["active"]
            and not record["held"]
            and record["path"].suffix == ".sqlite3"
            and record["size"] >= int(policy["compact_at_bytes"])
            for record in records
        )
    )
    prior = previous.get(store["id"], {}) if isinstance(previous, dict) else {}
    prior_at = prior.get("sampled_at_unix") if isinstance(prior, dict) else None
    prior_bytes = prior.get("current_bytes") if isinstance(prior, dict) else None
    growth = None
    forecast = None
    if isinstance(prior_at, (int, float)) and _is_int(prior_bytes) and now > prior_at:
        per_day = int(round((current_bytes - prior_bytes) * SECONDS_PER_DAY / (now - prior_at)))
        growth = max(0, per_day)
        forecast = current_bytes + growth
    status = "within_budget"
    if over_budget:
        status = (
            "operator_action_required"
            if store["data_class"] == PRIVATE_CLASS or not policy["auto_prune"]
            else "over_budget"
        )
    if rotation_pending or compaction_pending:
        status = "quiescence_required"
    return {
        "id": store["id"],
        "data_class": store["data_class"],
        "status": status,
        "current_bytes": current_bytes,
        "current_files": current_files,
        "max_bytes": max_bytes,
        "max_files": max_files,
        "max_age_days": max_age_days,
        "oldest_age_days": oldest_age_days,
        "age_over_budget": age_over_budget,
        "held_files": sum(1 for record in records if record["held"]),
        "enforcement": store.get("enforcement", "unspecified"),
        "sensitivity": store.get("sensitivity", "unspecified"),
        "rotation_pending": rotation_pending,
        "compaction_pending": compaction_pending,
        "growth_bytes_per_day": growth,
        "forecast_bytes_24h": forecast,
        "forecast_status": "measured" if forecast is not None else "insufficient_history",
    }


def run_cycle(
    *,
    root: Path | str = DEFAULT_ROOT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    dry_run: bool = False,
    now: float | None = None,
    disk_free_bytes: int | None = None,
    quiescent_store_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate, plan, then apply one bounded retention pass.

    A corrupt manifest, unsafe path, or failed pre-mutation scan returns a
    stable reason and performs no retention mutation.
    """

    root_path = Path(root).resolve()
    sampled_at = time.time() if now is None else float(now)
    quiescent_ids = set(quiescent_store_ids or ())
    try:
        manifest = load_manifest(manifest_path)
        known_ids = {store["id"] for store in manifest["stores"]}
        if not quiescent_ids <= known_ids:
            raise ManifestError("unknown quiescent store")
        plans = [
            _plan_store(
                root_path,
                store,
                sampled_at,
                quiescent=store["id"] in quiescent_ids,
            )
            for store in manifest["stores"]
        ]
        owners: dict[str, str] = {}
        for plan in plans:
            for record in plan["records"]:
                identity = str(record["path"].resolve())
                prior_owner = owners.setdefault(identity, plan["store"]["id"])
                if prior_owner != plan["store"]["id"]:
                    raise ManifestError("stores overlap")
        telemetry_path = root_path / manifest["telemetry_path"]
        receipt_path = root_path / manifest["receipt_path"]
        if not _beneath(root_path, telemetry_path) or not _beneath(root_path, receipt_path):
            raise ManifestError("manifest output path escaped the retention root")
    except ManifestError:
        log.warning("retention disabled reason=manifest_invalid")
        return {"schema": REPORT_SCHEMA, "ok": False, "reason": "manifest_invalid", "actions": []}

    try:
        free_bytes = (
            int(disk_free_bytes)
            if disk_free_bytes is not None
            else int(shutil.disk_usage(root_path).free)
        )
    except Exception:
        log.warning("retention disabled reason=disk_status_unavailable")
        return {
            "schema": REPORT_SCHEMA,
            "ok": False,
            "reason": "disk_status_unavailable",
            "actions": [],
        }
    previous_payload = _read_previous_telemetry(telemetry_path)
    previous_stores = previous_payload.get("stores", {}) if isinstance(previous_payload, dict) else {}
    actions: list[dict[str, Any]] = []
    disk_pressure = free_bytes < manifest["minimum_free_bytes"]
    for plan in plans:
        store_id = plan["store"]["id"]
        rotation_failed = False
        for operation in plan["operations"]:
            kind = operation["kind"]
            if kind == "prune" and rotation_failed:
                actions.append(
                    {
                        "store_id": store_id,
                        "kind": "prune_skipped_rotation_failed",
                        "bytes": operation["record"]["size"],
                        "success": False,
                    }
                )
                continue
            if kind == "compact" and disk_pressure:
                actions.append(
                    {
                        "store_id": store_id,
                        "kind": "compaction_skipped_disk_pressure",
                        "bytes": operation["record"]["size"],
                    }
                )
                continue
            if dry_run:
                actions.append(
                    {
                        "store_id": store_id,
                        "kind": f"would_{'rotate' if kind == 'rotate' else 'prune' if kind == 'prune' else 'compact'}",
                        "bytes": operation["record"]["size"],
                    }
                )
                continue
            success, outcome = (
                _rotate(operation)
                if kind == "rotate"
                else _prune(operation)
                if kind == "prune"
                else _compact(operation)
            )
            if kind == "rotate" and not success:
                rotation_failed = True
            if outcome != "compaction_not_needed":
                actions.append(
                    {
                        "store_id": store_id,
                        "kind": outcome,
                        "bytes": operation["record"]["size"],
                        "success": success,
                    }
                )

    stores = [
        _store_report(root_path, plan, previous=previous_stores, now=sampled_at)
        for plan in plans
    ]
    status = "pressure" if disk_pressure or any(item["status"] != "within_budget" for item in stores) else "within_budget"
    if any(action.get("success") is False for action in actions):
        status = "degraded"
    report = {
        "schema": REPORT_SCHEMA,
        "ok": True,
        "status": status,
        "dry_run": bool(dry_run),
        "sampled_at_unix": sampled_at,
        "schedule_seconds": manifest["schedule_seconds"],
        "disk_free_bytes": free_bytes,
        "minimum_free_bytes": manifest["minimum_free_bytes"],
        "disk_pressure": disk_pressure,
        "stores": stores,
        "actions": actions,
    }
    configured_store_ids = {plan["store"]["id"] for plan in plans}
    unmanaged_overrides = sorted(
        name
        for name, store_id in PATH_OVERRIDE_STORES.items()
        if store_id in configured_store_ids and os.environ.get(name)
    )
    report["unmanaged_path_overrides"] = unmanaged_overrides
    if unmanaged_overrides:
        report["status"] = "degraded"
    if not dry_run:
        telemetry = {
            "schema": "HiveMindRetentionTelemetry.v1",
            "sampled_at_unix": sampled_at,
            "stores": {
                item["id"]: {
                    "sampled_at_unix": sampled_at,
                    "current_bytes": item["current_bytes"],
                    "growth_bytes_per_day": item["growth_bytes_per_day"],
                    "forecast_bytes_24h": item["forecast_bytes_24h"],
                    "forecast_status": item["forecast_status"],
                }
                for item in stores
            },
        }
        try:
            _write_json_atomic(telemetry_path, telemetry)
            report["telemetry"] = "written"
        except Exception:
            report["telemetry"] = "write_failed"
            report["status"] = "pressure"
            log.warning("retention telemetry write failed")
    return report


def run_retention_cycle(
    *,
    root: Path | str = DEFAULT_ROOT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Scheduler-facing apply entry point."""

    return run_cycle(root=root, manifest_path=manifest_path, dry_run=False)


def retention_schedule_seconds(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> int:
    """Return the declared cadence, falling back safely if policy is corrupt."""

    try:
        return int(load_manifest(manifest_path)["schedule_seconds"])
    except ManifestError:
        return 300


def _find_store(manifest: dict[str, Any], store_id: str) -> dict[str, Any] | None:
    return next((store for store in manifest["stores"] if store["id"] == store_id), None)


def _authorization_receipt(
    authorize: AuthorizationFn | None,
    action: str,
    store_id: str,
) -> str | None:
    if authorize is None:
        return None
    try:
        receipt_id = authorize(action, store_id)
    except Exception:
        return None
    if not isinstance(receipt_id, str) or not AUTHORIZATION_ID.fullmatch(receipt_id):
        return None
    return "sha256:" + hashlib.sha256(receipt_id.encode("utf-8")).hexdigest()


def _append_receipt(root: Path, manifest: dict[str, Any], receipt: dict[str, Any]) -> bool:
    path = root / manifest["receipt_path"]
    if not _beneath(root, path):
        return False
    payload = {
        **receipt,
        "receipt_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except Exception:
        log.warning("retention receipt write failed")
        return False


def _refusal(action: str, store_id: str) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "action": action,
        "store_id": store_id,
        "outcome": "refused",
        "reason": "authorization_required",
    }


def _quiescence_proven(quiescent: QuiescenceFn | None, store_id: str) -> bool:
    if quiescent is None:
        return False
    try:
        return quiescent(store_id) is True
    except Exception:
        return False


def _store_records(root: Path, store: dict[str, Any]) -> list[dict[str, Any]]:
    records = _expand(root, store["paths"], store["policy"]["hold_suffix"])
    _mark_active(root, store, records)
    return records


def _opened_generation_matches(record: dict[str, Any], stat: os.stat_result) -> bool:
    return (
        int(stat.st_dev) == record["device"]
        and int(stat.st_ino) == record["inode"]
        and int(stat.st_size) == record["size"]
        and int(stat.st_mtime_ns) == record["mtime_ns"]
    )


def _write_record_to_zip(bundle: zipfile.ZipFile, record: dict[str, Any]) -> None:
    """Archive the already-scanned inode, never a later path substitution."""

    if not _same_generation(record):
        raise OSError("source generation changed")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(record["path"], flags)
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if not _opened_generation_matches(record, opened):
            raise OSError("opened source generation changed")
        info = zipfile.ZipInfo(record["relative"], date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (int(record.get("mode") or 0o600) & 0xFFFF) << 16
        remaining = record["size"]
        with bundle.open(info, "w", force_zip64=True) as output:
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("source truncated during export")
                output.write(chunk)
                remaining -= len(chunk)
        closed = os.fstat(source.fileno())
        if not _opened_generation_matches(record, closed):
            raise OSError("source changed during export")
    if not _same_generation(record):
        raise OSError("source path changed during export")


def _sqlite_backup_record(
    root: Path,
    records: list[dict[str, Any]],
    staging_dir: Path,
) -> dict[str, Any]:
    databases = [record for record in records if record["path"].suffix == ".sqlite3"]
    if len(databases) != 1 or not _same_generation(databases[0]):
        raise OSError("SQLite source is unavailable")
    source_record = databases[0]
    snapshot = staging_dir / "sqlite_snapshot.sqlite3"
    source_uri = f"file:{source_record['path'].resolve().as_posix()}?mode=ro"
    source = sqlite3.connect(source_uri, uri=True, timeout=5.0)
    target = sqlite3.connect(snapshot)
    try:
        source.backup(target)
        check = target.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise OSError("SQLite backup failed integrity check")
    finally:
        target.close()
        source.close()
    if not _same_generation(source_record):
        raise OSError("SQLite source changed during backup")
    stat = snapshot.stat()
    return {
        "path": snapshot,
        "relative": source_record["relative"],
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "mtime_ns": int(stat.st_mtime_ns),
        "mode": 0o600,
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "root": staging_dir,
        "hold_path": staging_dir / ".never-held",
        "held": False,
        "active": False,
    }


def export_store(
    *,
    root: Path | str,
    manifest_path: Path | str,
    store_id: str,
    destination: Path | str,
    authorize: AuthorizationFn | None = None,
    quiescent: QuiescenceFn | None = None,
) -> dict[str, Any]:
    """Export one store only after an injected operator authorization check."""

    root_path = Path(root).resolve()
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError:
        return _refusal("export", "unknown")
    store = _find_store(manifest, store_id)
    if store is None:
        receipt = {**_refusal("export", "unknown"), "reason": "store_not_authorized"}
        _append_receipt(root_path, manifest, receipt)
        return receipt
    authorization_id = _authorization_receipt(authorize, "export", store["id"])
    if authorization_id is None:
        receipt = _refusal("export", store["id"])
        _append_receipt(root_path, manifest, receipt)
        return receipt
    authorized = {
        "schema": RECEIPT_SCHEMA,
        "action": "export",
        "store_id": store_id,
        "outcome": "authorized",
        "authorization_fingerprint": authorization_id,
    }
    if not _append_receipt(root_path, manifest, authorized):
        return {
            **authorized,
            "outcome": "failed",
            "reason": "receipt_write_failed",
        }
    try:
        records = _store_records(root_path, store)
        sqlite_store = store["policy"]["rotation"] == "sqlite_vacuum"
        requires_quiescence = store["data_class"] == PRIVATE_CLASS or any(
            record["active"] for record in records
        )
        if requires_quiescence and not _quiescence_proven(quiescent, store["id"]):
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "action": "export",
                "store_id": store["id"],
                "outcome": "refused",
                "reason": "quiescence_required",
                "authorization_fingerprint": authorization_id,
            }
            _append_receipt(root_path, manifest, receipt)
            return receipt
        target = Path(destination).resolve()
        if target.exists() or any(target == record["path"].resolve() for record in records):
            raise OSError("unsafe export target")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        os.close(descriptor)
        try:
            with tempfile.TemporaryDirectory(prefix=".retention-export-", dir=target.parent) as staging:
                export_records = records
                if sqlite_store:
                    export_records = [
                        _sqlite_backup_record(root_path, records, Path(staging))
                    ]
                with zipfile.ZipFile(
                    temporary,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    strict_timestamps=False,
                ) as bundle:
                    for record in sorted(export_records, key=lambda item: item["relative"]):
                        _write_record_to_zip(bundle, record)
            with open(temporary, "r+b") as handle:
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.link(temporary, target)
            os.unlink(temporary)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        hasher = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "action": "export",
            "store_id": store_id,
            "outcome": "exported",
            "authorization_fingerprint": authorization_id,
            "file_count": len(export_records),
            "bytes": sum(record["size"] for record in export_records),
            "archive_sha256": digest,
        }
    except Exception as exc:
        log.warning("retention export failed reason=%s", type(exc).__name__)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "action": "export",
            "store_id": store_id,
            "outcome": "failed",
            "reason": "export_failed",
            "authorization_fingerprint": authorization_id,
        }
    if not _append_receipt(root_path, manifest, receipt) and receipt["outcome"] == "exported":
        receipt = {
            **receipt,
            "outcome": "committed_receipt_failed",
            "reason": "receipt_write_failed",
        }
    return receipt


def delete_store(
    *,
    root: Path | str,
    manifest_path: Path | str,
    store_id: str,
    authorize: AuthorizationFn | None = None,
    quiescent: QuiescenceFn | None = None,
) -> dict[str, Any]:
    """Delete one store only after authorization, preserving any held store."""

    root_path = Path(root).resolve()
    try:
        manifest = load_manifest(manifest_path)
    except ManifestError:
        return _refusal("delete", "unknown")
    store = _find_store(manifest, store_id)
    if store is None:
        receipt = {**_refusal("delete", "unknown"), "reason": "store_not_authorized"}
        _append_receipt(root_path, manifest, receipt)
        return receipt
    authorization_id = _authorization_receipt(authorize, "delete", store["id"])
    if authorization_id is None:
        receipt = _refusal("delete", store["id"])
        _append_receipt(root_path, manifest, receipt)
        return receipt
    authorized = {
        "schema": RECEIPT_SCHEMA,
        "action": "delete",
        "store_id": store_id,
        "outcome": "authorized",
        "authorization_fingerprint": authorization_id,
    }
    if not _append_receipt(root_path, manifest, authorized):
        return {
            **authorized,
            "outcome": "failed",
            "reason": "receipt_write_failed",
        }
    records = _store_records(root_path, store)
    if not _quiescence_proven(quiescent, store["id"]):
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "action": "delete",
            "store_id": store["id"],
            "outcome": "refused",
            "reason": "quiescence_required",
            "authorization_fingerprint": authorization_id,
        }
        _append_receipt(root_path, manifest, receipt)
        return receipt
    if any(record["held"] for record in records):
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "action": "delete",
            "store_id": store_id,
            "outcome": "refused",
            "reason": "evidence_hold",
            "authorization_fingerprint": authorization_id,
        }
        _append_receipt(root_path, manifest, receipt)
        return receipt
    staged: list[tuple[Path, Path]] = []
    try:
        for record in records:
            if not _same_generation(record) or _hold_present(record):
                raise OSError("source generation changed")
        for record in records:
            if _hold_present(record):
                raise OSError("evidence hold appeared")
            source: Path = record["path"]
            tombstone = source.with_name(f".{source.name}.authorized-delete.{uuid.uuid4().hex}")
            source.rename(tombstone)
            staged.append((source, tombstone))
            if _hold_present(record):
                raise OSError("evidence hold appeared")
    except Exception:
        for source, tombstone in reversed(staged):
            try:
                if tombstone.exists() and not source.exists():
                    tombstone.rename(source)
            except OSError:
                pass
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "action": "delete",
            "store_id": store_id,
            "outcome": "failed",
            "reason": "delete_staging_failed",
            "authorization_fingerprint": authorization_id,
        }
        _append_receipt(root_path, manifest, receipt)
        return receipt
    if any(_hold_present(record) for record in records):
        for source, tombstone in reversed(staged):
            try:
                if tombstone.exists() and not source.exists():
                    tombstone.rename(source)
            except OSError:
                pass
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "action": "delete",
            "store_id": store_id,
            "outcome": "refused",
            "reason": "evidence_hold",
            "authorization_fingerprint": authorization_id,
        }
        _append_receipt(root_path, manifest, receipt)
        return receipt
    deleted = 0
    for _source, tombstone in staged:
        try:
            tombstone.unlink()
            deleted += 1
        except OSError:
            break
    if deleted != len(staged):
        for source, tombstone in staged[deleted:]:
            try:
                if tombstone.exists() and not source.exists():
                    tombstone.rename(source)
            except OSError:
                pass
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "action": "delete",
        "store_id": store_id,
        "outcome": "deleted" if deleted == len(staged) else "failed",
        "authorization_fingerprint": authorization_id,
        "file_count": deleted,
        "bytes": sum(record["size"] for record in records[:deleted]),
    }
    if deleted != len(staged):
        receipt["reason"] = "delete_commit_failed"
    if not _append_receipt(root_path, manifest, receipt) and receipt["outcome"] == "deleted":
        receipt = {
            **receipt,
            "outcome": "committed_receipt_failed",
            "reason": "receipt_write_failed",
        }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report or enforce the MS3/MS4 retention manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Report only; this is the default.")
    mode.add_argument("--apply", action="store_true", help="Apply operational rotation/pruning.")
    args = parser.parse_args(argv)
    report = run_cycle(root=args.root, manifest_path=args.manifest, dry_run=not args.apply)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report.get("ok"):
        return 2
    return 0 if report.get("status") == "within_budget" else 1


if __name__ == "__main__":
    raise SystemExit(main())
