from __future__ import annotations

import importlib.util
import json
import logging
import os
import sqlite3
import sys
import types
import zipfile
from pathlib import Path

from machine_spirit_4 import retention


ROOT = Path(__file__).resolve().parents[2]


def _store(
    store_id: str,
    pattern: str,
    *,
    data_class: str = "operational",
    max_files: int | None = 2,
    max_bytes: int | None = 1_000_000,
    max_age_days: int | None = 30,
    rotation: str = "prune_files",
    active_paths: list[str] | None = None,
    compact_at_bytes: int | None = None,
    auto_prune: bool | None = None,
) -> dict:
    return {
        "id": store_id,
        "data_class": data_class,
        "sensitivity": "private_payload" if data_class == "private_psyche" else "test_metadata",
        "enforcement": "fixture_policy",
        "paths": [pattern],
        "active_paths": active_paths or [],
        "writer": "pytest fixture writer",
        "growth_driver": "seeded fixture writes",
        "access": {
            "read": "local_process_owner",
            "export": "human_approval_required",
            "delete": "human_approval_required",
            "at_rest": "operator_policy_required",
        },
        "policy": {
            "max_age_days": max_age_days,
            "max_bytes": max_bytes,
            "max_files": max_files,
            "rotation": rotation,
            "rotate_at_bytes": 10 if rotation == "rename_recreate" else None,
            "compaction": "sqlite_vacuum" if rotation == "sqlite_vacuum" else "none",
            "compact_at_bytes": compact_at_bytes,
            "auto_prune": data_class != "private_psyche" if auto_prune is None else auto_prune,
            "hold_suffix": ".hold",
        },
    }


def _write_manifest(root: Path, stores: list[dict], *, minimum_free_bytes: int = 0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "retention_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": retention.MANIFEST_SCHEMA,
                "schedule_seconds": 300,
                "minimum_free_bytes": minimum_free_bytes,
                "budget_authority": "pytest_fixture",
                "disk_pressure_behavior": "fixture_pruning_and_reporting",
                "telemetry_path": ".retention/telemetry.json",
                "receipt_path": ".retention/receipts.jsonl",
                "authorization_provider": "OPERATOR_INTEGRATION_REQUIRED",
                "stores": stores,
            }
        ),
        encoding="utf-8",
    )
    return path


def _seed(path: Path, payload: bytes, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, (mtime, mtime))


def test_checked_in_manifest_covers_required_store_classes() -> None:
    manifest = retention.load_manifest(ROOT / "machine_spirit_4" / "retention_manifest.json")
    stores = {store["id"]: store for store in manifest["stores"]}

    assert {
        "ms3_psyche_private_state",
        "ms4_audit_log",
        "ms4_voice_turn_ring",
        "ms4_runtime_sqlite",
        "hermes_updater_state",
        "hermes_updater_runtime",
        "ms4_runtime_registry_state",
        "ms4_stdout_stderr",
        "ms4_supervisor_logs",
        "ms4_canned_audio_cache",
        "tmr_psyche_live_collection",
        "tmr_psyche_consolidated_private",
        "tmr_psyche_generated_datasets",
        "ms4_retention_telemetry",
        "ms4_retention_receipts",
    } <= stores.keys()
    assert stores["ms3_psyche_private_state"]["data_class"] == "private_psyche"
    assert stores["ms3_psyche_private_state"]["policy"]["auto_prune"] is False
    for store in stores.values():
        assert store["paths"]
        assert store["writer"]
        assert store["growth_driver"]
        assert store["sensitivity"]
        assert store["enforcement"]
        assert store["policy"]["max_bytes"] is not None or store["policy"]["max_files"] is not None
        assert set(store["access"]) == {"read", "export", "delete", "at_rest"}
    assert "tmr-psyche/training_runs/**/*" in stores["tmr_psyche_training_runs"]["paths"]
    assert manifest["budget_authority"] == "tmr_p1_002_code_owned_safety_defaults"
    assert {
        "MS4_AUDIT_LOG",
        "MS4_VOICE_TURNS_LOG",
        "MS4_REFLEX_DIR",
        "MS4_VOICE_CANDIDATE_REVIEW_STORE",
        "MS4_QM_INDEX_DIR",
        "MS4_MCP_IMPORTS_PATH",
        "MS4_HERMES_DIR",
    } <= retention.PATH_OVERRIDE_STORES.keys()


def test_below_bounds_changes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log")])
    _seed(root / "logs/a.log", b"a", 100)
    _seed(root / "logs/b.log", b"b", 200)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=10_000,
        quiescent_store_ids={"logs"},
    )

    assert report["ok"] is True
    assert report["actions"] == []
    assert sorted(path.name for path in (root / "logs").iterdir()) == ["a.log", "b.log"]


def test_exactly_oldest_overflow_is_pruned(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log")])
    _seed(root / "logs/old.log", b"old", 100)
    _seed(root / "logs/middle.log", b"middle", 200)
    _seed(root / "logs/new.log", b"new", 300)

    report = retention.run_cycle(root=root, manifest_path=manifest, now=1_000, disk_free_bytes=10_000)

    assert not (root / "logs/old.log").exists()
    assert (root / "logs/middle.log").read_bytes() == b"middle"
    assert (root / "logs/new.log").read_bytes() == b"new"
    assert [action["kind"] for action in report["actions"]] == ["pruned"]


def test_age_and_total_size_each_prune_oldest_eligible_file(tmp_path: Path) -> None:
    age_root = tmp_path / "age"
    age_manifest = _write_manifest(
        age_root,
        [_store("logs", "logs/*.log", max_files=10, max_bytes=1_000, max_age_days=1)],
    )
    _seed(age_root / "logs/expired.log", b"old", 1)
    _seed(age_root / "logs/fresh.log", b"fresh", 2 * 86_400)

    age_report = retention.run_cycle(
        root=age_root,
        manifest_path=age_manifest,
        now=2 * 86_400 + 1,
        disk_free_bytes=10_000,
    )

    assert not (age_root / "logs/expired.log").exists()
    assert (age_root / "logs/fresh.log").exists()
    assert [action["kind"] for action in age_report["actions"]] == ["pruned"]

    size_root = tmp_path / "size"
    size_manifest = _write_manifest(
        size_root,
        [_store("logs", "logs/*.log", max_files=10, max_bytes=7, max_age_days=30)],
    )
    _seed(size_root / "logs/old.log", b"1234", 100)
    _seed(size_root / "logs/new.log", b"5678", 200)

    size_report = retention.run_cycle(
        root=size_root,
        manifest_path=size_manifest,
        now=1_000,
        disk_free_bytes=10_000,
    )

    assert not (size_root / "logs/old.log").exists()
    assert (size_root / "logs/new.log").read_bytes() == b"5678"
    assert [action["kind"] for action in size_report["actions"]] == ["pruned"]


def test_private_store_and_evidence_hold_are_never_auto_pruned(tmp_path: Path) -> None:
    root = tmp_path / "root"
    private = _store(
        "psyche",
        "psyche/**/*.json",
        data_class="private_psyche",
        max_files=1,
        max_bytes=1,
        max_age_days=0,
        rotation="none",
    )
    operational = _store("logs", "logs/*.log", max_files=2)
    manifest = _write_manifest(root, [private, operational], minimum_free_bytes=100)
    _seed(root / "psyche/sister/a.json", b"private-one", 10)
    _seed(root / "psyche/sister/b.json", b"private-two", 20)
    _seed(root / "logs/held.log", b"held", 10)
    _seed(root / "logs/held.log.hold", b"retention hold", 10)
    _seed(root / "logs/overflow.log", b"overflow", 20)
    _seed(root / "logs/new.log", b"new", 30)

    report = retention.run_cycle(root=root, manifest_path=manifest, now=1_000, disk_free_bytes=0)

    assert (root / "psyche/sister/a.json").read_bytes() == b"private-one"
    assert (root / "psyche/sister/b.json").read_bytes() == b"private-two"
    assert (root / "logs/held.log").read_bytes() == b"held"
    assert not (root / "logs/overflow.log").exists()
    assert report["disk_pressure"] is True
    psyche = next(item for item in report["stores"] if item["id"] == "psyche")
    assert psyche["status"] == "operator_action_required"
    assert psyche["held_files"] == 0
    assert psyche["age_over_budget"] is True


def test_hold_added_after_planning_prevents_prune(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest_path = _write_manifest(root, [_store("logs", "logs/*.log", max_files=0)])
    target = root / "logs/keep.log"
    _seed(target, b"evidence", 100)
    store = retention.load_manifest(manifest_path)["stores"][0]
    plan = retention._plan_store(root, store, 1_000, quiescent=False)
    assert plan["operations"][0]["kind"] == "prune"
    _seed(Path(f"{target}.hold"), b"late hold", 200)

    success, outcome = retention._prune(plan["operations"][0])

    assert success is False
    assert outcome in {"generation_changed", "evidence_hold"}
    assert target.read_bytes() == b"evidence"


def test_corrupt_manifest_disables_all_pruning_without_logging_payload(
    tmp_path: Path, caplog,
) -> None:
    root = tmp_path / "root"
    private_payload = "PRIVATE-PAYLOAD-MUST-NOT-LOG"
    target = root / "logs/keep.log"
    _seed(target, private_payload.encode(), 1)
    manifest = root / "retention_manifest.json"
    manifest.write_text('{"secret":"PRIVATE-PAYLOAD-MUST-NOT-LOG"', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        report = retention.run_cycle(root=root, manifest_path=manifest, now=1_000)

    assert report == {
        "schema": retention.REPORT_SCHEMA,
        "ok": False,
        "reason": "manifest_invalid",
        "actions": [],
    }
    assert target.read_text(encoding="utf-8") == private_payload
    assert private_payload not in caplog.text


def test_overlapping_store_ownership_disables_pruning(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(
        root,
        [
            _store("logs_a", "logs/*.log", max_files=0),
            _store("logs_b", "logs/keep.log", max_files=0),
        ],
    )
    target = root / "logs/keep.log"
    _seed(target, b"keep", 1)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=10_000,
    )

    assert report["ok"] is False
    assert report["reason"] == "manifest_invalid"
    assert target.read_bytes() == b"keep"


def test_oversized_active_log_rotates_by_atomic_rename(tmp_path: Path) -> None:
    root = tmp_path / "root"
    store = _store(
        "logs",
        "logs/app.log*",
        max_files=3,
        rotation="rename_recreate",
        active_paths=["logs/app.log"],
    )
    manifest = _write_manifest(root, [store])
    _seed(root / "logs/app.log", b"0123456789ABCDEF", 100)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=10_000,
        quiescent_store_ids={"logs"},
    )

    assert (root / "logs/app.log").read_bytes() == b""
    retained = list((root / "logs").glob("app.log.retained.*"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == b"0123456789ABCDEF"
    assert any(action["kind"] == "rotated" for action in report["actions"])


def test_scheduled_cycle_reports_active_rotation_without_touching_writer(tmp_path: Path) -> None:
    root = tmp_path / "root"
    store = _store(
        "logs",
        "logs/app.log*",
        max_files=3,
        rotation="rename_recreate",
        active_paths=["logs/app.log"],
    )
    manifest = _write_manifest(root, [store])
    active = root / "logs/app.log"
    _seed(active, b"0123456789ABCDEF", 100)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=10_000,
    )

    assert active.read_bytes() == b"0123456789ABCDEF"
    assert report["actions"] == []
    assert report["stores"][0]["rotation_pending"] is True
    assert report["stores"][0]["status"] == "quiescence_required"


def test_rotation_failure_preserves_original_and_marks_report_degraded(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "root"
    store = _store(
        "logs",
        "logs/app.log*",
        max_files=3,
        rotation="rename_recreate",
        active_paths=["logs/app.log"],
    )
    manifest = _write_manifest(root, [store])
    active = root / "logs/app.log"
    _seed(active, b"private log payload", 100)
    _seed(root / "logs/app.log.old1", b"old1", 10)
    _seed(root / "logs/app.log.old2", b"old2", 20)
    real_rename = Path.rename

    def fail_active_rename(path: Path, target: Path):
        if path == active:
            raise PermissionError("simulated sharing violation")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_active_rename)
    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=10_000,
        quiescent_store_ids={"logs"},
    )

    assert active.read_bytes() == b"private log payload"
    assert (root / "logs/app.log.old1").read_bytes() == b"old1"
    assert (root / "logs/app.log.old2").read_bytes() == b"old2"
    assert report["status"] == "degraded"
    assert report["actions"][0] == {
        "store_id": "logs",
        "kind": "rotation_failed",
        "bytes": 19,
        "success": False,
    }
    assert report["actions"][1]["kind"] == "prune_skipped_rotation_failed"


def test_dry_run_reports_without_mutation(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log", max_files=1)])
    _seed(root / "logs/old.log", b"old", 100)
    _seed(root / "logs/new.log", b"new", 200)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        dry_run=True,
        now=1_000,
        disk_free_bytes=10_000,
    )

    assert (root / "logs/old.log").exists()
    assert (root / "logs/new.log").exists()
    assert [action["kind"] for action in report["actions"]] == ["would_prune"]
    assert not (root / ".retention/telemetry.json").exists()


def test_cli_defaults_to_metadata_only_dry_run(tmp_path: Path, capsys) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log", max_files=0)])
    target = root / "logs/keep.log"
    _seed(target, b"keep", 100)

    code = retention.main(["--root", str(root), "--manifest", str(manifest)])

    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["dry_run"] is True
    assert report["actions"][0]["kind"] == "would_prune"
    assert target.read_bytes() == b"keep"


def test_capacity_telemetry_forecasts_from_two_bounded_samples(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log", max_files=10)])
    _seed(root / "logs/a.log", b"a" * 100, 100)

    retention.run_cycle(root=root, manifest_path=manifest, now=1_000, disk_free_bytes=10_000)
    _seed(root / "logs/b.log", b"b" * 100, 200)
    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000 + 86_400,
        disk_free_bytes=10_000,
    )

    logs = report["stores"][0]
    assert logs["current_bytes"] == 200
    assert logs["growth_bytes_per_day"] == 100
    assert logs["forecast_bytes_24h"] == 300
    assert logs["forecast_bytes_24h"] <= logs["max_bytes"]
    telemetry = json.loads((root / ".retention/telemetry.json").read_text(encoding="utf-8"))
    assert list(telemetry["stores"]) == ["logs"]
    assert telemetry["stores"]["logs"]["forecast_bytes_24h"] == 300


def test_sqlite_compaction_reclaims_free_pages_without_deleting_rows(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = root / "runtime/jobs.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, body BLOB)")
        connection.executemany(
            "INSERT INTO records(body) VALUES (?)",
            [(b"x" * 8_000,) for _ in range(200)],
        )
        connection.execute("DELETE FROM records WHERE id <= 190")
        connection.commit()
    before = db.stat().st_size
    store = _store(
        "sqlite",
        "runtime/*.sqlite3",
        max_files=3,
        max_bytes=10_000_000,
        rotation="sqlite_vacuum",
        compact_at_bytes=1,
        auto_prune=False,
    )
    manifest = _write_manifest(root, [store])

    report = retention.run_cycle(root=root, manifest_path=manifest, now=1_000, disk_free_bytes=10_000)

    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 10
    assert db.stat().st_size < before
    assert any(action["kind"] == "compacted" for action in report["actions"])


def test_held_sqlite_is_not_compacted_even_when_quiescent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = root / "runtime/jobs.sqlite3"
    _seed(db, b"held database bytes", 100)
    _seed(Path(f"{db}.hold"), b"hold", 100)
    store = _store(
        "sqlite",
        "runtime/*.sqlite3",
        max_files=3,
        max_bytes=10_000,
        rotation="sqlite_vacuum",
        compact_at_bytes=1,
        auto_prune=False,
    )
    manifest = _write_manifest(root, [store])

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=10_000,
        quiescent_store_ids={"sqlite"},
    )

    assert report["actions"] == []
    assert db.read_bytes() == b"held database bytes"


def test_disk_pressure_skips_space_hungry_sqlite_compaction(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = root / "runtime/jobs.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE records (body BLOB)")
        connection.executemany(
            "INSERT INTO records(body) VALUES (?)",
            [(b"x" * 8_000,) for _ in range(50)],
        )
        connection.execute("DELETE FROM records")
        connection.commit()
    before = db.stat().st_size
    store = _store(
        "sqlite",
        "runtime/*.sqlite3",
        max_files=3,
        max_bytes=10_000_000,
        rotation="sqlite_vacuum",
        compact_at_bytes=1,
        auto_prune=False,
    )
    manifest = _write_manifest(root, [store], minimum_free_bytes=100)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        now=1_000,
        disk_free_bytes=0,
        quiescent_store_ids={"sqlite"},
    )

    assert report["status"] == "pressure"
    assert report["actions"][0]["kind"] == "compaction_skipped_disk_pressure"
    assert db.stat().st_size == before


def test_sqlite_export_uses_consistent_backup_instead_of_raw_wal_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    db = root / "runtime/jobs.sqlite3"
    db.parent.mkdir(parents=True)
    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE records (body TEXT)")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("INSERT INTO records(body) VALUES ('committed-before-export')")
        connection.commit()
        store = _store(
            "sqlite",
            "runtime/*.sqlite3*",
            max_files=3,
            max_bytes=10_000_000,
            rotation="sqlite_vacuum",
            active_paths=["runtime/*.sqlite3*"],
            compact_at_bytes=1,
            auto_prune=False,
        )
        manifest = _write_manifest(root, [store])
        archive = tmp_path / "sqlite.zip"

        exported = retention.export_store(
            root=root,
            manifest_path=manifest,
            store_id="sqlite",
            destination=archive,
            authorize=lambda _action, _store_id: "approval-sqlite",
            quiescent=lambda _store_id: True,
        )
    finally:
        connection.close()

    assert exported["outcome"] == "exported"
    assert exported["file_count"] == 1
    extract = tmp_path / "extract"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extract)
    with sqlite3.connect(extract / "runtime/jobs.sqlite3") as restored:
        assert restored.execute("SELECT body FROM records").fetchone()[0] == "committed-before-export"


def test_export_delete_require_injected_authorization_and_emit_receipts(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(
        root,
        [_store("psyche", "psyche/**/*.json", data_class="private_psyche", rotation="none")],
    )
    source = root / "psyche/sister/state.json"
    _seed(source, b"private state", 100)
    archive = tmp_path / "export.zip"

    refused = retention.export_store(
        root=root,
        manifest_path=manifest,
        store_id="psyche",
        destination=archive,
    )
    assert refused == {
        "schema": retention.RECEIPT_SCHEMA,
        "action": "export",
        "store_id": "psyche",
        "outcome": "refused",
        "reason": "authorization_required",
    }
    assert not archive.exists()
    assert source.exists()

    def authorize(action: str, store_id: str) -> str:
        return f"approved-{action}-{store_id}"

    exported = retention.export_store(
        root=root,
        manifest_path=manifest,
        store_id="psyche",
        destination=archive,
        authorize=authorize,
        quiescent=lambda _store_id: True,
    )
    assert exported["outcome"] == "exported"
    assert exported["file_count"] == 1
    assert exported["authorization_fingerprint"].startswith("sha256:")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("psyche/sister/state.json") == b"private state"

    refused_delete = retention.delete_store(
        root=root,
        manifest_path=manifest,
        store_id="psyche",
    )
    assert refused_delete["outcome"] == "refused"
    assert "file_count" not in refused_delete
    assert source.exists()

    deleted = retention.delete_store(
        root=root,
        manifest_path=manifest,
        store_id="psyche",
        authorize=authorize,
        quiescent=lambda _store_id: True,
    )
    assert deleted["outcome"] == "deleted"
    assert deleted["file_count"] == 1
    assert not source.exists()
    receipts = (root / ".retention/receipts.jsonl").read_text(encoding="utf-8")
    assert "private state" not in receipts
    assert "state.json" not in receipts
    assert "approved-export-psyche" not in receipts


def test_invalid_store_id_is_not_echoed_or_persisted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log")])
    sentinel = "PRIVATE_PAYLOAD_SENTINEL"

    receipt = retention.export_store(
        root=root,
        manifest_path=manifest,
        store_id=sentinel,
        destination=tmp_path / "never.zip",
    )

    assert receipt["store_id"] == "unknown"
    assert sentinel not in json.dumps(receipt)
    persisted = (root / ".retention/receipts.jsonl").read_text(encoding="utf-8")
    assert sentinel not in persisted


def test_receipt_preflight_failure_prevents_authorized_export_and_delete(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("logs", "logs/*.log")])
    source = root / "logs/keep.log"
    _seed(source, b"keep", 100)
    monkeypatch.setattr(retention, "_append_receipt", lambda *_args, **_kwargs: False)

    def authorize(_action: str, _store_id: str) -> str:
        return "approval-1"

    def quiescent(_store_id: str) -> bool:
        return True

    exported = retention.export_store(
        root=root,
        manifest_path=manifest,
        store_id="logs",
        destination=tmp_path / "never.zip",
        authorize=authorize,
        quiescent=quiescent,
    )
    deleted = retention.delete_store(
        root=root,
        manifest_path=manifest,
        store_id="logs",
        authorize=authorize,
        quiescent=quiescent,
    )

    assert exported["reason"] == "receipt_write_failed"
    assert deleted["reason"] == "receipt_write_failed"
    assert not (tmp_path / "never.zip").exists()
    assert source.read_bytes() == b"keep"


def test_export_rejects_opened_inode_substitution(
    tmp_path: Path, monkeypatch, caplog,
) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(
        root,
        [_store("psyche", "psyche/*.json", data_class="private_psyche", rotation="none")],
    )
    source = root / "psyche/state.json"
    outside = tmp_path / "outside.json"
    _seed(source, b"inside", 100)
    _seed(outside, b"outside-private-payload", 100)
    real_open = os.open

    def substitute_open(path, flags, mode=0o777):
        selected = outside if Path(path) == source else path
        return real_open(selected, flags, mode)

    monkeypatch.setattr(retention.os, "open", substitute_open)
    archive = tmp_path / "never.zip"
    with caplog.at_level(logging.WARNING):
        receipt = retention.export_store(
            root=root,
            manifest_path=manifest,
            store_id="psyche",
            destination=archive,
            authorize=lambda _action, _store_id: "approval-1",
            quiescent=lambda _store_id: True,
        )

    assert receipt["outcome"] == "failed"
    assert not archive.exists()
    assert "outside-private-payload" not in caplog.text


def test_configured_path_override_is_named_but_value_is_not_disclosed(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "root"
    manifest = _write_manifest(root, [_store("ms4_audit_log", "logs/audit.jsonl")])
    sentinel = "PRIVATE_OVERRIDE_PATH_SENTINEL"
    monkeypatch.setenv("MS4_AUDIT_LOG", sentinel)

    report = retention.run_cycle(
        root=root,
        manifest_path=manifest,
        dry_run=True,
        now=1_000,
        disk_free_bytes=10_000,
    )

    assert report["status"] == "degraded"
    assert report["unmanaged_path_overrides"] == ["MS4_AUDIT_LOG"]
    assert sentinel not in json.dumps(report)


def test_supervisor_reuses_existing_watch_cadence_for_retention() -> None:
    source = (ROOT / "machine_spirit_4/scripts/supervise_ms4.py").read_text(encoding="utf-8")

    assert "run_retention_cycle" in source
    assert "retention outcome=" in source


def test_supervisor_calls_retention_once_and_continues_on_cycle_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    runtime_root = tmp_path / "tmr"
    ms4 = runtime_root / "machine_spirit_4"
    python = ms4 / ".venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    runtime = types.ModuleType("runtime_common")
    runtime.MS4 = ms4
    runtime.ROOT = runtime_root
    runtime.is_port_listening = lambda *_args: False
    runtime.venv_python = lambda: python
    monkeypatch.setitem(sys.modules, "runtime_common", runtime)
    path = ROOT / "machine_spirit_4/scripts/supervise_ms4.py"
    spec = importlib.util.spec_from_file_location("retention_supervisor_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    before_path = list(sys.path)
    spec.loader.exec_module(module)
    sys.path[:] = before_path
    calls: list[dict] = []
    child_calls: list[object] = []
    logs: list[str] = []
    child_log = tmp_path / "child.log"
    monkeypatch.setattr(module, "run_retention_cycle", lambda **kwargs: calls.append(kwargs) or {"status": "within_budget", "actions": []})
    monkeypatch.setattr(module, "down_ports", lambda: [9180])
    monkeypatch.setattr(module, "hivemind_health_status", lambda: "unreachable")
    monkeypatch.setattr(module, "write_log", logs.append)
    monkeypatch.setattr(module, "log_path", lambda: child_log)
    def fake_child(*args, **_kwargs):
        child_calls.append((args, _kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_child)

    assert module.supervise_once() == 0
    assert calls == [{"root": runtime_root, "manifest_path": ms4 / "retention_manifest.json"}]
    assert sum("retention outcome=" in message for message in logs) == 1

    monkeypatch.setattr(module, "run_retention_cycle", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("private payload")))
    assert module.supervise_once() == 0
    assert len(child_calls) == 2
    assert any("cycle_failed" in message for message in logs)
    assert all("private payload" not in message for message in logs)

    monkeypatch.setattr(
        module,
        "run_retention_cycle",
        lambda **_kwargs: {
            "status": "pressure",
            "actions": [],
            "disk_pressure": True,
        },
    )
    assert module.supervise_once() == 1
    assert len(child_calls) == 2
    assert "disk pressure blocks launch" in logs[-1]

    sleeps: list[int] = []

    def stop_after_sleep(seconds: int) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "acquire_watch_mutex", lambda: True)
    monkeypatch.setattr(module, "supervise_once", lambda: 0)
    monkeypatch.setattr(module, "retention_schedule_seconds", lambda _path: 120)
    monkeypatch.setattr(module.time, "sleep", stop_after_sleep)
    monkeypatch.setattr(
        sys,
        "argv",
        ["supervise_ms4.py", "--watch", "--interval-seconds", "600"],
    )

    assert module.main() == 0
    assert sleeps == [120]
