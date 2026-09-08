"""Durable content addressed files and consistent SQLite backups."""

from contextlib import closing
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile


class ArtifactStore:
    def put(
        self,
        value,
        kind,
        *,
        inline=True,
        media_type=None,
        redaction="none",
        capture="application",
        metadata=None,
    ):
        from .database import digest, json_bytes

        data = value if isinstance(value, bytes) else json_bytes(value)
        media_type = media_type or (
            "application/octet-stream"
            if isinstance(value, bytes)
            else "application/json"
        )
        is_inline = inline and len(data) <= 16384
        compressed = gzip.compress(data, mtime=0) if not is_inline else data
        storage_hash = digest(compressed)
        relative = None
        if not is_inline:
            relative = f"{storage_hash[:2]}/{storage_hash}.gz"
            target = self.artifact_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target.read_bytes()) != storage_hash:
                    raise ValueError("existing artifact hash mismatch")
            else:
                fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(compressed)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temporary, target)
                    directory_fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        return self.insert(
            "artifacts",
            content_sha256=digest(data),
            kind=kind,
            media_type=media_type,
            encoding="binary",
            compression="none" if is_inline else "gzip",
            logical_bytes=len(data),
            stored_bytes=len(compressed),
            storage_sha256=storage_hash,
            inline_bytes=compressed if is_inline else None,
            relative_path=relative,
            redaction_policy_version=redaction,
            capture_level=capture,
            metadata_json=metadata or {},
        )

    def read(self, artifact_id):
        from .database import digest

        record = self.one("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,))
        if record["relative_path"] is not None:
            path = (self.artifact_root / record["relative_path"]).resolve()
            if not path.is_relative_to(self.artifact_root.resolve()):
                raise ValueError("unsafe artifact path")
            data = path.read_bytes()
        else:
            data = record["inline_bytes"]
        if (
            len(data) != record["stored_bytes"]
            or digest(data) != record["storage_sha256"]
        ):
            raise ValueError("artifact storage hash mismatch")
        data = gzip.decompress(data) if record["compression"] == "gzip" else data
        if (
            len(data) != record["logical_bytes"]
            or digest(data) != record["content_sha256"]
        ):
            raise ValueError("artifact content hash mismatch")
        return data

    def json(self, artifact_id):
        return json.loads(self.read(artifact_id))

    def verify(self):
        from .database import digest, SCHEMA

        errors = []
        integrity = self.db.execute("PRAGMA integrity_check").fetchall()
        if [r[0] for r in integrity] != ["ok"]:
            errors.append({"database_integrity": [tuple(r) for r in integrity]})
        foreign = self.rows("PRAGMA foreign_key_check")
        if foreign:
            errors.append({"foreign_keys": foreign})
        if self.rows("SELECT version,migration_sha256 FROM schema_migrations") != [
            {"version": 1, "migration_sha256": digest(SCHEMA.read_bytes())}
        ]:
            errors.append({"migration": "hash mismatch"})
        from .invariants import check_invariants

        errors.extend(check_invariants(self))
        manifest_path = self.root / "backup_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_bytes())
                for relative, expected in manifest["files"].items():
                    path = (self.root / relative).resolve()
                    if (
                        not path.is_relative_to(self.root)
                        or digest(path.read_bytes()) != expected
                    ):
                        errors.append(
                            {"backup_file": relative, "error": "manifest hash mismatch"}
                        )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"backup_manifest": str(exc)})
        paths = set()
        artifacts = self.rows("SELECT artifact_id,relative_path FROM artifacts")
        for row in artifacts:
            try:
                self.read(row["artifact_id"])
            except (OSError, ValueError, EOFError) as exc:
                errors.append({"artifact_id": row["artifact_id"], "error": str(exc)})
            if row["relative_path"]:
                paths.add(row["relative_path"])
        orphans = [
            str(p.relative_to(self.artifact_root))
            for p in self.artifact_root.rglob("*")
            if p.is_file() and str(p.relative_to(self.artifact_root)) not in paths
        ]
        return {
            "ok": not errors,
            "checked_artifacts": len(artifacts),
            "errors": errors,
            "orphan_files": orphans,
        }

    def backup(self, output):
        from .database import digest, json_bytes, now, Store

        if self.read_only:
            raise ValueError(
                "backup archives are read-only; restore a working copy first"
            )
        output = Path(output).resolve()
        if output.exists():
            raise ValueError("backup output already exists")
        output.mkdir(parents=True)
        with closing(sqlite3.connect(output / "experiments.sqlite3")) as destination:
            self.db.backup(destination)
            refs = destination.execute(
                "SELECT DISTINCT relative_path FROM artifacts WHERE relative_path IS NOT NULL"
            ).fetchall()
        for (relative,) in refs:
            source = (self.artifact_root / relative).resolve()
            if not source.is_relative_to(self.artifact_root):
                raise ValueError("unsafe artifact path")
            target = output / "artifacts" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with Store(output) as restored:
            result = restored.verify()
            if not result["ok"]:
                raise ValueError("backup verification failed: " + str(result))
            restored.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        # The Backup API snapshot is checkpointed and closed. Only its stable database
        # and referenced immutable files belong in the portable archive, never WAL/SHM.
        stable_files = [output / "experiments.sqlite3"] + [
            output / "artifacts" / relative for (relative,) in refs
        ]
        files = {
            str(p.relative_to(output)): digest(p.read_bytes()) for p in stable_files
        }
        manifest = {"created_at": now(), "files": files, "verification": result}
        (output / "backup_manifest.json").write_bytes(json_bytes(manifest))
        with Store(output) as archived:
            if not archived.verify()["ok"]:
                raise ValueError("final backup manifest verification failed")
        with self.transaction():
            aid = self.put(manifest, "backup_manifest")
            self.event("artifact_id", aid, "backup_completed")
        return output
