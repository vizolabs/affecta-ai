"""SQLAlchemy engine/session setup (SQLite by default)."""

import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.api.core.config import settings


class Base(DeclarativeBase):
    pass


def _make_engine():
    kwargs = {"future": True}
    if not settings.db_url.startswith("sqlite"):
        return create_engine(settings.db_url, **kwargs)

    kwargs["connect_args"] = {"check_same_thread": False}
    kwargs["pool_pre_ping"] = True
    eng = create_engine(settings.db_url, **kwargs)

    from sqlalchemy import event

    @event.listens_for(eng, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - driver hook
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=8000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return eng


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    """FastAPI dependency yielding a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_columns() -> None:
    """Add columns introduced after first deploy (idempotent, SQLite-safe)."""
    insp = inspect(engine)
    if "settings" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("settings")}
    with engine.begin() as conn:
        if "narrator_detail" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN narrator_detail "
                    "VARCHAR(20) DEFAULT 'normal'"
                )
            )
        if "ema_alpha" not in cols:
            conn.execute(
                text("ALTER TABLE settings ADD COLUMN ema_alpha FLOAT DEFAULT 0.35")
            )
        if "analysis_fps" not in cols:
            conn.execute(
                text("ALTER TABLE settings ADD COLUMN analysis_fps INTEGER DEFAULT 8")
            )
        if "emotion_hold_seconds" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN emotion_hold_seconds "
                    "FLOAT DEFAULT 0.0"
                )
            )
        if "analysis_mode" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN analysis_mode "
                    "VARCHAR(20) DEFAULT 'live'"
                )
            )
        if "analysis_interval_seconds" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN analysis_interval_seconds "
                    "INTEGER DEFAULT 120"
                )
            )
        if "motion_trigger_enabled" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN motion_trigger_enabled "
                    "BOOLEAN DEFAULT 1"
                )
            )
        if "pulse_scan_seconds" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN pulse_scan_seconds "
                    "INTEGER DEFAULT 30"
                )
            )
        if "pulse_hold_seconds" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE settings ADD COLUMN pulse_hold_seconds "
                    "INTEGER DEFAULT 30"
                )
            )


# Registry seeds — metrics sourced from experiments/EXP-010-ENSEMBLE
# leaderboard (official RAF-DB test, n=3068) and EXP-007 test_metrics.json.
# key: (name, version, type, dataset, metrics, checkpoint, status)
_SEEDS = [
    (
        "dan", "1.0", "expression", "RAF-DB",
        {"test_acc": 89.70, "test_bacc": 82.75},
        "data/raw/dan_weights/RAF-DB.pth",
        "deployed",
    ),
    (
        "ddamfn", "2.0", "expression", "RAF-DB",
        {"test_acc": 83.83, "test_bacc": 73.40},
        "experiments/EXP-009-DDAMFN-PP/checkpoints/rafdb_epoch20_acc0.9204_bacc0.8617.pth",
        "validated",
    ),
    (
        "vit_small", "1.0", "expression", "RAF-DB",
        {"test_acc": 78.88, "test_bacc": 73.23},
        "experiments/EXP-008-VIT-SMALL/checkpoints/best.pt",
        "validated",
    ),
    (
        "efficientnet_b3", "1.0", "expression", "RAF-DB",
        {"test_acc": 66.49, "test_macro_f1": 59.13},
        "experiments/EXP-007-EFFICIENTNET-B3/checkpoints/best.pt",
        "trained",
    ),
    (
        "ensemble", "1.0", "expression", "RAF-DB",
        {"test_acc": 89.99, "test_bacc": 82.90},
        "experiments/EXP-010-ENSEMBLE/production_ensemble_results.json",
        "validated",
    ),
]


def init_db() -> None:
    """Create tables + upsert model registry seed (idempotent)."""
    from src.api import models  # noqa: F401  (register ORM mappings)
    from src.api.models import ModelRegistry

    Base.metadata.create_all(engine)
    _ensure_columns()
    # Restrict DB file permissions (emails + password hashes live here).
    if settings.db_url.startswith("sqlite"):
        db_path = settings.db_url.split("///", 1)[-1]
        try:
            os.chmod(db_path, 0o600)
        except OSError:
            pass

    with Session(engine) as db:
        for name, version, mtype, dataset, metrics, ckpt, status in _SEEDS:
            row = (
                db.query(ModelRegistry)
                .filter_by(model_name=name, version=version)
                .first()
            )
            if row is None:
                db.add(
                    ModelRegistry(
                        model_name=name,
                        version=version,
                        model_type=mtype,
                        dataset_used=dataset,
                        metrics=metrics,
                        checkpoint_path=ckpt,
                        status=status,
                    )
                )
            else:
                # Re-seed metrics/status so stale rows (e.g. the pre-training
                # B3 numbers) self-heal on restart.
                row.metrics = metrics
                row.status = status
                row.checkpoint_path = ckpt
        db.commit()
