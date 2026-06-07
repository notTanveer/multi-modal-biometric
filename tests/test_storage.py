"""Unit tests for the SQLite storage layer using an isolated temp database.

Covers the encoding (de)serialization round-trips and — importantly — that
deleting a user now cascades to their biometric templates (the PRAGMA
foreign_keys fix from the hardening pass).
"""

import numpy as np
import pytest

from src.database.storage import DatabaseManager


@pytest.fixture
def db(tmp_path):
    manager = DatabaseManager(db_path=str(tmp_path / "test.db"))
    manager.initialize()
    yield manager
    manager.close()


def _make_encodings(n=3):
    rng = np.random.default_rng(42)
    return [rng.random(128).astype(np.float64) for _ in range(n)]


def test_enroll_and_face_template_roundtrip(db):
    encodings = _make_encodings(3)
    result = db.enroll_user("u1", "Alice", encodings)
    assert result.success
    assert result.num_samples == 3

    record = db.get_face_template("u1")
    assert record is not None
    assert len(record.encodings) == 3
    for original, restored in zip(encodings, record.encodings):
        assert np.allclose(original, restored)


def test_duplicate_enrollment_rejected(db):
    db.enroll_user("u1", "Alice", _make_encodings())
    again = db.enroll_user("u1", "Alice Again", _make_encodings())
    assert not again.success
    assert "already" in again.message.lower()


def test_iris_template_roundtrip(db):
    db.enroll_user("u1", "Alice", _make_encodings())
    template = np.linspace(0, 1, 4096).astype(np.float32)
    db.save_iris_template("u1", template, "left")

    restored = db.get_iris_template("u1", "left")
    assert restored is not None
    assert np.allclose(template, restored)
    # No template for the other eye yet.
    assert db.get_iris_template("u1", "right") is None


def test_delete_user_cascades_to_templates(db):
    """Validates PRAGMA foreign_keys = ON makes ON DELETE CASCADE effective."""
    db.enroll_user("u1", "Alice", _make_encodings())
    db.save_iris_template("u1", np.zeros(4096, dtype=np.float32), "left")
    db.save_iris_template("u1", np.zeros(4096, dtype=np.float32), "right")

    assert db.get_face_template("u1") is not None
    assert db.get_iris_template("u1", "left") is not None

    assert db.delete_user("u1") is True

    # Templates must be gone, not orphaned.
    assert db.get_face_template("u1") is None
    assert db.get_iris_template("u1", "left") is None
    assert db.get_iris_template("u1", "right") is None
    assert db.get_user("u1") is None


def test_list_users_active_filter(db):
    db.enroll_user("u1", "Alice", _make_encodings())
    db.enroll_user("u2", "Bob", _make_encodings())
    db.update_user("u2", is_active=False)

    active = {u.user_id for u in db.list_users(active_only=True)}
    everyone = {u.user_id for u in db.list_users(active_only=False)}
    assert active == {"u1"}
    assert everyone == {"u1", "u2"}


def test_verification_logging_and_stats(db):
    db.enroll_user("u1", "Alice", _make_encodings())
    db.log_verification("u1", "combined", True, face_score=0.3, combined_score=0.8)
    db.log_verification("u1", "combined", False, face_score=0.7, combined_score=0.4)

    stats = db.get_verification_stats("u1")
    assert stats["total_attempts"] == 2
    assert stats["successful"] == 1
    assert stats["failed"] == 1
    assert stats["success_rate"] == pytest.approx(50.0)
