# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The virtual environment is at `.venv/` (not `venv/`).

## Running the System

```bash
# GUI application (tkinter)
python app.py

# CLI enrollment
python demo/face_demo.py enroll <user_id> "<name>" -s 3

# CLI verification (face only)
python demo/face_demo.py verify <user_id>

# Multi-modal authentication (face + iris)
python test_fusionauth.py <user_id>

# List enrolled users
python demo/face_demo.py list

# Delete a user
python demo/face_demo.py delete <user_id>
```

## Running Tests

There is no test suite runner — tests are standalone scripts run directly:

```bash
python test_enrollment.py
python test_iris.py
python test_iris_db.py
python test_verify_improvements.py
python check_db.py      # Inspect database contents
python check_system.py  # Check system/dependency status
```

## Architecture

The system is a dual-factor biometric authentication pipeline: face recognition followed by iris recognition. Both factors must pass for `MultiModalVerificationWorkflow` to succeed.

### Core data flow

1. **Camera** (`src/capture/camera.py`) — OpenCV wrapper, yields `CaptureResult`
2. **Face recognition** (`src/face/recognition.py`) — dlib-backed; detects faces and produces 128-D embeddings (`DetectedFace`)
3. **Iris recognition** (`src/iris/recognition.py`) — uses `face_recognition.face_landmarks` to locate eyes, crops them, and produces a flattened 64×64 grayscale feature vector (`IrisTemplate`)
4. **Database** (`src/database/storage.py`) — SQLite via raw `sqlite3` (not SQLAlchemy ORM despite the dependency); stores `UserRecord`, `FaceTemplateRecord`, and iris templates; thread-safe with a lock
5. **Workflows** (`src/workflows/`) — orchestrate the above:
   - `FaceEnrollmentWorkflow` — captures N samples, checks quality and pose diversity, saves face + iris templates in one pass
   - `FaceVerificationWorkflow` — 1:1 verify or 1:N identify
   - `IrisVerificationWorkflow` — iris-only verify
   - `MultiModalVerificationWorkflow` — runs face then iris sequentially; face weight 0.6, iris weight 0.4; combined threshold 70.0

### Key design notes

- Face detection model defaults to `"cnn"` (GPU); switch to `"hog"` in `config/settings.yaml` for CPU-only.
- Iris recognition uses RGB webcam landmarks (not NIR), so accuracy is 70–95% vs 99.9% for NIR hardware.
- The `app.py` GUI delegates enrollment/verification to subprocesses (`demo/face_demo.py`, `test_fusionauth.py`) via `subprocess.run`, so it does not share in-process state with the CLI.
- `DatabaseManager` must be initialized before use: `db = DatabaseManager(); db.initialize()`.
- Face templates store a list of raw encodings per user; matching uses the average encoding.

## Configuration

All tunable parameters live in `config/settings.yaml`. Key values:

| Key | Default | Notes |
|-----|---------|-------|
| `face_recognition.detection_model` | `cnn` | `hog` for CPU |
| `face_recognition.match_tolerance` | `0.6` | Lower = stricter |
| `enrollment.num_samples` | `5` | Clamped 3–10 |
| `database.path` | `data/biometric.db` | Relative to project root |
