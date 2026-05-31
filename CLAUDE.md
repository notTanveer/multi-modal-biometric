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
# GUI application (CustomTkinter; embedded live preview, frame-driven, in-process)
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

The system is a dual-factor biometric authentication pipeline: face recognition then iris recognition. Both factors must independently pass **and** a weighted fusion score must clear a threshold for `MultiModalVerificationWorkflow` to succeed.

### Core data flow

1. **Camera** (`src/capture/camera.py`) — OpenCV wrapper, yields `CaptureResult`
2. **Face recognition** (`src/face/recognition.py`) — dlib-backed; detects faces and produces 128-D embeddings (`DetectedFace`)
3. **Iris recognition** (`src/iris/recognition.py`) — uses `face_recognition.face_landmarks` to locate eyes, crops them **symmetrically (clamped)**, normalizes with **CLAHE**, and produces a flattened 64×64 feature vector. Also provides quality gating (`is_acceptable`), template averaging, a configurable distance metric (`rmse`/`correlation`), and **eye-aspect-ratio (EAR)** helpers for blink liveness.
4. **Database** (`src/database/storage.py`) — SQLite via raw `sqlite3` (not SQLAlchemy ORM despite the dependency); stores `UserRecord`, `FaceTemplateRecord`, iris templates, and verification logs; thread-safe with a lock. Has `delete_iris_template(user_id, eye=None)`.
5. **Workflows** (`src/workflows/`) — orchestrate the above:
   - `FaceEnrollmentWorkflow` — captures N face samples (quality + pose diversity), then averaged iris templates per eye. `store_enrollment()` persists a session with *pre-collected* iris templates (used when the GUI drives capture frame-by-frame).
   - `FaceVerificationWorkflow` — 1:1 verify or 1:N identify
   - `IrisVerificationWorkflow` — iris verify with optional blink-liveness capture loop
   - `MultiModalVerificationWorkflow` — runs face then iris; **fusion is config-driven**: normalizes each factor to [0,1], computes `face_weight*face + iris_weight*iris`, and requires `combined >= combined_threshold AND face_ok AND iris_ok`
   - `BiometricService` (`service.py`) — **in-process facade** the GUI uses. Returns structured result dataclasses (decide on `result.success`). Also exposes stateless per-frame helpers (`match_face`, `match_iris`, `fuse`, `collect_iris_sample`) so the GUI can drive capture without owning a camera loop.

### Key design notes

- Face detection model defaults to `"hog"` (CPU) in `config/settings.yaml`; switch to `"cnn"` for GPU.
- Iris recognition uses RGB webcam landmarks (not NIR), so accuracy is ~70–95% vs ~99.9% for NIR hardware. It is appearance-based eye-region matching, not an open-iris/NIR iris-code system.
- The `app.py` GUI talks to the workflows **in-process** via `BiometricService` (no subprocess, no stdout parsing). All capture is driven **frame-by-frame on the Tk main thread** through the embedded preview — **do not call `cv2.imshow`/`waitKey` from a worker thread** while Tk owns the event loop (that caused a hang). Overlays are drawn into the frame buffer with `cv2.putText`/`rectangle` (array ops, not HighGUI).
- `DatabaseManager` must be initialized before use: `db = DatabaseManager(); db.initialize()`.
- Face templates store a list of raw encodings per user; matching uses the min/avg distance to the set.
- **Iris templates enrolled before the iris-pipeline overhaul (symmetric crop + CLAHE + averaging) are incompatible with the current matcher — re-enroll affected users.** Face encodings are unaffected.

## Configuration

All tunable parameters live in `config/settings.yaml` (loaded via Pydantic models in `src/utils/config.py` — add a model field there when adding a new YAML key). Key values:

| Key | Default | Notes |
|-----|---------|-------|
| `face_recognition.detection_model` | `hog` | `cnn` for GPU |
| `face_recognition.match_tolerance` | `0.6` | Lower = stricter |
| `enrollment.num_samples` | `5` | Clamped 3–10 |
| `iris.threshold` | `0.65` | Iris distance threshold (lower = stricter) |
| `iris.num_samples` | `3` | Eye samples averaged per eye |
| `iris.metric` | `rmse` | `rmse` or `correlation` |
| `fusion.face_weight` / `fusion.iris_weight` | `0.6` / `0.4` | Should sum to 1.0 |
| `fusion.combined_threshold` | `0.6` | On the normalized [0,1] weighted score |
| `verification.require_liveness` | `true` | Blink anti-spoof |
| `liveness.ear_threshold` / `min_blinks` | `0.21` / `1` | Blink detection |
| `database.path` | `data/biometric.db` | Relative to project root |
