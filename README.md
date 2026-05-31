# Multi-Modal Biometric Authentication System

A dual-factor biometric authentication system combining **face recognition** and
**iris (eye-region) recognition** with blink-based liveness, for secure access
control. Both factors must pass and a weighted fusion score must clear a
threshold before access is granted.

## 🎯 Project Status

**Implemented and working:**

- ✅ Face detection + 128-D encoding (dlib via `face_recognition`)
- ✅ Face matching, 1:1 verification, 1:N identification
- ✅ Iris (eye-region) recognition on an RGB webcam — CLAHE-normalized,
  multi-sample averaged templates
- ✅ **Blink-based liveness** (eye-aspect-ratio) anti-spoof
- ✅ **Score-level fusion + AND gate** — weighted combine of both factors,
  config-driven, both factors required
- ✅ SQLite storage (users, face templates, iris templates, verification logs)
- ✅ Modern desktop GUI (CustomTkinter) with embedded live preview
- ✅ CLI tooling for enrollment / verification / multi-modal auth

> **Note on iris:** this uses appearance-based eye-region matching from an RGB
> webcam (not a NIR iris-code system like open-iris). Accuracy is inherently
> limited (~70–95%, lighting/eye-color dependent) and it is best treated as a
> second factor on top of face, not a standalone high-security iris system.

## 🚀 Quick Start

### Installation

```bash
cd multi-modal-biometric

# Create the virtual environment (the project expects .venv)
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac

# Install dependencies (includes customtkinter for the GUI)
pip install -r requirements.txt
```

### Run the GUI (recommended)

```bash
python app.py
```

The window shows a **live camera preview** and buttons for **Enroll**,
**Verify (Face)**, **Authenticate** (multi-modal), **List Users**, **Delete
User**, and **Cancel / Clear**. All capture happens inside the embedded preview
— there are no separate OpenCV pop-up windows.

- **Enroll** — collects several diverse face samples, then a few iris samples.
- **Verify (Face)** — face-only 1:1 check against the entered User ID.
- **Authenticate** — full multi-modal flow: face → iris (blink to confirm
  liveness) → weighted fusion. The status badge shows **AUTHENTICATED** or
  **DENIED**, with a per-factor breakdown.

During Verify/Authenticate the preview runs slower (~5 fps) because detection
runs per frame on the UI thread — this is expected, not a hang. **Blink
deliberately** (a slow, full blink) during the liveness step.

### Run from the CLI

```bash
# Enroll (face + iris)
python demo/face_demo.py enroll sahil "Sahil" -s 3

# Face-only verification
python demo/face_demo.py verify sahil

# Multi-modal authentication (face + iris + liveness)
python test_fusionauth.py sahil

# List / delete
python demo/face_demo.py list
python demo/face_demo.py delete sahil
```

**📖 See [USAGE.md](USAGE.md) for the detailed usage guide.**

## 🏗️ Architecture

```
src/
├── capture/        # Camera interface (OpenCV)
├── face/           # Face recognition (face_recognition / dlib)
├── iris/           # Iris (eye-region) recognition + EAR liveness helpers
├── database/       # SQLite storage (raw sqlite3, thread-safe)
├── workflows/      # Enrollment, verification, iris, fusion, and the service facade
│   ├── enrollment.py        # Face + iris enrollment
│   ├── verification.py      # Face 1:1 / 1:N
│   ├── iris_verification.py # Iris verify + blink liveness
│   ├── multimodelauth.py    # Weighted fusion + AND gate
│   └── service.py           # In-process BiometricService (used by the GUI)
└── utils/          # Configuration (YAML + Pydantic)

app.py              # CustomTkinter GUI (frame-driven, in-process)
demo/face_demo.py   # CLI interface
test_fusionauth.py  # CLI multi-modal authentication
config/settings.yaml
docs/IMPLEMENTATION_DOCUMENT.md
```

### How fusion works

```
face_norm = face_confidence / 100        # both factors normalized to [0,1]
iris_norm = iris_confidence / 100
combined  = face_weight*face_norm + iris_weight*iris_norm   # default 0.6 / 0.4
success   = (combined >= combined_threshold) AND face_ok AND iris_ok
```

All weights and the threshold live in `config/settings.yaml` (`fusion:`). A
strong face alone cannot grant access — both factors must independently pass.

### In-process, not subprocess

The GUI calls the workflows directly through `BiometricService` and decides the
outcome from structured result objects (`result.success`). It does **not** shell
out to CLI scripts or parse stdout. (An earlier subprocess design could report a
face-pass / iris-fail as "authenticated" by string-matching output — that path
is gone.)

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Face detection | `face_recognition` (dlib HOG/CNN) |
| Face encoding | dlib ResNet (128-D) |
| Iris recognition | Appearance-based eye-region matching (OpenCV, RGB) |
| Liveness | Eye-aspect-ratio (EAR) blink detection |
| Camera | OpenCV |
| Database | SQLite (raw `sqlite3`) |
| Config | YAML + Pydantic |
| GUI | CustomTkinter + Pillow |

## 🛠️ Configuration

Key settings in `config/settings.yaml`:

```yaml
face_recognition:
  detection_model: "hog"     # "hog" (CPU) or "cnn" (GPU)
  match_tolerance: 0.6       # lower = stricter

iris:
  threshold: 0.65            # distance threshold (lower = stricter)
  num_samples: 3             # eye samples averaged per eye at enrollment
  metric: "rmse"             # "rmse" or "correlation"

fusion:
  face_weight: 0.6           # must sum to 1.0 with iris_weight
  iris_weight: 0.4
  combined_threshold: 0.6    # on the normalized [0,1] weighted score

verification:
  require_liveness: true     # blink anti-spoof

liveness:
  ear_threshold: 0.21        # eye-aspect-ratio below this = eye closed
  min_blinks: 1
  timeout_seconds: 6.0
```

## 🔐 Security Notes

- Face templates stored as 128-D embeddings (not raw images); iris as
  normalized feature vectors.
- Local storage only (`data/biometric.db`); no cloud dependency.
- Both factors required (AND gate) plus a fused-score threshold.
- Blink liveness rejects a static photo held to the camera.
- Verification attempts (including the combined result) are logged for audit.

## 🚧 Limitations (RGB Webcam)

- **No NIR camera:** iris is appearance-based, not true NIR iris-code.
- **Accuracy:** ~70–95% for the iris factor vs ~99.9% with NIR hardware.
- **Lighting/eye-color dependent.**
- **Liveness is basic** (blink only) — not a full presentation-attack defense.

## ⚠️ Upgrade note

Iris templates enrolled **before the iris pipeline overhaul** (symmetric crop +
CLAHE + multi-sample averaging) are not compatible with the current matcher.
If a previously-enrolled user fails iris verification, **re-enroll** them. Face
encodings are unaffected.

## 📝 Documentation

- **[USAGE.md](USAGE.md)** — complete usage guide (GUI + CLI)
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** — end-to-end test sequence
- **[docs/IMPLEMENTATION_DOCUMENT.md](docs/IMPLEMENTATION_DOCUMENT.md)** — design document

## 🙏 Acknowledgments

Built using:
- [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey
- [dlib](http://dlib.net/) by Davis King
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) by Tom Schimansky
