# Usage Guide

This system authenticates users with **two factors**: face recognition and
iris (eye-region) recognition, with blink-based liveness. Both factors must
pass, and a weighted fusion score must clear a threshold, before authentication
succeeds.

There are two ways to use it: the **desktop GUI** (recommended) and the **CLI**.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The virtual environment lives at `.venv/`.

---

## GUI (recommended)

```bash
python app.py
```

A CustomTkinter window opens with a **live camera preview** on the left and
controls on the right. All capture happens inside the embedded preview — there
are no separate OpenCV pop-up windows.

### Buttons

| Button | What it does |
|--------|--------------|
| **Enroll User** | Captures diverse face samples, then a few iris samples, for the entered User ID + Name |
| **Verify (Face)** | Face-only 1:1 check against the entered User ID |
| **Authenticate** | Full multi-modal flow: face → iris (blink) → weighted fusion |
| **List Users** | Lists enrolled users |
| **Delete User** | Deletes the entered User ID and all its templates |
| **Cancel / Clear** | Cancels an in-progress operation, or clears the output log |

### Enrolling

1. Type a **User ID** and **Name**.
2. Click **Enroll User**.
3. **Step 1/2 — Face:** hold still and move slightly between captures so the
   samples are diverse. Progress shows `N/target captured`.
4. **Step 2/2 — Iris:** keep looking at the camera while it collects a few eye
   samples per eye.
5. The badge turns **ENROLLED** on success.

Tips: face the camera directly, ensure good lighting, no sunglasses/masks.

### Authenticating

1. Type the **User ID**.
2. Click **Authenticate**.
3. **Step 1/2 — Face:** look at the camera until the face factor passes.
4. **Step 2/2 — Iris:** **blink deliberately** (a slow, full blink) to confirm
   liveness; a static photo will not blink and is rejected.
5. The badge shows **✓ AUTHENTICATED** or **✗ DENIED**, with a per-factor
   breakdown (face %, iris %, combined %).

> The preview runs slower (~5 fps) during Verify/Authenticate because detection
> runs per frame on the UI thread. This is expected — not a hang.

---

## CLI

### Face enrollment + iris (one pass)

```bash
python demo/face_demo.py enroll sahil "Sahil" -s 3
```

Opens a camera preview window, auto-captures face samples (move slightly between
captures), then captures iris templates. Press `c` to capture manually, `q` to
cancel.

### Face-only verification (1:1)

```bash
python demo/face_demo.py verify sahil
```

### Multi-modal authentication (face + iris + liveness)

```bash
python test_fusionauth.py sahil
```

Runs face verification, then iris verification with blink liveness, then
weighted fusion. Prints a `MultiModalVerificationResult` with `success`,
per-factor confidences, and the `combined_confidence`.

### Other face CLI commands

| Command | Description | Example |
|---------|-------------|---------|
| `identify [-t TIMEOUT]` | Identify who is at the camera (1:N) | `identify -t 30` |
| `list [-a]` | List users | `list --all` |
| `delete <user_id> [-f]` | Delete a user | `delete sahil --force` |
| `stats [-u USER] [-d DAYS]` | Verification statistics | `stats -u sahil -d 7` |
| `continuous [USER_ID]` | Live monitoring | `continuous sahil` |

Run `python demo/face_demo.py --help` for the full reference.

---

## Configuration

All tunables live in `config/settings.yaml`.

```yaml
face_recognition:
  detection_model: "hog"     # "hog" (CPU) or "cnn" (GPU)
  match_tolerance: 0.6       # lower = stricter (0.0–1.0)
  min_face_size: 50

enrollment:
  num_samples: 5             # face samples (clamped 3–10)
  sample_interval: 0.5       # seconds between auto-captures

iris:
  threshold: 0.65            # distance threshold (lower = stricter)
  num_samples: 3             # eye samples averaged per eye
  metric: "rmse"             # "rmse" or "correlation"

fusion:
  face_weight: 0.6           # must sum to 1.0 with iris_weight
  iris_weight: 0.4
  combined_threshold: 0.6    # on the normalized [0,1] weighted score

verification:
  max_attempts: 3
  timeout_seconds: 30
  require_liveness: true     # blink anti-spoof

liveness:
  ear_threshold: 0.21        # EAR below this = eye closed
  min_blinks: 1
  timeout_seconds: 6.0
```

---

## Troubleshooting

### Camera won't open
```bash
ls /dev/video*                 # check available cameras
```
Edit `config/settings.yaml` → `camera.device_id` (try `1` if `0` fails).

### Face / eyes not detected
- Improve lighting, move closer, remove glasses/masks.
- For the iris step, make sure both eyes are clearly visible.

### Liveness never confirms
- **Blink deliberately** (slow, full blink). Detection runs at ~5 fps during the
  operation, so a fast flick of the eyelids may be missed.
- Adjust `liveness.ear_threshold` / `min_blinks` if needed, or set
  `verification.require_liveness: false` to disable it.

### A previously-enrolled user fails iris
Iris templates from before the iris-pipeline overhaul are incompatible with the
current matcher. **Re-enroll** the user (delete + enroll). Face encodings are
unaffected.

### Low accuracy
- Re-enroll with more samples (`-s 10` on the CLI).
- Tune `face_recognition.match_tolerance` and `iris.threshold` (lower = stricter).
- Enroll under lighting similar to where you'll authenticate.

### Reset the database
```bash
rm data/biometric.db
```

---

## How it fits together

```
Camera ─▶ Face detect+encode (128-D) ─▶ match vs template ─┐
                                                           ├─▶ weighted fusion ─▶ AND gate ─▶ verdict
Camera ─▶ Eye landmarks ─▶ EAR blink (liveness) + iris ────┘
                            crop → CLAHE → compare
```

Both factors must independently pass **and** the fused score must clear
`fusion.combined_threshold`. See `docs/IMPLEMENTATION_DOCUMENT.md` for the design.
