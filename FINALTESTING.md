# Multi-Modal Biometric Authentication System

A biometric authentication system that combines **Face Recognition** and **Iris Recognition** for secure user authentication.

## Features

* Face Enrollment
* Face Verification
* Iris Template Generation (CLAHE-normalized, multi-sample averaged)
* Iris Verification
* Blink-based Liveness (anti-spoof)
* Multi-Modal Authentication (weighted fusion + AND gate)
* CustomTkinter GUI (live preview, in-process)
* User Management
* SQLite Database Storage

---

# Project Structure

```text
multi-modal-biometric/
│
├── demo/
│   └── face_demo.py
│
├── src/
│   ├── face/
│   ├── iris/
│   ├── database/
│   ├── workflows/
│   └── capture/
│
├── test_fusionauth.py
├── requirements.txt
└── README.md
```

---

# Environment Setup

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

### Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies (includes `customtkinter` for the GUI):

```bash
pip install -r requirements.txt
```

---

# Run the GUI (recommended)

```bash
python app.py
```

A CustomTkinter window opens with a live camera preview and buttons for
**Enroll**, **Verify (Face)**, **Authenticate**, **List Users**, **Delete
User**, and **Cancel / Clear**. The GUI runs the workflows in-process and drives
capture inside the embedded preview (no separate OpenCV windows). During
**Authenticate**, blink deliberately when prompted for the liveness step.

The CLI workflow below is an alternative to the GUI.

---

# User Enrollment

Enroll a new user:

```bash
python demo/face_demo.py enroll user1 "Masum" -s 3
```

Examples:

```bash
python demo/face_demo.py enroll user2 "Demo User" -s 3

python demo/face_demo.py enroll user3 "Rahul" -s 3

python demo/face_demo.py enroll user5 "Masuma" -s 3
```

During enrollment:

* Face samples are captured.
* Face templates are generated.
* Iris templates are generated automatically.
* Data is stored in the database.

---

# Face Verification

Verify an enrolled user:

```bash
python demo/face_demo.py verify user1
```

Example:

```bash
python demo/face_demo.py verify user5
```

Expected Output:

```text
Face Verification
✓ VERIFIED
```

---

# Full Multi-Modal Authentication

Run complete Face + Iris authentication (pass the user ID):

```bash
python test_fusionauth.py user1
```

The flow runs face verification, then iris verification with a **blink liveness**
check, then weighted score fusion. Blink when the iris step starts.

Expected Output (final result object):

```text
FINAL RESULT
MultiModalVerificationResult(success=True, user_id='user1', face_success=True,
  iris_success=True, face_confidence=..., iris_confidence=...,
  combined_confidence=..., message='Authentication Successful')
```

`success=True` only when both factors pass **and** the combined score clears
`fusion.combined_threshold`. A face-pass / iris-fail returns `success=False`.

---

# List All Enrolled Users

```bash
python -c "from src.database.storage import DatabaseManager; db=DatabaseManager(); db.initialize(); [print(f'{u.user_id} -> {u.name}') for u in db.list_users(active_only=False)]"
```

Example Output:

```text
user1 -> Masum
user2 -> Masum
user3 -> Imtiyaj
user4 -> Arun
user5 -> Masuma
sahil -> Sahil
```

---

# Delete User

Delete a user and all templates (face + iris). Easiest from the GUI's **Delete
User** button, or on the CLI:

```bash
python -c "from src.database.storage import DatabaseManager; db=DatabaseManager(); db.initialize(); db.delete_face_template('user5'); db.delete_iris_template('user5'); db.delete_user('user5'); print('Deleted user5')"
```

---

# Re-Enroll Existing User

```bash
python demo/face_demo.py enroll user1 "Masum" -s 3
```

If the user already exists, delete the user first and then enroll again.

> **Note:** iris templates enrolled before the iris-pipeline overhaul (symmetric
> crop + CLAHE + multi-sample averaging) are incompatible with the current
> matcher. If a previously-enrolled user fails iris verification, re-enroll them.
> Face encodings are unaffected.

---

# Common Troubleshooting

## Camera Not Opening

Check camera permissions and verify camera availability:

```bash
ls /dev/video*
```

---

## Verify Webcam Using OpenCV

```bash
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

Expected Output:

```text
True
```

---

## Check Existing Users

```bash
python -c "from src.database.storage import DatabaseManager; db=DatabaseManager(); db.initialize(); print(db.list_users(active_only=False))"
```

---

# Demo Workflow

### Step 1 - Enroll User

```bash
python demo/face_demo.py enroll user5 "Masuma" -s 3
```

### Step 2 - Verify Face

```bash
python demo/face_demo.py verify user5
```

### Step 3 - Run Multi-Modal Authentication

```bash
python test_fusionauth.py
```

---

# Technology Stack

* Python
* OpenCV
* face_recognition
* NumPy
* SQLite
* Iris Recognition Module (appearance-based, RGB)
* Blink-based Liveness (eye-aspect-ratio)
* Multi-Modal Biometric Fusion (weighted score + AND gate)
* CustomTkinter GUI

---

# Author

Final Year Project

Multi-Modal Biometric Authentication Using Face and Iris Recognition
