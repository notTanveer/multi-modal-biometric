# Multi-Modal Biometric Authentication System

A biometric authentication system that combines **Face Recognition** and **Iris Recognition** for secure user authentication.

## Features

* Face Enrollment
* Face Verification
* Iris Template Generation
* Iris Verification
* Multi-Modal Authentication
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

Install dependencies:

```bash
pip install -r requirements.txt
```

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

Run complete Face + Iris authentication:

```bash
python test_fusionauth.py
```

Expected Output:

```text
STEP 1: FACE VERIFICATION
✓ PASSED

STEP 2: IRIS VERIFICATION
✓ PASSED

Authentication Successful
```

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

Delete a user and face template:

```bash
python -c "from src.database.storage import DatabaseManager; db=DatabaseManager(); db.initialize(); db.delete_face_template('user5'); db.delete_user('user5'); print('Deleted user5')"
```

Example:

```bash
python -c "from src.database.storage import DatabaseManager; db=DatabaseManager(); db.initialize(); db.delete_face_template('user1'); db.delete_user('user1'); print('Deleted user1')"
```

---

# Re-Enroll Existing User

```bash
python demo/face_demo.py enroll user1 "Masum" -s 3
```

If the user already exists, delete the user first and then enroll again.

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
* Iris Recognition Module
* Multi-Modal Biometric Fusion

---

# Author

Final Year Project

Multi-Modal Biometric Authentication Using Face and Iris Recognition
