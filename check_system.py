#!/usr/bin/env python3
"""
System Readiness Check - Multi-Modal Biometric System
Verifies all components before user testing
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_imports():
    """Check all required imports."""
    print("=" * 50)
    print("1. Checking Python Imports...")
    print("=" * 50)
    
    required = [
        ("cv2", "OpenCV"),
        ("face_recognition", "face_recognition"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("pydantic", "Pydantic"),
        ("customtkinter", "CustomTkinter"),
    ]
    
    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            all_ok = False
    
    return all_ok

def check_project_modules():
    """Check project modules can be imported."""
    print("\n" + "=" * 50)
    print("2. Checking Project Modules...")
    print("=" * 50)
    
    modules = [
        ("src.capture.camera", "Camera Module"),
        ("src.face.recognition", "Face Recognition"),
        ("src.database.storage", "Database Storage"),
        ("src.workflows.enrollment", "Enrollment Workflow"),
        ("src.workflows.verification", "Verification Workflow"),
        ("src.utils.config", "Configuration"),
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name} loaded")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            all_ok = False
    
    return all_ok

def check_config():
    """Check configuration file."""
    print("\n" + "=" * 50)
    print("3. Checking Configuration...")
    print("=" * 50)
    
    try:
        from src.utils.config import get_config
        config = get_config()
        
        print(f"✓ Configuration loaded")
        print(f"  - Detection model: {config.face_recognition.detection_model}")
        print(f"  - Match tolerance: {config.face_recognition.match_tolerance}")
        print(f"  - Enrollment samples: {config.enrollment.num_samples}")
        print(f"  - Verification attempts: {config.verification.max_attempts}")
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def check_database():
    """Check database initialization."""
    print("\n" + "=" * 50)
    print("4. Checking Database...")
    print("=" * 50)
    
    try:
        from src.database.storage import DatabaseManager
        
        db = DatabaseManager()
        users = db.list_users()
        
        print(f"✓ Database initialized")
        print(f"  - Users enrolled: {len(users)}")
        print(f"  - Database path: {db.db_path}")
        return True
    except Exception as e:
        print(f"✗ Database failed: {e}")
        return False

def check_face_system():
    """Check face recognition system."""
    print("\n" + "=" * 50)
    print("5. Checking Face Recognition System...")
    print("=" * 50)
    
    try:
        from src.face.recognition import FaceRecognitionSystem
        import numpy as np
        
        # Suppress initialization output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            system = FaceRecognitionSystem()
        
        print(f"✓ Face recognition system initialized")
        print(f"  - Detector: CNN model")
        print(f"  - Encoder: dlib ResNet (128-D)")
        print(f"  - Matcher: tolerance 0.6")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = system.detect_and_encode(dummy_image)
        print(f"✓ Detection pipeline functional (no faces in test image: {len(faces)})")
        
        return True
    except Exception as e:
        print(f"✗ Face system failed: {e}")
        return False

def check_cli():
    """Check CLI is functional."""
    print("\n" + "=" * 50)
    print("6. Checking CLI Commands...")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "demo/face_demo.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            commands = ["enroll", "verify", "identify", "list", "delete", "stats", "continuous"]
            output = result.stdout
            
            missing = [cmd for cmd in commands if cmd not in output]
            
            if not missing:
                print(f"✓ All {len(commands)} CLI commands available:")
                for cmd in commands:
                    print(f"  - {cmd}")
                return True
            else:
                print(f"✗ Missing commands: {missing}")
                return False
        else:
            print(f"✗ CLI failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI check failed: {e}")
        return False

def print_final_status(results):
    """Print final status and next steps."""
    print("\n" + "=" * 50)
    print("SYSTEM READINESS SUMMARY")
    print("=" * 50)
    
    checks = [
        "Python Imports",
        "Project Modules",
        "Configuration",
        "Database",
        "Face Recognition",
        "CLI Commands"
    ]
    
    for check, passed in zip(checks, results):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {check}")
    
    print("=" * 50)
    
    if all(results):
        print("\n🎉 SYSTEM READY FOR TESTING!")
        print("\nNext Steps:")
        print("1. Read TESTING_GUIDE.md for test instructions")
        print("2. Run: python demo/face_demo.py enroll sahil \"Sahil\" -s 3")
        print("3. Test verification: python demo/face_demo.py verify sahil")
        print("4. Check enrolled users: python demo/face_demo.py list")
        print("\n📖 See USAGE.md for full command reference")
        return 0
    else:
        print("\n❌ SYSTEM NOT READY - Fix failed checks above")
        print("\nIf you need help, check:")
        print("- requirements.txt for dependencies")
        print("- config/settings.yaml for configuration")
        print("- README.md for setup instructions")
        return 1

def main():
    """Run all checks."""
    print("\n" + "=" * 50)
    print("MULTI-MODAL BIOMETRIC SYSTEM")
    print("Readiness Check")
    print("=" * 50 + "\n")
    
    results = [
        check_imports(),
        check_project_modules(),
        check_config(),
        check_database(),
        check_face_system(),
        check_cli(),
    ]
    
    return print_final_status(results)

if __name__ == "__main__":
    sys.exit(main())
