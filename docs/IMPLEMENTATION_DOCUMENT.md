# Multi-Modal Biometric Authentication System
## Face + Iris Dual Authentication Implementation Document

**Version:** 1.0  
**Date:** January 10, 2026  
**Author:** Sahil  

---

## 1. Executive Summary

This document outlines the implementation plan for a **multi-modal biometric authentication system** that combines face recognition and iris recognition for dual-factor biometric authentication. The system will require successful verification of **both** biometric modalities before granting access, significantly enhancing security compared to single-modality systems.

### Key Technologies
- **Face Recognition:** [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition) - Built on dlib's state-of-the-art face recognition with 99.38% accuracy on LFW benchmark
- **Iris Recognition:** [worldcoin/open-iris](https://github.com/worldcoin/open-iris) - Production-grade iris recognition system designed for billions of users

---

## 1.1 Implementation Status (As-Built)

> This document captures the original design intent. The shipped system diverges
> from the plan in a few important ways — read this section alongside the rest.

- **Iris recognition is appearance-based, not open-iris/NIR.** The implemented
  iris factor (`src/iris/recognition.py`) locates eyes via
  `face_recognition.face_landmarks`, crops them symmetrically, normalizes with
  CLAHE, and matches flattened 64×64 feature vectors (RMSE or normalized
  correlation). It is not the open-iris NIR iris-code pipeline. Treat it as a
  second factor (~70–95% accuracy), not a standalone high-security iris system.
- **Fusion is weighted score-level + AND gate, config-driven.** Each factor is
  normalized to [0,1], combined as `face_weight*face + iris_weight*iris`
  (defaults 0.6 / 0.4), and access requires `combined >= combined_threshold AND
  face_ok AND iris_ok`. Weights/threshold live in `config/settings.yaml`
  (`fusion:`). See `src/workflows/multimodelauth.py`.
- **Blink liveness implemented.** Eye-aspect-ratio (EAR) blink detection
  (`liveness:` config) provides basic anti-spoof; a static photo is rejected.
- **Desktop GUI (CustomTkinter).** `app.py` is an in-process app with an
  embedded live preview that drives capture frame-by-frame on the main thread,
  via `BiometricService` (`src/workflows/service.py`). It decides outcomes from
  structured result objects, not by parsing CLI stdout.
- **Storage** is raw `sqlite3` (the SQLAlchemy dependency is unused).

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-MODAL BIOMETRIC SYSTEM                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐   │
│  │   Camera     │────▶│   Image      │────▶│   Pre-processing         │   │
│  │   Input      │     │   Capture    │     │   Module                 │   │
│  └──────────────┘     └──────────────┘     └──────────┬───────────────┘   │
│                                                        │                   │
│                              ┌─────────────────────────┴────────────────┐  │
│                              │                                          │  │
│                              ▼                                          ▼  │
│                 ┌────────────────────────┐            ┌─────────────────────┐
│                 │   FACE RECOGNITION     │            │   IRIS RECOGNITION  │
│                 │   MODULE               │            │   MODULE            │
│                 │                        │            │                     │
│                 │  • Face Detection      │            │  • IR Image Input   │
│                 │  • Face Encoding       │            │  • Segmentation     │
│                 │  • Face Comparison     │            │  • Normalization    │
│                 │  (128-D embedding)     │            │  • Feature Extract  │
│                 │                        │            │  • Iris Encoding    │
│                 └───────────┬────────────┘            └──────────┬──────────┘
│                             │                                    │          │
│                             ▼                                    ▼          │
│                 ┌────────────────────────┐            ┌─────────────────────┐
│                 │   Face Match Score     │            │  Hamming Distance   │
│                 │   (Euclidean Distance) │            │  Score              │
│                 └───────────┬────────────┘            └──────────┬──────────┘
│                             │                                    │          │
│                             └────────────────┬───────────────────┘          │
│                                              │                              │
│                                              ▼                              │
│                              ┌───────────────────────────────┐              │
│                              │      FUSION ENGINE            │              │
│                              │                               │              │
│                              │  • Score Normalization        │              │
│                              │  • Decision Fusion            │              │
│                              │  • Threshold Validation       │              │
│                              └───────────────┬───────────────┘              │
│                                              │                              │
│                                              ▼                              │
│                              ┌───────────────────────────────┐              │
│                              │    AUTHENTICATION RESULT      │              │
│                              │    (GRANT / DENY ACCESS)      │              │
│                              └───────────────────────────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| Camera Input | Capture RGB face images and IR iris images | OpenCV, USB/IP cameras |
| Pre-processing | Image quality assessment, ROI extraction | OpenCV, NumPy |
| Face Recognition | Face detection, encoding, comparison | `face_recognition` library |
| Iris Recognition | Iris segmentation, encoding, matching | `open-iris` library |
| Fusion Engine | Combine scores, make auth decision | Custom implementation |
| Database | Store enrolled user templates | SQLite/PostgreSQL + encrypted storage |

---

## 3. Technology Deep Dive

### 3.1 Face Recognition (`ageitgey/face_recognition`)

#### Key Features
- Built on dlib's deep learning face recognition model
- 99.38% accuracy on Labeled Faces in the Wild (LFW) benchmark
- Generates 128-dimensional face encodings
- Supports both HOG (CPU) and CNN (GPU) face detection models

#### Core API Functions

```python
import face_recognition

# 1. Load and encode a face image
image = face_recognition.load_image_file("person.jpg")
face_encodings = face_recognition.face_encodings(image)  # Returns list of 128-D vectors

# 2. Compare faces
results = face_recognition.compare_faces(
    known_face_encodings,    # List of known encodings
    unknown_face_encoding,   # Single encoding to check
    tolerance=0.6            # Distance threshold (lower = stricter)
)

# 3. Get distance for more granular comparison
distances = face_recognition.face_distance(
    known_face_encodings,
    unknown_face_encoding
)
```

#### Face Recognition Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | 0.6 | Match threshold (0.0-1.0). Lower = stricter matching |
| `model` | "hog" | Face detection model: "hog" (fast/CPU) or "cnn" (accurate/GPU) |
| `num_jitters` | 1 | Re-sampling count for encoding. Higher = more accurate but slower |

#### Dependencies
```
face_recognition_models>=0.3.0
Click>=6.0
dlib>=19.7
numpy
Pillow
```

---

### 3.2 Iris Recognition (`worldcoin/open-iris`)

#### Key Features
- Production-grade iris recognition pipeline
- Designed for large-scale verification (billions of users)
- Uses Gabor wavelets for feature extraction
- Hamming distance matching with rotation compensation

#### Pipeline Stages

```
IR Image ──▶ Segmentation ──▶ Normalization ──▶ Feature Extraction ──▶ Encoding
                 │                 │                    │                  │
                 ▼                 ▼                    ▼                  ▼
           Pupil/Iris         Rubber Sheet        Gabor Filter      Binary Iris
           Boundaries          Unwrapping           Response           Code
```

#### Core API Usage

```python
import cv2
import iris
from iris.io.dataclasses import IRImage

# 1. Create pipeline
iris_pipeline = iris.IRISPipeline()

# 2. Load IR image
img_pixels = cv2.imread("iris_image.png", cv2.IMREAD_GRAYSCALE)

# 3. Process image and get iris template
result = iris_pipeline(
    IRImage(
        img_data=img_pixels,
        image_id="unique_id",
        eye_side="left"  # or "right"
    )
)

# 4. Extract iris template (contains iris_codes and mask_codes)
iris_template = result["iris_template"]
```

#### Matching with Hamming Distance

```python
from iris.nodes.matcher import HammingDistanceMatcher

# Create matcher
matcher = iris.HammingDistanceMatcher(
    rotation_shift=15,      # Rotation compensation in columns
    normalise=True,         # Normalize Hamming distance
    norm_mean=0.45,        # Non-match mean for normalization
    separate_half_matching=True  # Match upper/lower halves separately
)

# Compare two iris templates
distance = matcher.run(template_probe, template_gallery)

# Lower distance = better match
# Typical threshold: 0.32-0.35 for verification
```

#### Iris Template Structure

```python
@dataclass
class IrisTemplate:
    iris_codes: List[np.ndarray]      # Binary iris code arrays
    mask_codes: List[np.ndarray]      # Validity mask arrays
    iris_code_version: str            # Version identifier (e.g., "v2.1")
```

#### Dependencies
```
# Base requirements
numpy
opencv-python
pydantic==1.10.16
pyyaml
onnx
onnxruntime
huggingface-hub
```

---

## 4. Implementation Plan

### 4.1 Project Structure

```
multi-modal-biometric/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── settings.yaml           # System configuration
│   └── thresholds.yaml         # Match threshold settings
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── camera.py           # Camera interface
│   │   └── image_quality.py    # Quality assessment
│   ├── face/
│   │   ├── __init__.py
│   │   ├── detector.py         # Face detection wrapper
│   │   ├── encoder.py          # Face encoding
│   │   └── matcher.py          # Face comparison
│   ├── iris/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Iris pipeline wrapper
│   │   ├── encoder.py          # Iris encoding
│   │   └── matcher.py          # Iris comparison
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── normalizer.py       # Score normalization
│   │   ├── decision.py         # Decision fusion logic
│   │   └── strategies.py       # Fusion strategies (AND, weighted, etc.)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py           # Data models
│   │   ├── storage.py          # Template storage
│   │   └── encryption.py       # Template encryption
│   └── utils/
│       ├── __init__.py
│       ├── logging.py          # Logging utilities
│       └── visualization.py    # Debug visualization
├── tests/
│   ├── __init__.py
│   ├── test_face.py
│   ├── test_iris.py
│   ├── test_fusion.py
│   └── test_integration.py
├── data/
│   ├── enrolled/               # Enrolled user data (encrypted)
│   └── samples/                # Test samples
└── docs/
    ├── IMPLEMENTATION_DOCUMENT.md
    ├── API.md
    └── DEPLOYMENT.md
```

### 4.2 Development Phases

#### Phase 1: Foundation (Week 1-2)
- [ ] Set up project structure
- [ ] Install and configure dependencies
- [ ] Implement camera capture module
- [ ] Create basic configuration management

#### Phase 2: Face Recognition Module (Week 2-3)
- [ ] Implement face detection wrapper
- [ ] Implement face encoding functionality
- [ ] Implement face comparison with configurable thresholds
- [ ] Add quality checks (blur, lighting, pose)
- [ ] Unit tests for face module

#### Phase 3: Iris Recognition Module (Week 3-4)
- [ ] Implement IRISPipeline wrapper
- [ ] Handle IR camera integration
- [ ] Implement HammingDistanceMatcher integration
- [ ] Add quality checks (occlusion, gaze, blur)
- [ ] Unit tests for iris module

#### Phase 4: Fusion Engine (Week 4-5)
- [ ] Implement score normalization (min-max, z-score)
- [ ] Implement decision fusion strategies:
  - AND Rule (both must pass)
  - SUM Rule (weighted sum of scores)
  - PRODUCT Rule
- [ ] Configurable threshold management
- [ ] Unit tests for fusion module

#### Phase 5: Database & Storage (Week 5-6)
- [ ] Design database schema for user enrollment
- [ ] Implement encrypted template storage
- [ ] Add user management (enroll, delete, update)
- [ ] Implement template versioning

#### Phase 6: Integration & Testing (Week 6-7)
- [ ] End-to-end integration testing
- [ ] Performance benchmarking
- [ ] Security audit
- [ ] Documentation

#### Phase 7: UI & Deployment (Week 7-8)
- [ ] Create simple UI for enrollment/verification
- [ ] Containerization (Docker)
- [ ] Deployment documentation

---

## 5. Core Implementation Details

### 5.1 Enrollment Workflow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Capture    │────▶│   Quality    │────▶│   Encode     │────▶│   Store      │
│   Images     │     │   Check      │     │   Templates  │     │   Encrypted  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
      │                    │                    │                    │
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
 • RGB Face            • Face: blur,       • Face: 128-D        • AES-256
 • IR Iris               lighting,           encoding             encryption
   (both eyes)           pose              • Iris: Binary        • User ID
                       • Iris: occlusion,    iris code            mapping
                         gaze, quality       + mask code
```

### 5.2 Verification Workflow

```python
def verify_user(user_id: str, face_image: np.ndarray, iris_image: np.ndarray) -> AuthResult:
    """
    Dual-modal biometric verification.
    
    Returns AuthResult with:
        - authenticated: bool
        - face_score: float
        - iris_score: float
        - fusion_score: float
        - confidence: float
    """
    # 1. Retrieve enrolled templates
    enrolled = database.get_user_templates(user_id)
    
    # 2. Face verification
    face_encoding = face_encoder.encode(face_image)
    face_distance = face_matcher.compare(enrolled.face_template, face_encoding)
    face_score = 1.0 - face_distance  # Convert distance to similarity
    
    # 3. Iris verification
    iris_template = iris_pipeline.process(iris_image)
    iris_distance = iris_matcher.compare(enrolled.iris_template, iris_template)
    iris_score = 1.0 - iris_distance  # Normalized score
    
    # 4. Score fusion (AND rule: both must pass)
    face_pass = face_score >= config.FACE_THRESHOLD
    iris_pass = iris_score >= config.IRIS_THRESHOLD
    
    authenticated = face_pass and iris_pass
    fusion_score = (face_score + iris_score) / 2  # Simple average
    
    return AuthResult(
        authenticated=authenticated,
        face_score=face_score,
        iris_score=iris_score,
        fusion_score=fusion_score,
        confidence=min(face_score, iris_score)  # Conservative confidence
    )
```

### 5.3 Score Normalization

Since face recognition uses Euclidean distance (0.0-1.0, lower=better) and iris uses Hamming distance (0.0-0.5+, lower=better), we need to normalize scores:

```python
def normalize_face_score(distance: float, threshold: float = 0.6) -> float:
    """Convert face distance to similarity score [0, 1]."""
    # Clamp to valid range
    distance = max(0.0, min(distance, 1.0))
    # Convert: 0 distance = 1.0 score, threshold distance = 0.5 score
    return max(0.0, 1.0 - (distance / threshold) * 0.5)

def normalize_iris_score(hamming_distance: float, threshold: float = 0.35) -> float:
    """Convert Hamming distance to similarity score [0, 1]."""
    # Clamp to valid range
    distance = max(0.0, min(hamming_distance, 0.5))
    # Convert: 0 distance = 1.0 score, threshold = 0.5 score
    return max(0.0, 1.0 - (distance / threshold) * 0.5)
```

### 5.4 Decision Fusion Strategies

```python
class FusionStrategy(Enum):
    AND_RULE = "and"          # Both modalities must pass
    OR_RULE = "or"            # At least one must pass (NOT recommended for security)
    SUM_RULE = "sum"          # Sum of normalized scores
    WEIGHTED_SUM = "weighted" # Weighted sum based on reliability
    PRODUCT_RULE = "product"  # Product of likelihoods

class FusionEngine:
    def __init__(self, strategy: FusionStrategy, weights: dict = None):
        self.strategy = strategy
        self.weights = weights or {"face": 0.5, "iris": 0.5}
    
    def fuse(self, face_score: float, iris_score: float) -> tuple[bool, float]:
        if self.strategy == FusionStrategy.AND_RULE:
            passed = (face_score >= self.face_threshold and 
                     iris_score >= self.iris_threshold)
            score = min(face_score, iris_score)
            
        elif self.strategy == FusionStrategy.SUM_RULE:
            score = (face_score + iris_score) / 2
            passed = score >= self.fusion_threshold
            
        elif self.strategy == FusionStrategy.WEIGHTED_SUM:
            score = (self.weights["face"] * face_score + 
                    self.weights["iris"] * iris_score)
            passed = score >= self.fusion_threshold
            
        return passed, score
```

---

## 6. Configuration

### 6.1 Default Thresholds

```yaml
# config/thresholds.yaml
face_recognition:
  detection_model: "cnn"       # USE CNN - GTX 1660 handles it efficiently
  encoding_model: "large"      # Use "large" for better accuracy (GPU accelerated)
  num_jitters: 2               # Slightly higher for better encoding (GPU handles it)
  match_tolerance: 0.6         # Distance threshold (lower = stricter)
  strict_tolerance: 0.5        # For high-security scenarios
  use_cuda: true               # Enable CUDA acceleration

iris_recognition:
  rotation_shift: 15           # Rotation compensation columns
  normalise: true
  norm_mean: 0.45
  norm_gradient: 0.00005
  match_threshold: 0.35        # Hamming distance threshold
  strict_threshold: 0.32       # For high-security scenarios

fusion:
  strategy: "weighted"         # Weighted sum (better for unequal reliability)
  face_weight: 0.7             # Higher weight - face is more reliable with RGB webcam
  iris_weight: 0.3             # Lower weight - RGB iris less reliable
  combined_threshold: 0.65     # For weighted strategy
  
  # AND rule fallback for high-security scenarios
  require_both: true           # Still require both to pass minimum thresholds
  face_minimum: 0.55           # Face must at least pass this
  iris_minimum: 0.45           # Iris must at least pass this (relaxed)
```

### 6.2 Hardware Requirements

**Your Hardware:** AMD Ryzen 5 5600H + NVIDIA GTX 1660 + Built-in 720p Webcam

| Component | Ideal | Your Setup | Status |
|-----------|-------|------------|--------|
| CPU | Intel i5 / Ryzen 5 | **Ryzen 5 5600H** (6C/12T, 3.3-4.2GHz) | ✅ Excellent |
| RAM | 8 GB | 8-16 GB (typical) | ✅ Good |
| GPU | GTX 1060 (for CNN) | **GTX 1660** (6GB VRAM, 1408 CUDA cores) | ✅ Excellent |
| Face Camera | 1080p RGB | **720p Built-in Webcam** | ⚠️ Adequate |
| Iris Camera | NIR Camera | **720p Built-in Webcam (RGB)** | ⚠️ Limited* |
| Storage | 50 GB SSD | - | - |

> ⚠️ **Important Limitation:** You're using a single RGB webcam instead of a dedicated NIR (Near-Infrared) iris camera. See **Section 6.4** for implications and adaptations.

#### GPU Optimization Notes (GTX 1660)

 Your GTX 1660 is well-suited for this project:
- **CUDA Cores:** 1408 - sufficient for CNN-based face detection
- **VRAM:** 6GB - ample for both dlib CNN and ONNX inference
- **Compute Capability:** 7.5 (Turing) - supports modern CUDA features

**Recommended Settings for Your Hardware:**
```yaml
# Optimized for Ryzen 5 5600H + GTX 1660
face_recognition:
  detection_model: "cnn"      # USE CNN - your GPU handles it well
  batch_size: 4               # Process multiple faces in parallel
  
iris_recognition:
  execution_provider: "CUDAExecutionProvider"  # Use GPU for ONNX
  mode: "visible_light"       # RGB webcam mode (not NIR)
  
performance:
  face_detection_gpu: true
  iris_inference_gpu: true
  parallel_processing: true   # Use both CPU cores and GPU
```

### 6.3 Camera Setup (Single 720p Webcam)

**Your Setup:** Single built-in 720p RGB webcam for BOTH face and iris capture.

```
┌─────────────────────────────────────────────────────────────┐
│                    SINGLE CAMERA WORKFLOW                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐                                          │
│   │   720p RGB   │                                          │
│   │   Webcam     │                                          │
│   └──────┬───────┘                                          │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────────────────────────────────────┐          │
│   │         CAPTURE SEQUENCE                     │          │
│   │                                              │          │
│   │   Step 1: Face Capture (normal distance)    │          │
│   │           └─▶ ~50-80cm from camera          │          │
│   │                                              │          │
│   │   Step 2: Eye/Iris Capture (close-up)       │          │
│   │           └─▶ ~15-25cm from camera          │          │
│   │           └─▶ User guided to position eye   │          │
│   │                                              │          │
│   └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Webcam Specifications:**
- Resolution: 1280x720 (720p)
- Type: RGB color (visible light)
- Limitations: No NIR capability, fixed focus likely

---

### 6.4 Visible-Light Iris Recognition (RGB Webcam Adaptation)

> ⚠️ **Critical Consideration:** The `open-iris` library is designed for **Near-Infrared (NIR)** iris images, not RGB. Using a standard webcam will require adaptations and will have **reduced accuracy**.

#### Why NIR is Preferred for Iris Recognition:

| Aspect | NIR Camera | RGB Webcam (Your Setup) |
|--------|------------|-------------------------|
| Iris texture visibility | Excellent (melanin transparent to NIR) | Limited (dark irises appear uniform) |
| Pupil/iris contrast | High | Medium-Low |
| Works with dark eyes | ✅ Yes | ⚠️ Challenging |
| Works with light eyes | ✅ Yes | ✅ Better than dark |
| Ambient light sensitivity | Low | High |
| Expected accuracy | ~99.9% | ~85-95%* |

*Accuracy varies significantly based on eye color and lighting conditions.

#### Adaptation Strategy for RGB Webcam:

```python
# Visible-light iris preprocessing pipeline
class RGBIrisPreprocessor:
    """
    Adapt RGB webcam images for iris recognition.
    This is a workaround - accuracy will be lower than NIR.
    """
    
    def preprocess(self, rgb_image: np.ndarray) -> np.ndarray:
        # 1. Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Detect and crop eye region (using face landmarks)
        eye_region = self.extract_eye_region(enhanced)
        
        # 4. Resize to expected input size
        resized = cv2.resize(eye_region, (640, 480))
        
        # 5. Additional contrast enhancement for iris texture
        iris_enhanced = self.enhance_iris_texture(resized)
        
        return iris_enhanced
    
    def enhance_iris_texture(self, eye_image: np.ndarray) -> np.ndarray:
        # Enhance iris patterns visible in RGB
        # Apply unsharp masking
        gaussian = cv2.GaussianBlur(eye_image, (0, 0), 2.0)
        enhanced = cv2.addWeighted(eye_image, 1.5, gaussian, -0.5, 0)
        return enhanced
```

#### Recommendations for RGB Iris Capture:

1. **Lighting Conditions:**
   - Use bright, diffused lighting (avoid harsh shadows)
   - Avoid direct light causing reflections on the eye
   - Consistent lighting during enrollment and verification

2. **User Positioning:**
   - Eye should be 15-25cm from camera
   - Guide user with on-screen oval/circle target
   - Capture when eye is centered and fully open

3. **Eye Color Considerations:**
   - Light eyes (blue, green, hazel): Better texture visibility
   - Dark eyes (brown, black): More challenging, may need to **lower threshold** or rely more heavily on face recognition

4. **Adjusted Thresholds:**
   ```yaml
   # Adjusted for RGB webcam (less strict due to lower quality)
   iris_recognition:
     match_threshold: 0.40        # Relaxed from 0.35 (NIR)
     strict_threshold: 0.38       # Relaxed from 0.32 (NIR)
     min_quality_score: 0.5       # Accept lower quality images
   ```

#### Alternative: Hybrid Approach

Given the RGB webcam limitation, consider a **weighted hybrid** where face recognition carries more weight:

```yaml
fusion:
  strategy: "weighted"
  face_weight: 0.7              # Higher weight on reliable face recognition
  iris_weight: 0.3              # Lower weight on RGB iris (less reliable)
  combined_threshold: 0.65
  
  # Fallback: If iris quality is too low, use face-only with stricter threshold
  fallback_face_only: true
  face_only_threshold: 0.45     # Stricter when iris unavailable
```

#### Future Upgrade Path

If you want to improve iris recognition accuracy later, consider:

| Option | Cost | Improvement |
|--------|------|-------------|
| USB NIR Camera (e.g., ELP IR camera) | $30-50 | Significant |
| Dedicated iris scanner (e.g., IriTech) | $200-500 | Excellent |
| Dual camera setup (RGB + NIR) | $50-100 | Good |

---

## 7. Security Considerations

### 7.1 Template Protection

```python
from cryptography.fernet import Fernet
import hashlib

class TemplateProtection:
    def __init__(self, master_key: bytes):
        # Derive encryption key from master key
        self.key = hashlib.pbkdf2_hmac(
            'sha256', master_key, b'salt', 100000
        )
        self.cipher = Fernet(base64.urlsafe_b64encode(self.key[:32]))
    
    def encrypt_template(self, template: bytes) -> bytes:
        return self.cipher.encrypt(template)
    
    def decrypt_template(self, encrypted: bytes) -> bytes:
        return self.cipher.decrypt(encrypted)
```

### 7.2 Anti-Spoofing Measures

| Attack Type | Face Countermeasure | Iris Countermeasure |
|------------|---------------------|---------------------|
| Photo attack | Liveness detection (blink, head movement) | NIR vs visible light response |
| Video replay | 3D depth sensing | Pupil dilation response |
| Mask attack | Texture analysis, thermal imaging | Specular reflection analysis |
| Printed iris | N/A | Texture frequency analysis |

### 7.3 Data Privacy

- Templates stored encrypted at rest (AES-256)
- Templates never transmitted in plaintext
- User consent required for enrollment
- Right to deletion of biometric data
- Audit logging of all access attempts

---

## 8. Performance Metrics

### 8.1 Expected Performance

**Estimated Performance on Your Hardware (Ryzen 5 5600H + GTX 1660 + 720p RGB Webcam):**

| Metric | Face Recognition | Iris Recognition (RGB)* | Combined |
|--------|-----------------|------------------------|----------|
| FAR (False Accept Rate) | 0.1% | 1-2% | ~0.01% |
| FRR (False Reject Rate) | 1-2% | 5-10% | 6-12% |
| Processing Time (GPU) | **50-80ms** | **100-200ms** | **200-350ms** |
| Processing Time (CPU only) | 150-300ms | 300-600ms | 500-1000ms |
| Template Size | 512 bytes | ~8 KB | ~9 KB |

> *RGB webcam iris recognition has significantly lower accuracy than NIR. Light-colored eyes will perform better.

#### Accuracy by Eye Color (RGB Webcam Estimate)

| Eye Color | Iris Recognition Accuracy | Recommendation |
|-----------|---------------------------|----------------|
| Blue/Gray | 90-95% | Works reasonably well |
| Green/Hazel | 85-92% | Works with good lighting |
| Light Brown | 80-88% | May need threshold adjustment |
| Dark Brown/Black | 70-82% | Consider higher face weight |

#### GPU Acceleration Benefits (GTX 1660)

| Operation | CPU (HOG) | GPU (CNN) | Speedup |
|-----------|-----------|-----------|--------|
| Face Detection | ~150ms | ~30ms | **5x** |
| Face Encoding | ~100ms | ~40ms | **2.5x** |
| Iris Segmentation | ~200ms | ~80ms | **2.5x** |
| Iris Encoding | ~100ms | ~50ms | **2x** |

> 💡 **Tip:** With GPU acceleration, you can achieve near real-time authentication (~3-5 FPS for continuous verification).

### 8.2 Benchmarking Plan

```python
def benchmark_system():
    """Run performance benchmarks."""
    metrics = {
        "enrollment_time": [],
        "verification_time": [],
        "face_encoding_time": [],
        "iris_encoding_time": [],
        "fusion_time": []
    }
    
    for _ in range(100):
        # Benchmark each component
        t0 = time.time()
        face_encoding = face_encoder.encode(face_image)
        metrics["face_encoding_time"].append(time.time() - t0)
        
        t0 = time.time()
        iris_template = iris_pipeline.process(iris_image)
        metrics["iris_encoding_time"].append(time.time() - t0)
        
        # ... etc
    
    return {k: {"mean": np.mean(v), "std": np.std(v)} 
            for k, v in metrics.items()}
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_face.py
class TestFaceRecognition:
    def test_face_detection(self):
        """Test face detection on known images."""
        
    def test_face_encoding_consistency(self):
        """Same face should produce similar encodings."""
        
    def test_face_matching_accuracy(self):
        """Known pairs should match, impostors should not."""

# tests/test_iris.py  
class TestIrisRecognition:
    def test_iris_pipeline_execution(self):
        """Pipeline should complete without errors."""
        
    def test_iris_template_structure(self):
        """Template should have correct structure."""
        
    def test_hamming_distance_calculation(self):
        """Distance calculation should be correct."""
```

### 9.2 Integration Tests

```python
# tests/test_integration.py
class TestDualAuthentication:
    def test_genuine_user_authentication(self):
        """Enrolled user with valid biometrics should authenticate."""
        
    def test_impostor_rejection(self):
        """Non-enrolled user should be rejected."""
        
    def test_partial_match_rejection(self):
        """User with only face OR iris match should be rejected."""
        
    def test_enrollment_verification_cycle(self):
        """Full enrollment and verification workflow."""
```

---

## 10. Dependencies Summary

### 10.1 requirements.txt

```txt
# Core dependencies
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0

# Face recognition
face_recognition>=1.3.0
dlib>=19.22.0
face_recognition_models>=0.3.0
Click>=8.0.0

# Iris recognition
open-iris>=1.0.0
pydantic>=1.10.0,<2.0.0
onnx>=1.12.0
onnxruntime>=1.12.0
huggingface-hub>=0.10.0
pyyaml>=6.0

# Database
sqlalchemy>=1.4.0
cryptography>=3.4.0

# Utilities
tqdm>=4.60.0
loguru>=0.6.0

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0

# GPU support (RECOMMENDED for GTX 1660)
onnxruntime-gpu>=1.12.0      # CUDA-accelerated ONNX inference
# Note: dlib with CUDA requires manual compilation with CUDA toolkit
# See installation commands below for GPU-enabled dlib
```

### 10.2 Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev

# ============================================
# GPU SETUP (Recommended for GTX 1660)
# ============================================

# 1. Verify CUDA installation (should be 11.x or 12.x)
nvidia-smi
nvcc --version

# 2. Install CUDA toolkit if not present (Ubuntu)
# sudo apt install nvidia-cuda-toolkit

# 3. Install dlib with CUDA support (GTX 1660 optimized)
# Option A: Try pip first (may not have CUDA)
pip install dlib

# Option B: Build dlib with CUDA (recommended for CNN acceleration)
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build . --config Release
cd ..
python setup.py install
cd ..

# 4. Install face_recognition
pip install face_recognition

# 5. Install ONNX Runtime with GPU support (for open-iris)
pip install onnxruntime-gpu  # Uses your GTX 1660

# 6. Install open-iris
IRIS_ENV=SERVER pip install open-iris

# 7. Install remaining dependencies
pip install -r requirements.txt

# ============================================
# Verify GPU is being used
# ============================================
python -c "import dlib; print('CUDA available:', dlib.DLIB_USE_CUDA)"
python -c "import onnxruntime as ort; print('Available providers:', ort.get_available_providers())"
```

---

## 11. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **RGB iris low accuracy** | High | High | Use weighted fusion favoring face; add fallback mode |
| **Dark eye color poor iris match** | High | Medium | Adaptive thresholds; increase face weight dynamically |
| Poor lighting conditions | Medium | Medium | Add quality check; guide user to better lighting |
| 720p resolution insufficient | Medium | Low | Face works fine; iris needs close-up capture |
| dlib compilation fails | Medium | Medium | Provide pre-built wheels or Docker image |
| High FRR in production | High | Medium | Adjust thresholds, add retry mechanism |
| Template database compromise | Critical | Low | Encrypt templates, implement access controls |
| Spoofing attacks | High | Medium | Implement liveness detection |

---

## 12. Future Enhancements

1. **Liveness Detection**
   - Add anti-spoofing for face (blink detection, 3D depth)
   - Add anti-spoofing for iris (pupil response, reflection analysis)

2. **Performance Optimization**
   - GPU acceleration for face recognition (CNN model)
   - Batch processing for enrollment

3. **Additional Modalities**
   - Fingerprint integration (third factor)
   - Voice recognition (fourth factor)

4. **Advanced Fusion**
   - Machine learning-based score fusion
   - Adaptive thresholds based on quality scores

5. **Cloud Deployment**
   - REST API for remote verification
   - Scalable microservices architecture

---

## 13. Glossary

| Term | Definition |
|------|------------|
| **FAR** | False Accept Rate - probability of accepting an impostor |
| **FRR** | False Reject Rate - probability of rejecting a genuine user |
| **Hamming Distance** | Number of bit positions that differ between two binary codes |
| **NIR** | Near-Infrared - light wavelength used for iris imaging (~850nm) |
| **Template** | Mathematical representation of biometric features |
| **Fusion** | Combining multiple biometric modalities for decision |
| **Enrollment** | Process of registering a user's biometric templates |
| **Verification** | 1:1 comparison to confirm claimed identity |

---

## 14. Next Steps

After reviewing this document, the recommended next steps are:

1. **Review and Approve** - Validate architecture and approach
2. **Hardware Procurement** - Order IR camera if not available
3. **Environment Setup** - Set up development environment with dependencies
4. **Prototype Development** - Start with Phase 1 implementation
5. **Iterative Testing** - Test each module as it's developed

---

**Document Status:** Ready for Review  
**Last Updated:** January 10, 2026
