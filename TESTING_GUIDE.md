# Testing Guide - Multi-Modal Biometric System

This guide walks you through testing the system end-to-end — both the desktop
GUI and the CLI. The CLI sections below focus on the face factor; the GUI and
the multi-modal sections cover the full face + iris + liveness flow.

## ✅ System Status

**All components are functional and ready for testing with a real webcam.**

```
✅ Camera interface working
✅ Face detection/encoding working
✅ Iris (eye-region) recognition working
✅ Blink-based liveness working
✅ Weighted fusion + AND gate working
✅ Database storage working
✅ Enrollment / verification / identification workflows ready
✅ CustomTkinter GUI (in-process, frame-driven)
✅ CLI commands functional
```

## 🖥️ GUI Testing (recommended first)

```bash
python app.py
```

1. **Enroll:** type a User ID + Name, click **Enroll User**. Watch Step 1/2
   (face samples) then Step 2/2 (iris samples). Badge → **ENROLLED**.
2. **Authenticate:** type the User ID, click **Authenticate**. Step 1/2 face
   passes, then Step 2/2 asks you to **blink**. Badge → **✓ AUTHENTICATED**
   with a face/iris/combined breakdown.
3. **Negative — wrong person:** authenticate as an enrolled ID while someone
   else is in frame → **✗ DENIED**.
4. **Negative — face-pass / iris-fail:** this must show **✗ DENIED** (it was a
   historical bug where it showed "authenticated").
5. **Negative — liveness:** hold a static photo of the enrolled face to the
   camera → iris step never blinks → **✗ DENIED** after the liveness timeout.

> The preview slows to ~5 fps during Verify/Authenticate (per-frame detection on
> the UI thread). That is expected — not a hang. **Blink deliberately.**

## 🔗 Multi-Modal CLI Testing

```bash
python test_fusionauth.py sahil
```

Runs face → iris (with blink liveness) → weighted fusion. The printed
`MultiModalVerificationResult` should show `success=True` only when both factors
pass and `combined_confidence` clears the threshold; face-pass/iris-fail returns
`success=False`.

## 🔧 Pre-Testing Checklist

Run these commands to verify system readiness:

```bash
# 1. Check all commands are available
python demo/face_demo.py --help

# 2. Verify database is initialized
python demo/face_demo.py list

# 3. Check stats (should show 0 users)
python demo/face_demo.py stats
```

**Expected Output:**
- Help shows 7 commands: enroll, verify, identify, list, delete, stats, continuous
- List shows "No users enrolled"
- Stats shows 0 total attempts

## 🧪 Test Sequence

### Test 1: Enroll Yourself

```bash
python demo/face_demo.py enroll sahil "Sahil" -s 3
```

**What will happen:**
1. Camera window opens showing live video
2. System displays: "Face Enrollment for: Sahil (sahil)"
3. Face detection starts automatically
4. When your face is detected, green box appears around it
5. System auto-captures 3 samples (with 0.5s between each)
6. Progress displayed: "Captured 1/3", "Captured 2/3", "Captured 3/3"
7. Enrollment completes and saves to database
8. Camera closes automatically

**Instructions:**
- Look directly at camera
- Keep still for first capture
- Move slightly (tilt head, turn slightly) for diversity
- Don't move too much - face must stay in frame

**Expected Success Message:**
```
✓ Enrolled: Sahil (sahil) with 3 samples
```

**Manual Controls:**
- Press **'c'** to manually capture a sample
- Press **'q'** to cancel enrollment
- Press **'ESC'** to quit

---

### Test 2: List Enrolled Users

```bash
python demo/face_demo.py list
```

**Expected Output:**
```
Enrolled Users (1):
  1. sahil - Sahil (3 samples)
```

---

### Test 3: Verify Your Identity (1:1 Match)

```bash
python demo/face_demo.py verify sahil
```

**What will happen:**
1. Camera opens
2. System shows: "Verifying identity for: sahil"
3. Face detection starts
4. When face detected, system compares to your template
5. Match result displayed with confidence score

**Expected Success:**
```
✓ Verification PASSED
  User: sahil (Sahil)
  Match distance: 0.42
  Confidence: 73.5%
  Attempts: 1/3
```

**If Failed:**
- Make sure lighting is good
- Look directly at camera
- System allows 3 attempts
- Distance must be ≤ 0.6 to pass

---

### Test 4: Identify (1:N Search)

```bash
python demo/face_demo.py identify
```

**What will happen:**
1. Camera opens
2. System searches all enrolled users
3. Finds best match from database

**Expected Success:**
```
✓ MATCH FOUND
  User: sahil (Sahil)
  Match distance: 0.45
  Confidence: 68.2%
```

---

### Test 5: Enroll Another Person

If you have someone else nearby:

```bash
python demo/face_demo.py enroll friend "Friend Name" -s 3
```

Then test identification again - system should correctly identify each person.

---

### Test 6: Continuous Verification Mode

```bash
python demo/face_demo.py continuous verify sahil
```

**What will happen:**
- Camera stays open continuously
- Real-time verification against your template
- Shows match status overlaid on video
- Press 'q' to quit

**Use Cases:**
- Testing system in different lighting
- Testing with glasses on/off
- Testing with different poses
- Simulating real authentication scenario

---

### Test 7: Check Statistics

```bash
python demo/face_demo.py stats
```

**Expected Output:**
```
Verification Statistics (last 30 days):
  Total attempts: 5
  Successful: 4
  Failed: 1
  Success rate: 80.0%
```

---

### Test 8: Delete a User

```bash
python demo/face_demo.py delete friend
```

**Expected Output:**
```
✓ User 'friend' deleted successfully
```

Then verify deletion:
```bash
python demo/face_demo.py list
```

---

## 🐛 Troubleshooting

### Camera Won't Open

**Error:** `Camera failed to open`

**Solutions:**
```bash
# Check available cameras
ls /dev/video*

# Try different camera IDs
python demo/face_demo.py enroll user "Name" -c 1  # Try camera 1
python demo/face_demo.py enroll user "Name" -c 2  # Try camera 2
```

### No Face Detected

**Issue:** Green box not appearing

**Solutions:**
- Improve lighting (face well-lit from front)
- Move closer to camera (but not too close)
- Ensure face is fully visible (not tilted too much)
- Remove obstructions (hands, hair covering face)
- Check camera focus

### Verification Failing

**Issue:** "✗ Verification FAILED"

**Possible Causes:**
1. **Different lighting** - Enroll and verify in similar conditions
2. **Face angle** - Look straight at camera like during enrollment  
3. **Glasses/accessories** - If you enrolled with glasses, verify with them
4. **Distance from camera** - Try to match enrollment distance
5. **Tolerance too strict** - Edit `config/settings.yaml`:
   ```yaml
   face_recognition:
     match_tolerance: 0.65  # Increase from 0.6 (less strict)
   ```

### Low Confidence Scores

**Issue:** Match succeeds but confidence < 50%

**Solutions:**
1. Re-enroll with more samples:
   ```bash
   python demo/face_demo.py enroll sahil "Sahil" -s 10
   ```

2. Improve sample diversity:
   - Tilt head slightly between captures
   - Different expressions (neutral, smile)
   - Slight left/right rotation

3. Check camera quality:
   - Clean camera lens
   - Increase resolution in settings

### System Freezing

**Issue:** Camera preview frozen

**Solution:**
- Press 'q' or ESC to quit
- If stuck, Ctrl+C to force stop
- Restart camera:
  ```bash
  sudo modprobe -r uvcvideo
  sudo modprobe uvcvideo
  ```

---

## 📊 Expected Performance

Based on your hardware (Ryzen 5 5600H + GTX 1660):

| Operation | Time | Notes |
|-----------|------|-------|
| Face Detection | 50-100ms | CNN on GPU |
| Face Encoding | 100-150ms | Per face |
| Matching | <1ms | Very fast |
| Total Verification | 150-250ms | Acceptable for real-time |

---

## 🎯 Success Criteria

After testing, your system should achieve:

- ✅ Enrollment completes in <30 seconds
- ✅ Face detected within 2 seconds
- ✅ Verification succeeds with >60% confidence  
- ✅ False rejection rate < 5% (same person, good conditions)
- ✅ No false acceptances (different people never match)
- ✅ Database persists across sessions

---

## 📝 Test Report Template

After testing, fill this out:

```
=== FACE RECOGNITION TEST REPORT ===

Date: _______________
Tester: ______________

Test 1 - Enrollment:
  ☐ Pass  ☐ Fail
  Samples captured: ___/3
  Issues: _________________

Test 2 - List Users:
  ☐ Pass  ☐ Fail
  Users shown: ___________

Test 3 - Verification:
  ☐ Pass  ☐ Fail
  Confidence: ___%
  Attempts: ___/3
  Distance: _____

Test 4 - Identification:
  ☐ Pass  ☐ Fail
  Correct match: ☐ Yes ☐ No

Test 5 - Multi-user (if tested):
  ☐ Pass  ☐ Fail
  Confusion occurred: ☐ Yes ☐ No

Test 6 - Continuous Mode:
  ☐ Pass  ☐ Fail
  FPS: ~___ fps

Overall Assessment:
  ☐ Ready for Phase 2 (Iris Recognition)
  ☐ Needs improvements

Notes:
_____________________________
_____________________________
```

---

## 🚀 Next Steps After Testing

Once face recognition is working well:

1. **Report Results**: Share test report with any issues
2. **Phase 2 Planning**: Begin iris recognition integration
3. **Optimization** (optional): Fine-tune tolerance, sample count
4. **Documentation**: Update with any lessons learned

---

## 💡 Pro Tips

1. **Best Lighting**: Soft, diffused light from front (no harsh shadows)
2. **Camera Position**: Eye level, 50-70cm distance
3. **Enrollment**: Enroll in conditions you'll verify in
4. **Multiple Templates**: Enroll same person in different conditions
5. **Testing Variety**: Test with glasses on/off, different expressions
6. **Sample Count**: More samples (5-10) = better accuracy but slower enrollment

---

## ❓ Questions to Answer

After testing, you should know:

- [ ] Does the camera open successfully?
- [ ] Are faces detected reliably?
- [ ] Does enrollment complete without errors?
- [ ] Does verification work in good lighting?
- [ ] What's the typical confidence score?
- [ ] Can the system tell different people apart?
- [ ] Is the system fast enough for real-time use?
- [ ] Are there any camera compatibility issues?

---

**Ready to test? Start with Test 1 and work through sequentially!** 🎉

If you encounter issues, check [USAGE.md](USAGE.md) for detailed troubleshooting.
