"""Modern desktop GUI for the multi-modal biometric authentication system.

Built on CustomTkinter with an embedded live camera preview. The app talks to
the workflows **in-process** through ``BiometricService`` and decides outcomes
from structured results (``result.success``) — not by scraping stdout. That
removes the old security bug where a face-pass / iris-fail showed as authenticated.

All capture (enroll / verify / authenticate) is driven **frame-by-frame on the
Tk main thread** through the embedded preview. There are no OpenCV windows and
no camera work on background threads, because OpenCV's HighGUI is not safe to
drive from a worker thread while Tk owns the event loop (that combination caused
the app to hang). Overlays are drawn directly onto the frame buffer with
``cv2.putText`` / ``cv2.rectangle`` (array ops, not HighGUI).
"""

import logging
import time

import cv2
import customtkinter as ctk
from PIL import Image

from src.workflows.service import BiometricService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PREVIEW_SIZE = (480, 360)

COLOR_OK = "#2fa572"
COLOR_FAIL = "#d9534f"
COLOR_BUSY = "#3b8ed0"
COLOR_IDLE = "#4a4a4a"

# BGR overlay colors (drawn onto the frame buffer)
BGR_GREEN = (0, 255, 0)
BGR_RED = (0, 0, 255)
BGR_ORANGE = (0, 165, 255)
BGR_WHITE = (255, 255, 255)


class BiometricApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Multi-Modal Biometric Authentication System")
        self.geometry("1040x680")
        self.minsize(940, 620)

        self.service = BiometricService()
        cfg = self.service.config
        self.face_timeout = cfg.verification.timeout_seconds
        self.iris_timeout = cfg.liveness.timeout_seconds
        self.ear_threshold = cfg.liveness.ear_threshold
        self.open_threshold = cfg.liveness.ear_threshold * 1.1
        self.min_blinks = cfg.liveness.min_blinks
        self.require_liveness = cfg.verification.require_liveness
        self.iris_samples = cfg.iris.num_samples
        self.sample_interval = cfg.enrollment.sample_interval

        self.op = None              # active operation state dict, or None
        self._preview_image = None  # keep a reference so it isn't GC'd

        self._build_ui()
        self._update_preview()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self, corner_radius=12)
        left.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

        ctk.CTkLabel(left, text="● Live Camera", font=("Arial", 16, "bold")).pack(pady=(14, 6))
        self.preview_label = ctk.CTkLabel(
            left, text="Starting camera…", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1]
        )
        self.preview_label.pack(padx=14, pady=8)

        self.status_badge = ctk.CTkLabel(
            left, text="IDLE", font=("Arial", 26, "bold"),
            fg_color=COLOR_IDLE, corner_radius=10, height=64,
        )
        self.status_badge.pack(fill="x", padx=14, pady=(8, 6))

        self.breakdown = ctk.CTkLabel(left, text="face — · iris — · combined —", font=("Arial", 13))
        self.breakdown.pack(pady=(0, 14))

        right = ctk.CTkFrame(self, corner_radius=12)
        right.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(
            right, text="Multi-Modal Biometric Auth", font=("Arial", 20, "bold")
        ).grid(row=0, column=0, padx=16, pady=(16, 12), sticky="w")

        self.user_id = ctk.CTkEntry(right, placeholder_text="User ID")
        self.user_id.grid(row=1, column=0, padx=16, pady=6, sticky="ew")

        self.name = ctk.CTkEntry(right, placeholder_text="Name (for enrollment)")
        self.name.grid(row=2, column=0, padx=16, pady=6, sticky="ew")

        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.grid(row=3, column=0, padx=12, pady=10, sticky="ew")
        btns.grid_columnconfigure((0, 1), weight=1)

        self.buttons: list[ctk.CTkButton] = []

        def add_btn(parent, text, cmd, row, col, **kw):
            b = ctk.CTkButton(parent, text=text, command=cmd, height=40, **kw)
            b.grid(row=row, column=col, padx=6, pady=6, sticky="ew")
            self.buttons.append(b)
            return b

        add_btn(btns, "Enroll User", self.enroll_user, 0, 0)
        add_btn(btns, "Verify (Face)", self.verify_user, 0, 1)
        add_btn(btns, "Authenticate", self.authenticate_user, 1, 0,
                fg_color=COLOR_OK, hover_color="#268a5e")
        add_btn(btns, "List Users", self.list_users, 1, 1)
        add_btn(btns, "Delete User", self.delete_user, 2, 0,
                fg_color=COLOR_FAIL, hover_color="#b94440")
        add_btn(btns, "Cancel / Clear", self.cancel_or_clear, 2, 1, fg_color="gray30")

        ctk.CTkLabel(right, text="System Output", font=("Arial", 14, "bold")).grid(
            row=6, column=0, padx=16, pady=(8, 2), sticky="w")

        self.output = ctk.CTkTextbox(right, font=("Courier", 12))
        self.output.grid(row=7, column=0, padx=16, pady=(0, 16), sticky="nsew")

    # ------------------------------------------------------------------
    # Main-thread preview + operation loop
    # ------------------------------------------------------------------
    def _update_preview(self):
        frame = None
        try:
            result = self.service.camera.read_frame()
            if result.success and result.frame is not None:
                frame = result.frame
        except Exception:
            logging.debug("preview frame skipped", exc_info=True)

        if frame is not None:
            # Process the active operation against this frame (may finish it).
            if self.op is not None:
                try:
                    self._process_op(frame)
                except Exception:
                    logging.exception("operation step failed")
                    self._finish("✗ ERROR", COLOR_FAIL, ["Operation error (see console)"])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize(PREVIEW_SIZE)
            self._preview_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=PREVIEW_SIZE)
            self.preview_label.configure(image=self._preview_image, text="")

        self.after(20, self._update_preview)

    def _process_op(self, frame):
        kind = self.op["kind"]
        if kind == "verify_face":
            self._step_verify_face(frame)
        elif kind == "authenticate":
            self._step_authenticate(frame)
        elif kind == "enroll":
            self._step_enroll(frame)

    # ------------------------------------------------------------------
    # Small UI helpers
    # ------------------------------------------------------------------
    def write(self, text: str):
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _set_status(self, text: str, color: str):
        self.status_badge.configure(text=text, fg_color=color)

    def _set_breakdown(self, face=None, iris=None, combined=None):
        def fmt(v):
            return f"{v:.0f}%" if isinstance(v, (int, float)) else "—"
        self.breakdown.configure(
            text=f"face {fmt(face)} · iris {fmt(iris)} · combined {fmt(combined)}")

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        for b in self.buttons:
            # Keep the Cancel/Clear button live so the user can abort.
            if b.cget("text") == "Cancel / Clear":
                continue
            b.configure(state=state)

    def _start_op(self, op: dict, badge: str):
        self.op = op
        self._set_busy(True)
        self._set_status(badge, COLOR_BUSY)

    def _finish(self, badge: str, color: str, lines: list[str], breakdown: dict | None = None):
        self.op = None
        self._set_busy(False)
        self._set_status(badge, color)
        for line in lines:
            self.write(line)
        if breakdown is not None:
            self._set_breakdown(**breakdown)

    @staticmethod
    def _draw(frame, lines, color=BGR_WHITE, box=None, box_color=BGR_GREEN):
        if box is not None:
            l, t, r, b = box
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
        y = 28
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 28

    # ------------------------------------------------------------------
    # Verify (face) — 1:1
    # ------------------------------------------------------------------
    def verify_user(self):
        uid = self.user_id.get().strip()
        if not uid:
            self.write("Enter a User ID to verify.")
            return
        template = self.service.load_face_template(uid)
        if template is None:
            self.write(f"User '{uid}' has no face template (not enrolled?).")
            return
        self.write(f"\n===== VERIFYING (face) {uid} =====")
        self._start_op(
            {"kind": "verify_face", "user_id": uid, "template": template,
             "deadline": time.time() + self.face_timeout},
            "VERIFYING…",
        )

    def _step_verify_face(self, frame):
        op = self.op
        m = self.service.match_face(frame, op["template"])

        if not m.found:
            self._draw(frame, ["No face detected", "Look at the camera"], BGR_ORANGE)
        else:
            box_color = BGR_GREEN if m.is_match else BGR_RED
            self._draw(frame, [f"distance {m.distance:.3f}"], box_color, m.box, box_color)

        if m.found and m.is_match:
            self._finish("✓ VERIFIED", COLOR_OK,
                         [f"Verified (confidence {m.confidence:.0f}%, distance {m.distance:.3f})"],
                         {"face": m.confidence})
        elif time.time() > op["deadline"]:
            self._finish("✗ NOT VERIFIED", COLOR_FAIL, ["Verification timed out / no match"],
                         {"face": m.confidence if m.found else None})

    # ------------------------------------------------------------------
    # Authenticate — face then iris (with blink liveness), score fusion
    # ------------------------------------------------------------------
    def authenticate_user(self):
        uid = self.user_id.get().strip()
        if not uid:
            self.write("Enter a User ID to authenticate.")
            return
        template = self.service.load_face_template(uid)
        if template is None:
            self.write(f"User '{uid}' has no face template (not enrolled?).")
            return
        iris = self.service.load_iris_templates(uid)
        if iris is None:
            self.write(f"User '{uid}' has no iris templates (re-enroll needed?).")
            return
        self.write(f"\n===== MULTI-MODAL AUTHENTICATION ({uid}) =====")
        self._start_op(
            {"kind": "authenticate", "user_id": uid, "template": template,
             "iris_left": iris[0], "iris_right": iris[1],
             "phase": "face", "deadline": time.time() + self.face_timeout,
             "face_confidence": 0.0, "face_distance": 1.0,
             "blinks": 0, "eye_closed": False, "iris_candidate": None},
            "AUTHENTICATING…",
        )

    def _step_authenticate(self, frame):
        op = self.op
        if op["phase"] == "face":
            self._auth_face_phase(frame)
        else:
            self._auth_iris_phase(frame)

    def _auth_face_phase(self, frame):
        op = self.op
        m = self.service.match_face(frame, op["template"])

        if not m.found:
            self._draw(frame, ["Step 1/2: FACE", "No face detected"], BGR_ORANGE)
        else:
            box_color = BGR_GREEN if m.is_match else BGR_RED
            self._draw(frame, ["Step 1/2: FACE", f"distance {m.distance:.3f}"], box_color, m.box, box_color)

        if m.found and m.is_match:
            op["face_confidence"] = m.confidence
            op["face_distance"] = m.distance
            op["phase"] = "iris"
            op["deadline"] = time.time() + self.iris_timeout
            self.write(f"  Face passed (confidence {m.confidence:.0f}%). Now blink for liveness…")
            self._set_breakdown(face=m.confidence)
        elif time.time() > op["deadline"]:
            self._deny_auth(face_conf=m.confidence if m.found else 0.0, iris_conf=0.0,
                            reason="Face verification failed")

    def _auth_iris_phase(self, frame):
        op = self.op
        m = self.service.match_iris(frame, op["iris_left"], op["iris_right"])

        if not m.found:
            self._draw(frame, ["Step 2/2: IRIS", "Eyes not detected"], BGR_ORANGE)
        else:
            # Blink state machine (with hysteresis) for liveness.
            if m.ear < self.ear_threshold:
                op["eye_closed"] = True
            elif op["eye_closed"] and m.ear >= self.open_threshold:
                op["eye_closed"] = False
                op["blinks"] += 1

            # Capture the best open-eyed, good-quality iris match as the candidate.
            if m.ear >= self.open_threshold and m.quality_ok:
                if op["iris_candidate"] is None or m.distance < op["iris_candidate"].distance:
                    op["iris_candidate"] = m

            live = "live" if op["blinks"] >= self.min_blinks else "blink please"
            color = BGR_GREEN if op["blinks"] >= self.min_blinks else BGR_ORANGE
            self._draw(frame, ["Step 2/2: IRIS (blink to confirm)",
                               f"blinks {op['blinks']}/{self.min_blinks} ear {m.ear:.2f} [{live}]"], color)

        liveness_ok = (op["blinks"] >= self.min_blinks) or not self.require_liveness
        if liveness_ok and op["iris_candidate"] is not None:
            self._finalize_auth(op["iris_candidate"])
        elif time.time() > op["deadline"]:
            cand = op["iris_candidate"]
            if not liveness_ok:
                self._deny_auth(op["face_confidence"], cand.confidence if cand else 0.0,
                                reason="Liveness check failed (no blink detected)")
            elif cand is None:
                self._deny_auth(op["face_confidence"], 0.0,
                                reason="Could not capture a clear iris sample")
            else:
                self._finalize_auth(cand)

    def _finalize_auth(self, iris_match):
        op = self.op
        face_conf = op["face_confidence"]
        iris_conf = iris_match.confidence
        combined_norm, combined_pct = self.service.fuse(face_conf, iris_conf)

        success = (
            combined_norm >= self.service.config.fusion.combined_threshold
            and iris_match.is_match  # face already passed to reach this phase
        )

        self.service.db.log_verification(
            user_id=op["user_id"], verification_type="combined", success=success,
            face_score=op["face_distance"], iris_score=iris_match.distance,
            combined_score=combined_norm,
        )

        lines = [
            f"  face_ok=True iris_ok={iris_match.is_match} combined={combined_pct:.1f}%",
        ]
        breakdown = {"face": face_conf, "iris": iris_conf, "combined": combined_pct}
        if success:
            self._finish("✓ AUTHENTICATED", COLOR_OK, ["Authentication Successful"] + lines, breakdown)
        else:
            reason = ("Iris did not match" if not iris_match.is_match
                      else f"Combined score {combined_norm:.2f} below threshold")
            self._finish("✗ DENIED", COLOR_FAIL, [reason] + lines, breakdown)

    def _deny_auth(self, face_conf, iris_conf, reason):
        op = self.op
        self.service.db.log_verification(
            user_id=op["user_id"], verification_type="combined", success=False,
            face_score=op.get("face_distance"), combined_score=None,
        )
        self._finish("✗ DENIED", COLOR_FAIL, [reason],
                     {"face": face_conf or None, "iris": iris_conf or None, "combined": None})

    # ------------------------------------------------------------------
    # Enroll — collect diverse face samples, then iris samples
    # ------------------------------------------------------------------
    def enroll_user(self):
        uid = self.user_id.get().strip()
        name = self.name.get().strip()
        if not uid or not name:
            self.write("Enter both User ID and Name to enroll.")
            return
        if self.service.db.get_user(uid) is not None:
            self.write(f"User '{uid}' is already enrolled.")
            return

        workflow = self.service.make_enrollment_workflow()
        session = workflow.start_session(uid, name)
        self.write(f"\n===== ENROLLING {uid} ({name}) =====")
        self.write("Hold still; move slightly between captures for diversity.")
        self._start_op(
            {"kind": "enroll", "user_id": uid, "workflow": workflow, "session": session,
             "phase": "face", "deadline": time.time() + 120.0, "last_capture": 0.0,
             "left_bucket": [], "right_bucket": []},
            "ENROLLING…",
        )

    def _step_enroll(self, frame):
        op = self.op
        if op["phase"] == "face":
            self._enroll_face_phase(frame)
        else:
            self._enroll_iris_phase(frame)

    def _enroll_face_phase(self, frame):
        op = self.op
        session = op["session"]
        now = time.time()

        if now - op["last_capture"] >= self.sample_interval:
            captured, msg, _ = op["workflow"].capture_sample(session, frame)
            if captured:
                op["last_capture"] = now
                self.write(f"✓ {msg}")

        self._draw(frame, ["Step 1/2: FACE samples",
                           f"{session.samples_collected}/{session.target_samples} captured"], BGR_GREEN)

        if session.is_complete:
            op["phase"] = "iris"
            op["deadline"] = now + self.iris_timeout + 4.0
            self.write("Face samples done. Capturing iris (look at the camera)…")
        elif now > op["deadline"]:
            self._finish("✗ FAILED", COLOR_FAIL, ["Enrollment timed out collecting face samples"])

    def _enroll_iris_phase(self, frame):
        op = self.op
        done = self.service.collect_iris_sample(
            frame, op["left_bucket"], op["right_bucket"], self.iris_samples)

        self._draw(frame, ["Step 2/2: IRIS samples",
                           f"L {len(op['left_bucket'])}/{self.iris_samples} "
                           f"R {len(op['right_bucket'])}/{self.iris_samples}"], BGR_GREEN)

        if done or time.time() > op["deadline"]:
            self._finalize_enroll()

    def _finalize_enroll(self):
        op = self.op
        left = op["left_bucket"]
        right = op["right_bucket"]
        iris_sys = self.service.iris_system

        if left and right:
            left_t = iris_sys.average_templates(left)
            right_t = iris_sys.average_templates(right)
        else:
            left_t = right_t = None

        result = op["workflow"].store_enrollment(op["session"], left_t, right_t)

        if result.success:
            iris_note = (f"iris L:{len(left)} R:{len(right)}" if left and right
                         else "WARNING: no iris captured (face only)")
            self._finish("✓ ENROLLED", COLOR_OK, [result.message, f"  {iris_note}"])
        else:
            self._finish("✗ FAILED", COLOR_FAIL, [result.message])

    # ------------------------------------------------------------------
    # User management (fast DB ops — run inline on the main thread)
    # ------------------------------------------------------------------
    def list_users(self):
        self.write("\n===== ENROLLED USERS =====")
        users = self.service.list_users()
        if not users:
            self.write("No users found")
            return
        for u in users:
            self.write(f"{u.user_id} -> {u.name}")

    def delete_user(self):
        uid = self.user_id.get().strip()
        if not uid:
            self.write("Enter a User ID to delete.")
            return
        ok, message = self.service.delete_user(uid)
        self.write(message)

    def cancel_or_clear(self):
        if self.op is not None:
            self.write("Operation cancelled.")
            self._finish("IDLE", COLOR_IDLE, [])
            return
        self.output.delete("1.0", "end")
        self._set_status("IDLE", COLOR_IDLE)
        self._set_breakdown()

    # ------------------------------------------------------------------
    def _on_close(self):
        self.op = None
        try:
            self.service.camera.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = BiometricApp()
    app.mainloop()
