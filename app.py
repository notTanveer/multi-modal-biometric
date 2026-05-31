"""Modern desktop GUI for the multi-modal biometric authentication system.

Built on CustomTkinter with an embedded live camera preview. Unlike the previous
version, this app talks to the workflows **in-process** through
``BiometricService`` and decides outcomes from structured result objects
(``result.success``) — not by string-matching subprocess stdout. That removes the
old security bug where a face-pass / iris-fail was reported as authenticated.

Guided capture steps (face/iris/enroll) still open their own OpenCV windows with
alignment overlays and the blink-liveness prompt; the embedded preview is paused
and the camera released while an operation runs, then resumed afterwards.
"""

import logging
import queue
import threading

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


class BiometricApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Multi-Modal Biometric Authentication System")
        self.geometry("1040x680")
        self.minsize(940, 620)

        self.service = BiometricService()

        self.ui_queue: queue.Queue = queue.Queue()
        self.busy = False
        self.preview_active = True
        self._preview_image = None  # keep a reference so it isn't GC'd

        self._build_ui()
        self._update_preview()
        self.after(100, self._process_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Left: live preview ----
        left = ctk.CTkFrame(self, corner_radius=12)
        left.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")

        ctk.CTkLabel(left, text="● Live Camera", font=("Arial", 16, "bold")).pack(pady=(14, 6))
        self.preview_label = ctk.CTkLabel(left, text="Starting camera…", width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview_label.pack(padx=14, pady=8)

        self.status_badge = ctk.CTkLabel(
            left, text="IDLE", font=("Arial", 26, "bold"),
            fg_color=COLOR_IDLE, corner_radius=10, height=64,
        )
        self.status_badge.pack(fill="x", padx=14, pady=(8, 6))

        self.breakdown = ctk.CTkLabel(left, text="face — · iris — · combined —", font=("Arial", 13))
        self.breakdown.pack(pady=(0, 14))

        # ---- Right: controls + log ----
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
        add_btn(btns, "Clear Output", self.clear_output, 2, 1, fg_color="gray30")

        ctk.CTkLabel(right, text="System Output", font=("Arial", 14, "bold")).grid(
            row=6, column=0, padx=16, pady=(8, 2), sticky="w")

        self.output = ctk.CTkTextbox(right, font=("Courier", 12))
        self.output.grid(row=7, column=0, padx=16, pady=(0, 16), sticky="nsew")

    # ------------------------------------------------------------------
    # Live preview loop (main thread)
    # ------------------------------------------------------------------
    def _update_preview(self):
        if self.preview_active and not self.busy:
            try:
                result = self.service.camera.read_frame()
                if result.success and result.frame is not None:
                    rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb).resize(PREVIEW_SIZE)
                    self._preview_image = ctk.CTkImage(light_image=pil, dark_image=pil, size=PREVIEW_SIZE)
                    self.preview_label.configure(image=self._preview_image, text="")
            except Exception:  # camera hiccup — keep the loop alive
                logging.debug("preview frame skipped", exc_info=True)
            self.after(33, self._update_preview)
        else:
            # Paused (an operation owns the camera); poll less often.
            self.after(150, self._update_preview)

    # ------------------------------------------------------------------
    # Worker plumbing
    # ------------------------------------------------------------------
    def write(self, text: str):
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in self.buttons:
            b.configure(state=state)
        if busy:
            # Release the camera so the workflow can take it over cleanly.
            self.preview_active = False
            try:
                self.service.camera.close()
            except Exception:
                pass
        else:
            self.preview_active = True

    def _run(self, fn, *args):
        """Run a blocking service call on a worker thread."""
        if self.busy:
            return
        self._set_busy(True)

        def worker():
            try:
                payload = fn(*args)
                self.ui_queue.put(("result", payload))
            except Exception as exc:  # surface, don't crash the UI
                logging.exception("operation failed")
                self.ui_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _process_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "error":
                    self.write(f"Error: {payload}")
                    self._set_status("✗ ERROR", COLOR_FAIL)
                    self._set_busy(False)
                elif kind == "result":
                    handler, value = payload
                    handler(value)
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(100, self._process_queue)

    def _set_status(self, text: str, color: str):
        self.status_badge.configure(text=text, fg_color=color)

    def _set_breakdown(self, face=None, iris=None, combined=None):
        def fmt(v):
            return f"{v:.0f}%" if isinstance(v, (int, float)) else "—"
        self.breakdown.configure(
            text=f"face {fmt(face)} · iris {fmt(iris)} · combined {fmt(combined)}")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def enroll_user(self):
        uid = self.user_id.get().strip()
        name = self.name.get().strip()
        if not uid or not name:
            self.write("Enter both User ID and Name to enroll.")
            return
        self.write(f"\n===== ENROLLING {uid} =====")
        self._set_status("ENROLLING…", COLOR_BUSY)
        self._run(lambda: (self._on_enroll, self.service.enroll(uid, name)))

    def _on_enroll(self, result):
        self.write(result.message)
        self._set_status("✓ ENROLLED" if result.success else "✗ FAILED",
                         COLOR_OK if result.success else COLOR_FAIL)

    def verify_user(self):
        uid = self.user_id.get().strip()
        if not uid:
            self.write("Enter a User ID to verify.")
            return
        self.write(f"\n===== VERIFYING (face) {uid} =====")
        self._set_status("VERIFYING…", COLOR_BUSY)
        self._run(lambda: (self._on_verify, self.service.verify_face(uid)))

    def _on_verify(self, result):
        self.write(result.message)
        self._set_breakdown(face=result.confidence)
        self._set_status("✓ VERIFIED" if result.success else "✗ NOT VERIFIED",
                         COLOR_OK if result.success else COLOR_FAIL)

    def authenticate_user(self):
        uid = self.user_id.get().strip()
        if not uid:
            self.write("Enter a User ID to authenticate.")
            return
        self.write(f"\n===== MULTI-MODAL AUTHENTICATION ({uid}) =====")
        self._set_status("AUTHENTICATING…", COLOR_BUSY)
        self._run(lambda: (self._on_authenticate, self.service.authenticate(uid)))

    def _on_authenticate(self, result):
        self.write(result.message)
        self.write(
            f"  face_ok={result.face_success} iris_ok={result.iris_success} "
            f"combined={result.combined_confidence:.1f}%")
        self._set_breakdown(
            face=result.face_confidence,
            iris=result.iris_confidence,
            combined=result.combined_confidence,
        )
        # Verdict comes straight from the structured result — no string scraping.
        if result.success:
            self._set_status("✓ AUTHENTICATED", COLOR_OK)
        else:
            self._set_status("✗ DENIED", COLOR_FAIL)

    def list_users(self):
        self.write("\n===== ENROLLED USERS =====")
        self._run(lambda: (self._on_list, self.service.list_users()))

    def _on_list(self, users):
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
        self._run(lambda: (self._on_delete, self.service.delete_user(uid)))

    def _on_delete(self, outcome):
        ok, message = outcome
        self.write(message)

    def clear_output(self):
        self.output.delete("1.0", "end")
        self._set_status("IDLE", COLOR_IDLE)
        self._set_breakdown()

    # ------------------------------------------------------------------
    def _on_close(self):
        self.preview_active = False
        try:
            self.service.camera.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = BiometricApp()
    app.mainloop()
