
import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import threading

from src.database.storage import DatabaseManager


class BiometricApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal Biometric Authentication System")
        self.root.geometry("900x700")

        self.build_ui()

        self.result_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 28, "bold")
        )
        self.result_label.pack(pady=10)

    def build_ui(self):

        title = tk.Label(
            self.root,
            text="Multi-Modal Biometric Authentication System",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=10)

        tk.Label(self.root, text="User ID").pack()

        self.user_id = tk.Entry(self.root, width=40)
        self.user_id.pack(pady=5)

        tk.Label(self.root, text="Name").pack()

        self.name = tk.Entry(self.root, width=40)
        self.name.pack(pady=5)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=15)

        tk.Button(
            button_frame,
            text="Enroll User",
            width=18,
            command=self.enroll_user
        ).grid(row=0, column=0, padx=5, pady=5)

        tk.Button(
            button_frame,
            text="Verify User",
            width=18,
            command=self.verify_user
        ).grid(row=0, column=1, padx=5, pady=5)

        tk.Button(
            button_frame,
            text="Authenticate",
            width=18,
            command=self.authenticate_user
        ).grid(row=1, column=0, padx=5, pady=5)

        tk.Button(
            button_frame,
            text="List Users",
            width=18,
            command=self.list_users
        ).grid(row=1, column=1, padx=5, pady=5)

        tk.Button(
            button_frame,
            text="Delete User",
            width=18,
            command=self.delete_user
        ).grid(row=2, column=0, padx=5, pady=5)

        tk.Button(
            button_frame,
            text="Clear Output",
            width=18,
            command=self.clear_output
        ).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(
            self.root,
            text="System Output"
        ).pack()

        self.output = scrolledtext.ScrolledText(
            self.root,
            width=100,
            height=20
        )
        self.output.pack(
            padx=10,
            pady=10
        )

    def write(self, text):
        self.output.insert(
            tk.END,
            text + "\n"
        )
        self.output.see(tk.END)

    def run_command(self, command):

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            if result.stdout:
                self.write(result.stdout)

            if result.stderr:
                self.write(result.stderr)

        except Exception as e:
            self.write(f"Error: {e}")

    def run_authentication(self, command):

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )

            output = result.stdout + result.stderr

            self.write(output)

            if (
                    "Authentication Successful" in output
                    or "success=np.True_" in output
                    or "face_success=True" in output
            ):
                self.result_label.config(
                    text="✓ AUTHENTICATED USER",
                    fg="green"
                )
            else:
                self.result_label.config(
                    text="✗ NOT AUTHENTICATED",
                    fg="red"
                )

        except Exception as e:

            self.write(f"Error: {e}")

            self.result_label.config(
                text="✗ ERROR",
                fg="red"
            )

    def enroll_user(self):

        user_id = self.user_id.get().strip()
        name = self.name.get().strip()

        if not user_id or not name:
            messagebox.showerror(
                "Error",
                "Enter User ID and Name"
            )
            return

        self.write(
            f"\n===== ENROLLING {user_id} =====\n"
        )

        command = [
            "python",
            "demo/face_demo.py",
            "enroll",
            user_id,
            name,
            "-s",
            "3"
        ]

        threading.Thread(
            target=self.run_command,
            args=(command,),
            daemon=True
        ).start()

    def verify_user(self):

        user_id = self.user_id.get().strip()

        if not user_id:
            messagebox.showerror(
                "Error",
                "Enter User ID"
            )
            return

        self.write(
            f"\n===== VERIFYING {user_id} =====\n"
        )

        command = [
            "python",
            "demo/face_demo.py",
            "verify",
            user_id
        ]

        threading.Thread(
            target=self.run_command,
            args=(command,),
            daemon=True
        ).start()

    def authenticate_user(self):

        user_id = self.user_id.get().strip()

        if not user_id:
            messagebox.showerror(
                "Error",
                "Enter User ID"
            )
            return

        self.result_label.config(
            text="AUTHENTICATING...",
            fg="blue"
        )

        self.write(
            f"\n===== MULTI-MODAL AUTHENTICATION ({user_id}) =====\n"
        )

        command = [
            "python",
            "test_fusionauth.py",
            user_id
        ]

        threading.Thread(
            target=self.run_authentication,
            args=(command,),
            daemon=True
        ).start()

    def list_users(self):

        try:

            db = DatabaseManager()
            db.initialize()

            users = db.list_users(
                active_only=False
            )

            self.write(
                "\n===== ENROLLED USERS ====="
            )

            if not users:
                self.write(
                    "No users found"
                )
                return

            for user in users:
                self.write(
                    f"{user.user_id} -> {user.name}"
                )

        except Exception as e:
            self.write(
                f"Error: {e}"
            )

    def delete_user(self):

        user_id = self.user_id.get().strip()

        if not user_id:
            messagebox.showerror(
                "Error",
                "Enter User ID"
            )
            return

        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete user '{user_id}'?"
        )

        if not confirm:
            return

        try:

            db = DatabaseManager()
            db.initialize()

            try:
                db.delete_face_template(
                    user_id
                )
            except:
                pass

            try:
                db.delete_iris_template(
                    user_id
                )
            except:
                pass

            try:
                db.delete_user(
                    user_id
                )
            except:
                pass

            self.write(
                f"Deleted user: {user_id}"
            )

        except Exception as e:
            self.write(
                f"Error: {e}"
            )

    def clear_output(self):

        self.output.delete(
            "1.0",
            tk.END
        )

        self.result_label.config(
            text=""
        )


if __name__ == "__main__":

    root = tk.Tk()

    app = BiometricApp(root)

    root.mainloop()
