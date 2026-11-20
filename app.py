"""Desktop launcher. Place beside main.py, compare.py, video.py and mark_spaces.py."""
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


PROJECT = Path(__file__).resolve().parent


class ParkingApp:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.stopping = False
        root.title("Parking Detector")
        root.geometry("900x660")
        root.minsize(740, 570)
        root.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(root, padding=20)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(11, weight=1)
        ttk.Label(body, text="Parking Detector", font=("Segoe UI", 21, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(body, text="MobileNetV2 + slot context • Choose files, then run a demo.").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 18))
        self.image = tk.StringVar()
        self.video = tk.StringVar()
        self.csv = tk.StringVar()
        self.context = tk.StringVar(value=str(PROJECT / "models" / "slot_context.pth"))
        self.file_row(body, 2, "Image", self.image, [("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*")])
        self.file_row(body, 3, "Video", self.video, [("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*")])
        self.file_row(body, 4, "Parking spaces CSV", self.csv, [("CSV", "*.csv")])
        self.file_row(body, 5, "Context checkpoint", self.context, [("PyTorch checkpoint", "*.pth")])
        options = ttk.Frame(body)
        options.grid(row=6, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Label(options, text="Video prediction interval (frames)").pack(side="left")
        self.every = tk.StringVar(value="5")
        ttk.Spinbox(options, from_=1, to=120, width=5, textvariable=self.every).pack(side="left", padx=10)
        ttk.Label(body, text="Use a CSV marked for this exact camera view. For a new lot, mark a reference image first.",
                  wraplength=800).grid(row=7, column=0, columnspan=3, sticky="w", pady=(0, 12))
        actions = ttk.Frame(body)
        actions.grid(row=8, column=0, columnspan=3, sticky="w")
        self.buttons = []
        for title, action in (("Detect image", "detect"), ("Compare models", "compare"),
                              ("Play video", "video"), ("Mark spaces", "mark")):
            button = ttk.Button(actions, text=title, command=lambda a=action: self.start(a))
            button.pack(side="left", padx=(0, 8))
            self.buttons.append(button)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(body, textvariable=self.status).grid(row=9, column=0, columnspan=3, sticky="w", pady=10)
        ttk.Label(body, text="Results open in a separate window. Marking controls: four corners per space, U undo, S save, Esc cancel.",
                  wraplength=800).grid(row=10, column=0, columnspan=3, sticky="w", pady=(0, 8))
        self.log = scrolledtext.ScrolledText(body, height=12, state="disabled", wrap="word", font=("Consolas", 10))
        self.log.grid(row=11, column=0, columnspan=3, sticky="nsew")
        self.write("Image outputs are saved by your scripts in the project's output folder.\n"
                   "The launcher uses the Python environment that started this app.\n")
        root.after(100, self.poll)

    def file_row(self, parent, row, title, variable, types):
        ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=6)
        def browse():
            path = filedialog.askopenfilename(parent=self.root, title=f"Choose {title.lower()}",
                                              initialdir=PROJECT, filetypes=types)
            if path:
                variable.set(path)
        ttk.Button(parent, text="Browse…", command=browse).grid(row=row, column=2, padx=(8, 0))

    def checked_path(self, variable, label):
        value = variable.get().strip()
        if not value:
            raise ValueError(f"Choose {label} first.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = PROJECT / path
        if not path.is_file():
            raise ValueError(f"File does not exist\n{path}")
        return path.resolve()

    def command(self, action, marked_output=None):
        # Child processes keep OpenCV's window loop away from Tkinter's event loop.
        scripts = {"detect": "main.py", "compare": "compare.py", "video": "video.py", "mark": "mark_spaces.py"}
        script = PROJECT / scripts[action]
        if not script.is_file():
            raise ValueError(f"Place app.py beside {script.name}. Missing\n{script}")
        source = self.checked_path(self.video if action == "video" else self.image,
                                   "a video" if action == "video" else "an image")
        command = [sys.executable, "-u", str(script), "--video" if action == "video" else "--image", str(source)]
        if action == "mark":
            return command + ["--output", str(marked_output)]
        command += ["--camera-csv", str(self.checked_path(self.csv, "the parking spaces CSV")),
                    "--context", str(self.checked_path(self.context, "the context checkpoint"))]
        if action == "video":
            try:
                every = int(self.every.get())
            except ValueError:
                raise ValueError("Video prediction interval must be a positive integer.") from None
            if every < 1:
                raise ValueError("Video prediction interval must be a positive integer.")
            command += ["--every", str(every)]
        return command

    def start(self, action):
        if self.process is not None:
            return
        marked_output = None
        try:
            if action == "mark":
                self.checked_path(self.image, "a reference image")
                selected = filedialog.asksaveasfilename(parent=self.root, title="Save new parking spaces CSV",
                    initialdir=PROJECT, defaultextension=".csv", filetypes=[("CSV", "*.csv")])
                if not selected:
                    return
                marked_output = Path(selected)
                if marked_output.exists():
                    raise ValueError("The marking tool needs a new CSV filename. Choose a file that does not exist yet.")
            command = self.command(action, marked_output)
            self.process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        except (OSError, ValueError) as error:
            messagebox.showerror("Unable to start", str(error), parent=self.root)
            return
        self.stopping = False
        self.status.set(f"Running {action} — loading models may take a moment")
        self.write("\n" + subprocess.list2cmdline(command) + "\n")
        for button in self.buttons:
            button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        threading.Thread(target=self.read_output, args=(self.process, action, marked_output), daemon=True).start()

    def read_output(self, process, action, marked_output):
        # Only the main Tk thread updates widgets; the worker sends queue events.
        try:
            with process.stdout:
                for line in process.stdout:
                    self.events.put(("log", line))
        finally:
            self.events.put(("done", (process.wait(), action, marked_output)))

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.write(payload)
                    continue
                code, action, output = payload
                self.process = None
                for button in self.buttons:
                    button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if self.stopping:
                    self.status.set("Stopped")
                elif code:
                    self.status.set("Run failed — see the log below")
                    messagebox.showerror("Run failed", "The script reported an error. See the log for details.", parent=self.root)
                elif action == "mark":
                    # The marker also exits successfully when canceled.
                    if output.is_file():
                        self.csv.set(str(output))
                        self.status.set("Spaces saved — CSV selected for the next run")
                    else:
                        self.status.set("Marking canceled — no CSV saved")
                else:
                    self.status.set("Finished")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.stopping = True
            self.status.set("Stopping…")
            self.process.terminate()
            process = self.process
            self.root.after(2000, lambda: process.kill() if process.poll() is None else None)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Exit", "Stop the running demo and close?", parent=self.root):
                return
            self.process.kill()
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    ParkingApp(window)
    window.mainloop()
