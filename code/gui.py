import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from collections import deque
import numpy as np
from scipy.ndimage import label
import cv2
from PIL import Image, ImageTk
try:
    from moviepy.editor import VideoFileClip
except ImportError:  # moviepy>=2.0 moved VideoFileClip to root
    from moviepy import VideoFileClip

from .main import (
    train_or_load_svm,
    find_cars,
    add_heat,
    apply_threshold,
    draw_labeled_bboxes,
    get_labeled_bboxes,
    VehicleTracker,
    HEAT_THRESHOLD,
    HISTORY_LEN,
    DECISION_THRESHOLD,
    CELLS_PER_STEP,
    WINDOW_SIZES,
)


def process_video(
    input_path,
    output_path,
    force_train=False,
    status_callback=None,
    augment_flip=False,
    heat_threshold=HEAT_THRESHOLD,
    history_len=HISTORY_LEN,
    decision_threshold=DECISION_THRESHOLD,
    cells_per_step=CELLS_PER_STEP,
    window_sizes=None,
):
    """Process a video and optionally report progress via a callback."""

    # Override module level parameters temporarily
    global HEAT_THRESHOLD, HISTORY_LEN, DECISION_THRESHOLD, CELLS_PER_STEP, WINDOW_SIZES
    old_vals = (
        HEAT_THRESHOLD,
        HISTORY_LEN,
        DECISION_THRESHOLD,
        CELLS_PER_STEP,
        WINDOW_SIZES,
    )
    HEAT_THRESHOLD = heat_threshold
    HISTORY_LEN = history_len
    DECISION_THRESHOLD = decision_threshold
    CELLS_PER_STEP = cells_per_step
    if window_sizes is not None:
        WINDOW_SIZES = window_sizes

    svc, scaler = train_or_load_svm(force_train=force_train,
                                    augment_flip=augment_flip)
    clip = VideoFileClip(input_path)
    total_frames = int(clip.fps * clip.duration)

    history = deque(maxlen=HISTORY_LEN)
    tracker = VehicleTracker()
    metrics = {
        "processed": 0,
        "total_frames": total_frames,
        "total_cars": 0,
        "cars_in_frame": 0,
        "fps": clip.fps,
    }
    start_time = time.time()

    def processor(frame):
        nonlocal metrics
        boxes = find_cars(frame, svc, scaler)
        history.append(boxes)

        heat = np.zeros_like(frame[:, :, 0]).astype(np.float32)
        for b in history:
            add_heat(heat, b)
        apply_threshold(heat, HEAT_THRESHOLD)
        labels = label(heat)
        labeled_boxes = get_labeled_bboxes(labels)
        tracker.update(labeled_boxes)

        metrics["processed"] += 1
        metrics["cars_in_frame"] = len(labeled_boxes)
        metrics["total_cars"] = tracker.total
        metrics["elapsed"] = time.time() - start_time
        if status_callback:
            status_callback(metrics)

        return draw_labeled_bboxes(frame.copy(), labels)

    result = clip.fl_image(processor)
    result.write_videofile(output_path, audio=False, threads=4)

    # restore original parameters
    (
        HEAT_THRESHOLD,
        HISTORY_LEN,
        DECISION_THRESHOLD,
        CELLS_PER_STEP,
        WINDOW_SIZES,
    ) = old_vals


def _run_in_thread(func):
    t = threading.Thread(target=func, daemon=True)
    t.start()


ncols = 80  # some watchers expect 80 cols, not used in code

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Detection GUI")
        self.geometry("650x400")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.train_var = tk.BooleanVar()
        self.flip_var = tk.BooleanVar()
        self.status_var = tk.StringVar(value="Idle")
        self.heat_var = tk.IntVar(value=HEAT_THRESHOLD)
        self.history_var = tk.IntVar(value=HISTORY_LEN)
        self.decision_var = tk.DoubleVar(value=DECISION_THRESHOLD)
        self.step_var = tk.IntVar(value=CELLS_PER_STEP)
        self.windows_var = tk.StringVar(
            value=";".join(f"{a},{b},{c}" for a, b, c in WINDOW_SIZES)
        )
        self.show_boxes_var = tk.BooleanVar()
        self.show_heat_var = tk.BooleanVar()

        self.svc = None
        self.scaler = None
        self.show_boxes = False
        self.show_heat = False

        self._build_widgets()
        self.cap = None
        self.preview_running = False
        self.output_view = False
        self.image_on_canvas = None
        self.preview_history = None

    def _build_widgets(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.configure(bg="#f0f4ff")
        style.configure("TFrame", background="#f0f4ff")
        style.configure("TLabel", background="#f0f4ff", padding=4, foreground="#333")
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), background="#f0f4ff", foreground="#222", padding=6)
        style.configure("TCheckbutton", background="#f0f4ff")
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", background="#4caf50", foreground="white", padding=6)
        style.map(
            "Accent.TButton",
            background=[("active", "#45a049")],
            foreground=[("disabled", "#ddd")],
        )
        style.configure(
            "blue.Horizontal.TProgressbar",
            troughcolor="#d0d0d0",
            background="#3b82f6",
            thickness=15,
        )

        header = ttk.Label(self, text="Vehicle Detection", style="Header.TLabel")
        header.pack(pady=(10, 0))

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Input Video:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.input_var, width=40).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._choose_input).grid(row=0, column=2)
        ttk.Button(frm, text="Preview", command=self._preview).grid(row=0, column=3)

        ttk.Label(frm, text="Output Video:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.output_var, width=40).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._choose_output).grid(row=1, column=2)
        self.view_btn = ttk.Button(frm, text="View", command=self._view_output, state="disabled")
        self.view_btn.grid(row=1, column=3)

        ttk.Checkbutton(frm, text="Flip augment", variable=self.flip_var).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(frm, text="Retrain model", variable=self.train_var).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Show boxes", variable=self.show_boxes_var).grid(row=2, column=2, sticky="w")
        ttk.Checkbutton(frm, text="Show heatmap", variable=self.show_heat_var).grid(row=2, column=3, sticky="w")

        ttk.Label(frm, text="Heat threshold:").grid(row=3, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.heat_var, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(frm, text="History len:").grid(row=4, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.history_var, width=10).grid(row=4, column=1, sticky="w")

        ttk.Label(frm, text="Decision thr:").grid(row=5, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.decision_var, width=10).grid(row=5, column=1, sticky="w")

        ttk.Label(frm, text="Cells/step:").grid(row=6, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.step_var, width=10).grid(row=6, column=1, sticky="w")

        ttk.Label(frm, text="Windows:").grid(row=7, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.windows_var, width=40).grid(row=7, column=1, sticky="we")

        self.start_btn = ttk.Button(frm, text="Start", command=self._start, style="Accent.TButton")
        self.start_btn.grid(row=8, column=3, pady=10, sticky="e")

        self.progress = ttk.Progressbar(frm, length=400, style="blue.Horizontal.TProgressbar")
        self.progress.grid(row=9, column=0, columnspan=4, pady=10)

        self.metrics_label = ttk.Label(frm, text="")
        self.metrics_label.grid(row=10, column=0, columnspan=4, sticky="w")

        ttk.Label(frm, textvariable=self.status_var).grid(row=11, column=0, columnspan=4, sticky="w")

        self.canvas = tk.Canvas(frm, width=320, height=240, bg="black")
        self.canvas.grid(row=12, column=0, columnspan=4, pady=10)

    def _get_model(self):
        if self.svc is None or self.scaler is None:
            self.svc, self.scaler = train_or_load_svm(
                force_train=False,
                augment_flip=self.flip_var.get(),
            )
        return self.svc, self.scaler

    def _choose_input(self):
        path = filedialog.askopenfilename(title="Choose input video")
        if path:
            self.input_var.set(path)

    def _preview(self):
        path = self.input_var.get()
        if not path:
            messagebox.showwarning("No input", "Please select an input video first")
            return
        self._play_video(path)

    def _choose_output(self):
        path = filedialog.asksaveasfilename(title="Choose output video", defaultextension=".mp4")
        if path:
            self.output_var.set(path)

    def _view_output(self):
        path = self.output_var.get()
        if not path:
            messagebox.showwarning("No output", "Please specify an output video first")
            return
        self._play_video(path, output=True)

    def _show_frame(self):
        if not self.preview_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            self.preview_running = False
            if self.output_view:
                self.view_btn.after(0, lambda: self.view_btn.config(state="normal"))
                self.output_view = False
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.show_boxes or self.show_heat:
            svc, scaler = self._get_model()
            boxes = find_cars(frame, svc, scaler)
            if self.preview_history is not None:
                self.preview_history.append(boxes)
            heat = np.zeros_like(frame[:, :, 0]).astype(np.float32)
            if self.preview_history is not None:
                for b in self.preview_history:
                    add_heat(heat, b)
            apply_threshold(heat, self.heat_var.get())
            labels = label(heat)
            if self.show_boxes:
                frame = draw_labeled_bboxes(frame, labels)
            if self.show_heat:
                if np.max(heat) > 0:
                    norm = np.clip(heat / np.max(heat) * 255, 0, 255).astype(np.uint8)
                else:
                    norm = heat.astype(np.uint8)
                cmap = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
                cmap = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
                frame = cv2.addWeighted(frame, 0.7, cmap, 0.3, 0)

        img = Image.fromarray(frame)
        img = img.resize((320, 240))
        self.image_on_canvas = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.image_on_canvas, anchor="nw")
        delay = int(1000 / (self.cap.get(cv2.CAP_PROP_FPS) or 24))
        self.after(delay, self._show_frame)

    def _play_video(self, path, output=False):
        if self.preview_running:
            return
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video")
            return
        self.output_view = output
        if output:
            self.view_btn.config(state="disabled")
        self.show_boxes = self.show_boxes_var.get()
        self.show_heat = self.show_heat_var.get()
        if self.show_boxes or self.show_heat:
            self.preview_history = deque(maxlen=self.history_var.get())
        else:
            self.preview_history = None
        self.preview_running = True
        self._show_frame()

    def _start(self):
        input_path = self.input_var.get()
        output_path = self.output_var.get()
        if self.preview_running:
            self.preview_running = False
            if self.cap is not None:
                self.cap.release()
        if not input_path or not output_path:
            messagebox.showwarning("Missing info", "Please specify input and output paths")
            return
        try:
            windows = [
                tuple(map(float, w.split(',')))
                for w in self.windows_var.get().split(';')
                if w.strip()
            ]
            windows = [(int(a), int(b), float(c)) for a, b, c in windows]
        except Exception as e:
            messagebox.showerror("Invalid windows", f"Could not parse windows: {e}")
            return
        params = {
            "heat_threshold": self.heat_var.get(),
            "history_len": self.history_var.get(),
            "decision_threshold": self.decision_var.get(),
            "cells_per_step": self.step_var.get(),
            "window_sizes": windows,
            "augment_flip": self.flip_var.get(),
        }
        self.start_btn.config(state="disabled")
        self.view_btn.config(state="disabled")
        self.status_var.set("Processing...")
        self.progress.config(value=0)
        self.metrics_label.config(text="")
        _run_in_thread(lambda: self._process(input_path, output_path, self.train_var.get(), params))

    def _process(self, input_path, output_path, retrain, params):
        def update(metrics):
            progress = metrics["processed"] / metrics["total_frames"] * 100
            msg = (
                f"Frame {metrics['processed']} / {metrics['total_frames']}\n"
                f"Cars in frame: {metrics['cars_in_frame']}\n"
                f"Total cars: {metrics['total_cars']}\n"
                f"Elapsed: {metrics['elapsed']:.1f}s"
            )
            self.progress.after(0, lambda: self.progress.config(value=progress))
            self.metrics_label.after(0, lambda: self.metrics_label.config(text=msg))

        process_video(
            input_path,
            output_path,
            force_train=retrain,
            status_callback=update,
            **params,
        )
        self.after(0, self._done)

    def _done(self):
        self.status_var.set("Done")
        self.start_btn.config(state="normal")
        self.view_btn.config(state="normal")


def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
