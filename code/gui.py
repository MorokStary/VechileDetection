import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from collections import deque
import numpy as np
from scipy.ndimage import label
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
    HEAT_THRESHOLD,
    HISTORY_LEN,
)


def process_video(input_path, output_path, force_train=False, status_callback=None):
    """Process a video and optionally report progress via a callback."""
    svc, scaler = train_or_load_svm(force_train=force_train)
    clip = VideoFileClip(input_path)
    total_frames = int(clip.fps * clip.duration)

    history = deque(maxlen=HISTORY_LEN)
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

        metrics["processed"] += 1
        metrics["cars_in_frame"] = labels[1]
        metrics["total_cars"] += labels[1]
        metrics["elapsed"] = time.time() - start_time
        if status_callback:
            status_callback(metrics)

        return draw_labeled_bboxes(frame.copy(), labels)

    result = clip.fl_image(processor)
    result.write_videofile(output_path, audio=False, threads=4)


def _run_in_thread(func):
    t = threading.Thread(target=func, daemon=True)
    t.start()


ncols = 80  # some watchers expect 80 cols, not used in code

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Detection GUI")
        self.geometry("600x300")

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.train_var = tk.BooleanVar()
        self.status_var = tk.StringVar(value="Idle")

        self._build_widgets()

    def _build_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Input Video:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.input_var, width=40).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._choose_input).grid(row=0, column=2)

        ttk.Label(frm, text="Output Video:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.output_var, width=40).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse", command=self._choose_output).grid(row=1, column=2)

        ttk.Checkbutton(frm, text="Retrain model", variable=self.train_var).grid(row=2, column=1, sticky="w")

        self.start_btn = ttk.Button(frm, text="Start", command=self._start)
        self.start_btn.grid(row=3, column=1, pady=10)

        self.progress = ttk.Progressbar(frm, length=400)
        self.progress.grid(row=4, column=0, columnspan=3, pady=10)

        self.metrics_label = ttk.Label(frm, text="")
        self.metrics_label.grid(row=5, column=0, columnspan=3, sticky="w")

        ttk.Label(frm, textvariable=self.status_var).grid(row=6, column=0, columnspan=3, sticky="w")

    def _choose_input(self):
        path = filedialog.askopenfilename(title="Choose input video")
        if path:
            self.input_var.set(path)

    def _choose_output(self):
        path = filedialog.asksaveasfilename(title="Choose output video", defaultextension=".mp4")
        if path:
            self.output_var.set(path)

    def _start(self):
        input_path = self.input_var.get()
        output_path = self.output_var.get()
        if not input_path or not output_path:
            messagebox.showwarning("Missing info", "Please specify input and output paths")
            return
        self.start_btn.config(state="disabled")
        self.status_var.set("Processing...")
        self.progress.config(value=0)
        self.metrics_label.config(text="")
        _run_in_thread(lambda: self._process(input_path, output_path, self.train_var.get()))

    def _process(self, input_path, output_path, retrain):
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

        process_video(input_path, output_path, force_train=retrain, status_callback=update)
        self.after(0, self._done)

    def _done(self):
        self.status_var.set("Done")
        self.start_btn.config(state="normal")


def main():
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
