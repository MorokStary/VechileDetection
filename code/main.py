#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import glob
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# `scipy.ndimage.measurements` is deprecated in newer SciPy versions.
# Import `label` directly from `scipy.ndimage` to avoid warnings.
from scipy.ndimage import label
from joblib import dump, load
try:
    from moviepy.editor import VideoFileClip
except ImportError:  # moviepy>=2.0 moved VideoFileClip to root
    from moviepy import VideoFileClip

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Sliding-window search regions: list of (ystart, ystop, scale)
WINDOW_SIZES = [
    (400, 480, 1.0),
    (400, 528, 1.5),
    (400, 528, 2.0),
]
# How many HOG cells to step per window
CELLS_PER_STEP = 2
# Heatmap threshold: any pixel ≤ this will be zeroed
HEAT_THRESHOLD = 5
# How many past frames to accumulate heat over
HISTORY_LEN = 5
# SVM decision_function margin threshold
DECISION_THRESHOLD = 0.5

# HOG parameters
ORIENT           = 9
PIX_PER_CELL     = 8
CELL_PER_BLOCK   = 2
HOG_CHANNEL      = 'ALL'  # 0,1,2 or "ALL"

# Paths for caching
FEATURES_FILE = 'hog_features.npz'
MODEL_FILE    = 'svc_model.joblib'


# -----------------------------------------------------------------------------
# Feature extraction & model training / loading
# -----------------------------------------------------------------------------
def extract_hog_features(img_paths):
    """
    Given a list of image file paths, compute HOG features for each.
    Returns array of shape (n_images, n_features).
    """
    features = []
    for fname in img_paths:
        img = cv2.imread(fname)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert to chosen color space (we'll do HOG on R/G/B)
        if HOG_CHANNEL == 'ALL':
            hog_feats = []
            for ch in range(3):
                hog_feats.append(hog(
                    img[:,:,ch], 
                    orientations=ORIENT,
                    pixels_per_cell=(PIX_PER_CELL,PIX_PER_CELL),
                    cells_per_block=(CELL_PER_BLOCK,CELL_PER_BLOCK),
                    feature_vector=True
                ))
            features.append(np.hstack(hog_feats))
        else:
            ch = int(HOG_CHANNEL)
            features.append(hog(
                img[:,:,ch],
                orientations=ORIENT,
                pixels_per_cell=(PIX_PER_CELL,PIX_PER_CELL),
                cells_per_block=(CELL_PER_BLOCK,CELL_PER_BLOCK),
                feature_vector=True
            ))
    return np.array(features)


def train_or_load_svm(force_train=False):
    """
    Train a LinearSVC on cached HOG features (or raw images if cache missing),
    or load from disk if available and not forcing retrain.
    Returns (svc, scaler).
    """
    if (not force_train) and os.path.isfile(MODEL_FILE):
        svc, scaler = load(MODEL_FILE)
        print(f"[INFO] Loaded SVM model and scaler from '{MODEL_FILE}'")
        return svc, scaler

    # Need to train
    # 1) Load or compute HOG features for cars and non-cars
    if os.path.isfile(FEATURES_FILE) and (not force_train):
        data = np.load(FEATURES_FILE)
        X_car   = data['car']
        X_nocar = data['nocar']
        print(f"[INFO] Loaded cached HOG features from '{FEATURES_FILE}'")
    else:
        car_images   = glob.glob('./vehicles/**/*.png',   recursive=True)
        nocar_images = glob.glob('./non-vehicles/**/*.png', recursive=True)
        print(f"[INFO] Extracting HOG features from {len(car_images)} cars and {len(nocar_images)} non-cars...")
        X_car   = extract_hog_features(car_images)
        X_nocar = extract_hog_features(nocar_images)
        np.savez(FEATURES_FILE, car=X_car, nocar=X_nocar)
        print(f"[INFO] Saved HOG features to '{FEATURES_FILE}'")

    # 2) Stack and label
    X = np.vstack((X_car, X_nocar)).astype(np.float64)
    y = np.hstack((np.ones(len(X_car)), np.zeros(len(X_nocar))))

    # 3) Scale features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # 4) Train Linear SVM
    svc = LinearSVC(max_iter=5000)
    svc.fit(X_scaled, y)
    dump((svc, scaler), MODEL_FILE)
    print(f"[INFO] Trained LinearSVC and saved to '{MODEL_FILE}'")

    return svc, scaler


# -----------------------------------------------------------------------------
# Sliding-window search & post-processing
# -----------------------------------------------------------------------------
def find_cars(img, svc, scaler):
    """
    Run sliding window + HOG on one image, return list of bounding boxes.
    """
    boxes = []
    img = img.astype(np.float32) / 255.0

    for ystart, ystop, scale in WINDOW_SIZES:
        crop = img[ystart:ystop, :, :]
        if scale != 1.0:
            crop = cv2.resize(crop, 
                (int(crop.shape[1]/scale), int(crop.shape[0]/scale)))

        # compute per-channel HOG once
        if HOG_CHANNEL == 'ALL':
            hog_channels = [hog(
                crop[:,:,ch],
                orientations=ORIENT,
                pixels_per_cell=(PIX_PER_CELL,PIX_PER_CELL),
                cells_per_block=(CELL_PER_BLOCK,CELL_PER_BLOCK),
                feature_vector=False
            ) for ch in range(3)]
        else:
            ch = int(HOG_CHANNEL)
            h = hog(
                crop[:,:,ch],
                orientations=ORIENT,
                pixels_per_cell=(PIX_PER_CELL,PIX_PER_CELL),
                cells_per_block=(CELL_PER_BLOCK,CELL_PER_BLOCK),
                feature_vector=False
            )
            hog_channels = [h]

        # number of blocks
        n_xblocks = (crop.shape[1] // PIX_PER_CELL) - CELL_PER_BLOCK + 1
        n_yblocks = (crop.shape[0] // PIX_PER_CELL) - CELL_PER_BLOCK + 1
        window = 64
        nblocks_per_window = (window // PIX_PER_CELL) - CELL_PER_BLOCK + 1

        steps_x = (n_xblocks - nblocks_per_window) // CELLS_PER_STEP + 1
        steps_y = (n_yblocks - nblocks_per_window) // CELLS_PER_STEP + 1

        for xb in range(steps_x):
            for yb in range(steps_y):
                xpos = xb * CELLS_PER_STEP
                ypos = yb * CELLS_PER_STEP
                # extract HOG patch
                feat = []
                for h in hog_channels:
                    patch = h[ypos:ypos + nblocks_per_window,
                              xpos:xpos + nblocks_per_window].ravel()
                    feat.append(patch)
                feature_vector = np.hstack(feat)
                # scale & decision
                scaled = scaler.transform(feature_vector.reshape(1,-1))
                score  = svc.decision_function(scaled)[0]
                if score > DECISION_THRESHOLD:
                    xleft = xpos * PIX_PER_CELL
                    ytop  = ypos * PIX_PER_CELL
                    xbox  = int(xleft * scale)
                    ytop_draw = int(ytop * scale)
                    win_draw  = int(window * scale)
                    boxes.append(((xbox, ytop_draw + ystart),
                                  (xbox + win_draw, ytop_draw + win_draw + ystart)))
    return boxes


def add_heat(heatmap, bbox_list):
    """Add +1 to all pixels inside each bbox."""
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap


def apply_threshold(heatmap, threshold):
    """Zero out pixels below the threshold."""
    heatmap[heatmap <= threshold] = 0
    return heatmap


def get_labeled_bboxes(labels) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return bounding boxes for labeled regions."""
    boxes = []
    for car_num in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_num).nonzero()
        ys, xs = nonzero[0], nonzero[1]
        boxes.append(((int(np.min(xs)), int(np.min(ys))),
                     (int(np.max(xs)), int(np.max(ys)))))
    return boxes


def draw_labeled_bboxes(img, labels):
    """Draw bounding boxes around labeled regions."""
    for box in get_labeled_bboxes(labels):
        cv2.rectangle(img, box[0], box[1], (0, 0, 255), 6)
    return img


class VehicleTracker:
    """Track vehicles across frames to count unique cars."""

    def __init__(self, max_lost: int = 3, iou_threshold: float = 0.3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.tracks: List[dict] = []
        self.total = 0

    @staticmethod
    def _iou(a: Tuple[Tuple[int, int], Tuple[int, int]],
             b: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        xA = max(a[0][0], b[0][0])
        yA = max(a[0][1], b[0][1])
        xB = min(a[1][0], b[1][0])
        yB = min(a[1][1], b[1][1])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        inter = interW * interH
        if inter == 0:
            return 0.0
        areaA = (a[1][0] - a[0][0]) * (a[1][1] - a[0][1])
        areaB = (b[1][0] - b[0][0]) * (b[1][1] - b[0][1])
        return inter / float(areaA + areaB - inter)

    def update(self, boxes: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
        matched = [False] * len(boxes)
        new_tracks = []

        for track in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for i, box in enumerate(boxes):
                if matched[i]:
                    continue
                iou_val = self._iou(track['box'], box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_iou >= self.iou_threshold:
                track['box'] = boxes[best_idx]
                track['lost'] = 0
                matched[best_idx] = True
                new_tracks.append(track)
            else:
                track['lost'] += 1
                if track['lost'] <= self.max_lost:
                    new_tracks.append(track)

        for i, box in enumerate(boxes):
            if not matched[i]:
                new_tracks.append({'box': box, 'lost': 0})
                self.total += 1

        self.tracks = new_tracks
        return self.tracks


# -----------------------------------------------------------------------------
# Frame-processing callback
# -----------------------------------------------------------------------------
def make_frame_processor(svc, scaler):
    history = deque(maxlen=HISTORY_LEN)
    def process_frame(frame):
        boxes = find_cars(frame, svc, scaler)
        history.append(boxes)

        heat = np.zeros_like(frame[:,:,0]).astype(np.float)
        for b in history:
            add_heat(heat, b)
        apply_threshold(heat, HEAT_THRESHOLD)
        labels = label(heat)
        result = draw_labeled_bboxes(frame.copy(), labels)
        return result
    return process_frame


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
def _parse_windows(values):
    windows = []
    for v in values or []:
        try:
            parts = [float(x) for x in v.split(',')]
            if len(parts) != 3:
                raise ValueError
            windows.append((int(parts[0]), int(parts[1]), float(parts[2])))
        except Exception:
            raise argparse.ArgumentTypeError(
                f"Invalid window specification '{v}', expected ystart,ystop,scale")
    return windows


def main():
    global HEAT_THRESHOLD, HISTORY_LEN, DECISION_THRESHOLD, CELLS_PER_STEP, WINDOW_SIZES
    p = argparse.ArgumentParser(description="Vehicle detection with HOG+SVM")
    p.add_argument("--input",  "-i", required=True, help="Path to input video")
    p.add_argument("--output", "-o", required=True, help="Path to output video")
    p.add_argument("--train",  "-t", action="store_true",
                   help="Force retrain the SVM (ignores cached model)")
    p.add_argument("--heat-threshold", type=int, default=HEAT_THRESHOLD,
                   help="Heatmap threshold (default %(default)s)")
    p.add_argument("--history-len", type=int, default=HISTORY_LEN,
                   help="Frames of heatmap history (default %(default)s)")
    p.add_argument("--decision-threshold", type=float, default=DECISION_THRESHOLD,
                   help="SVM decision threshold (default %(default)s)")
    p.add_argument("--cells-per-step", type=int, default=CELLS_PER_STEP,
                   help="HOG cells to step per window (default %(default)s)")
    p.add_argument("--window", action="append",
                   help="Search window as 'ystart,ystop,scale'. Can be repeated")

    args = p.parse_args()
    HEAT_THRESHOLD = args.heat_threshold
    HISTORY_LEN = args.history_len
    DECISION_THRESHOLD = args.decision_threshold
    CELLS_PER_STEP = args.cells_per_step
    if args.window:
        WINDOW_SIZES = _parse_windows(args.window)

    svc, scaler = train_or_load_svm(force_train=args.train)
    processor = make_frame_processor(svc, scaler)

    clip = VideoFileClip(args.input)
    # use multiple threads for speed
    result = clip.fl_image(processor)
    result.write_videofile(args.output, audio=False, threads=4)


if __name__ == "__main__":
    main()
