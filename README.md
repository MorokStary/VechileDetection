# Vehicle Detection

This repository provides a Python script that detects vehicles in video
footage using a HOG + SVM pipeline. The code lives in
[`code/main.py`](code/main.py).

## Prerequisites

The script requires Python 3.11 and several Python packages. Install them
with:

```bash
pip install opencv-python-headless scikit-image scikit-learn joblib moviepy pillow
```

(The commands above work for recent versions of `pip`.)

## Training a Model

The SVM classifier is trained on image datasets stored in the directories
`vehicles/` and `non-vehicles/`. Each directory should contain PNG images of
cars and non-car objects respectively. To train a model from scratch (or
retrain regardless of any saved model), run:

```bash
python3 code/main.py --input <video file> --output <output video> --train
```

During training, HOG features will be computed and cached in
`hog_features.npz` (compressed) using 32‑bit floats to keep memory usage
reasonable. The trained model with its scaler will be saved to
`svc_model.joblib`. On subsequent runs without `--train`, these cached files
are loaded automatically so training does not have to be repeated.
You can optionally augment the training data by horizontally flipping all
images using `--augment-flip`. When enabled, the cached files are stored as
`hog_features_flip.npz` and `svc_model_flip.joblib`.

Feature extraction runs in parallel and will use all available CPU cores.

## Running Vehicle Detection

Once the model is available, you can process a video with:

```bash
python3 code/main.py --input input_video.mp4 --output detected_output.mp4
```

The script reads the input video, detects vehicles frame by frame, draws
bounding boxes around detected cars, and writes the resulting video to the
path specified by `--output`. Additional options allow tweaking the detection
parameters, for example:

```bash
python3 code/main.py -i input.mp4 -o out.mp4 --heat-threshold 3 \
    --history-len 8 --decision-threshold 0.2 \
    --window 400,500,1.0 --window 400,550,1.5
```

## Tkinter GUI

For a simple graphical interface built with Tkinter, run:

```bash
python3 -m code.gui
```

The GUI lets you choose an input video, specify an output path and optional
retraining of the SVM. You can also enable *Flip augment* to train a model on
horizontally flipped images. A **Preview** button allows watching the selected source
video directly inside the window. Two checkboxes let you overlay bounding boxes
and the heatmap during preview. After processing finishes a **View** button
becomes active so you can immediately watch the generated output. The controls
have been styled with softer colors, a bold title and a green accent for action
buttons. The progress bar uses a blue accent for easier tracking. During
processing the interface displays progress information such as the current frame
number and how many vehicles are detected.

## Notes

* Detection parameters such as the sliding window sizes and heatmap
  thresholds can now be changed via command line options or directly in the
  GUI fields.
* If you experience missing dependencies, ensure all required Python
  packages are installed. Running `python3 code/main.py -h` should show the
  command line options without errors.
* The GUI counts unique vehicles using a simple tracker so the "Total cars"
  metric reflects how many different cars appeared rather than detections per
  frame.
* You can adjust how long a detection must persist to be counted and how long a
  car may be missing before starting a new track via the **Confirm time** and
  **Lost time** fields in the GUI.
