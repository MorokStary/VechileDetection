# Vehicle Detection

This repository provides a Python script that detects vehicles in video
footage using a HOG + SVM pipeline. The code lives in
[`code/main.py`](code/main.py).

## Prerequisites

The script requires Python 3.11 and several Python packages. Install them
with:

```bash
pip install opencv-python-headless scikit-image scikit-learn joblib moviepy
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
`hog_features.npz`, and the trained model with its scaler will be saved to
`svc_model.joblib`. On subsequent runs without `--train`, these cached files
are loaded automatically so training does not have to be repeated.

## Running Vehicle Detection

Once the model is available, you can process a video with:

```bash
python3 code/main.py --input input_video.mp4 --output detected_output.mp4
```

The script reads the input video, detects vehicles frame by frame, draws
bounding boxes around detected cars, and writes the resulting video to the
path specified by `--output`.

## Notes

* Detection parameters such as the sliding window sizes and heatmap
  thresholds can be adjusted at the top of `code/main.py`.
* If you experience missing dependencies, ensure all required Python
  packages are installed. Running `python3 code/main.py -h` should show the
  command line options without errors.
