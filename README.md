# Single-cell–resolved laser-programmable control of endosomal escape enables predictable intracellular delivery

Code to reproduce the image analysis in our paper "Single-cell–resolved laser-programmable control of endosomal escape enables predictable intracellular delivery"


The pipeline takes raw brightfield and fluorescence micrographs of a
multi-well plate, isolates each well, segments cells with a U-Net, and
reports the mean fluorescence intensity and cell area per well together with
the laser condition that well received.

```
raw micrographs  ──▶  well crops  ──▶  circular masks   ──▶  cell masks  ──▶  FL_intensity per cell

```

---

## Requirements

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

| package | used for |
|---|---|
| numpy, pandas | arrays, tabular output |
| opencv-python | circle detection, masking, contour area |
| pillow | image cropping |
| torch, torchvision | U-Net training and inference |

Training and inference use a CUDA GPU when one is available and fall back to
CPU otherwise. Inference on CPU is slow but works.

---

## Data

Download the archives from Zenodo (DOI: [DOI]).

### `images/` — experimental data

One directory per sample and exposure state, named `<sample>_before` and
`<sample>_after`:

```
images/
├── withDPP_sample_1_before/
│   ├── BF_original/          raw brightfield micrographs
│   └── BF_split_rename/      laser condition assigned to each well
├── withDPP_sample_1_after/
│   ├── BF_original/          raw brightfield micrographs
│   └── FL_original/          raw fluorescence micrographs
├── withDPP_sample_2_before/
│   ...
└── withoutDPP_sample_3_after/
```

Fluorescence is imaged only after exposure, so `*_before` directories hold
brightfield only.

`BF_split_rename/` contains one empty-named file per **exposed** well, whose
filename carries the laser condition:

```
BF_split_<region>_<position_x>_<position_y>_<position_x_sub>_<position_y_sub>_<power>_<duration>.jpg
                                       									       │        └── exposure time, s
                                       								           └── laser power, % of full scale
```

Laser conditions are assigned once per experiment, before exposure, so the
`_after` steps read this directory from the paired `_before` sample. Wells
absent from it were never exposed and are treated as unexposed controls.


### `training_images/` — U-Net training set

```
training_images/
├── BF_circle/     well images
└── mask_circle/   hand-drawn binary cell masks (white = cell)
```

Images and masks are paired by identical filename.

---

## Running the pipeline

Each script takes `--root images` and finds every sample underneath, so no
paths need editing. Run them in order.

### 1. Crop wells out of the raw micrographs

```bash
python well_crop.py --root images
```

Writes `<sample>_<state>/BF_split/` and `FL_split/` alongside each
`*_original/` directory. Wells sit at fixed pixel coordinates in the field;
see `WELL_LAYOUTS` in the script.

### 2. Mask each well to its circular boundary and attach its condition

```bash
python circle_crop_and_condition_match.py --root images
```

Writes `<sample>_<state>/BF_circle/`. The well wall is located with a Hough
circle transform, everything outside it is set to zero, and the laser
condition is appended to the filename. The run reports how many wells were
exposed, how many were controls, and how many produced no circle.

### 3. Train the U-Net *(optional — a trained checkpoint is included)*

```bash
python unet_train.py --images training_images/BF_circle \
                     --masks  training_images/mask_circle \
                     --output cell_unet_model.pth
```

Skip this step to use `cell_unet_model.pth`, the checkpoint used in the
paper. Download it from Zenodo (DOI: [DOI]) and place it in the repository root. Retraining will not reproduce it bit-for-bit.

### 4. Predict cell masks

```bash
python unet_pred.py --root images --model cell_unet_model.pth
```

Writes `<sample>_<state>/result_mask/`. Predictions are thresholded at 0.5
and connected components smaller than `--min-area` pixels are discarded.

### 5. Measure fluorescence and cell area

```bash
python FL_intensity.py --root images --output fl_intensity.csv
```

Produces one CSV covering every sample:

| column | meaning |
|---|---|
| `sample_name` | e.g. `withDPP_sample_1` |
| `filename` | fluorescence image the row was measured from |
| `power` | laser power, % of full scale (0 = unexposed control) |
| `duration` | exposure time, s |
| `after_fl_intensity` | mean green-channel value over the post-exposure cell mask |
| `after_area` | post-exposure cell area, px |
| `before_area` | pre-exposure cell area, px |


