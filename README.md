# Geo_Sat_AI (OpenGEO-AI)

Land-use / land-cover classification of satellite imagery using deep learning.

The current work lives in [`notebooks/google_colab.ipynb`](notebooks/google_colab.ipynb).
It uses the [`geoai-py`](https://pypi.org/project/geoai-py/) library to train an image
classifier on the [EuroSAT RGB](https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset)
dataset (64×64 Sentinel-2 image patches across 10 land-cover classes, e.g. Forest,
River, Residential, Industrial, AnnualCrop).

## Goals

What this project aims to do:

1. **Classify satellite image patches by land cover.** Train a CNN on EuroSAT RGB
   that labels a patch as one of 10 classes.
2. **Evaluate it properly.** Report accuracy and per-class metrics, plot training
   curves and a confusion matrix, and show sample predictions.
3. **Make it reproducible.** Move the notebook logic into reusable modules under
   `src/`, with settings in `configs/` and runnable entry points in `scripts/`.
4. **Use it on new imagery.** Run the trained model on new GeoTIFF / RGB tiles and
   export predictions as geospatial layers (GeoPandas / Shapely).
5. **Grow beyond EuroSAT.** Try other datasets (e.g. via Hugging Face `datasets`),
   multispectral bands, and tasks such as segmentation or change detection.

## Status / roadmap

- [x] Repo structure and dependencies
- [x] Colab notebook: install `geoai-py`, import helpers, set up dataset download
- [ ] Fix dataset download (see known issue below)
- [ ] Train the classifier (`train_image_classifier`)
- [ ] Evaluate and plot (`evaluate_classifier`, `plot_training_history`,
      `plot_confusion_matrix`, `plot_predictions`)
- [ ] Save trained weights to `models/` and metrics/plots to `results/`
- [ ] Refactor notebook code into `src/` + `scripts/`
- [ ] Add tests in `tests/`
- [ ] Inference on new imagery with geospatial export

## Notebook workflow

`notebooks/google_colab.ipynb` is written to run in Google Colab:

1. Install `geoai-py` (`%pip install geoai-py`).
2. Import the helpers from `geoai.recognize`: `load_image_dataset`,
   `train_image_classifier`, `predict_images`, `evaluate_classifier`,
   `plot_training_history`, `plot_confusion_matrix`, `plot_predictions`.
3. Download the EuroSAT RGB dataset with `geoai.utils.download_file`.
4. Load the dataset from the `EuroSAT_RGB` folder with `load_image_dataset`.

Training, evaluation and plotting haven't been added yet; the helpers for them are
already imported.

> **Known issue:** the dataset URL in the notebook points at a Kaggle web page
> (`...?select=EuroSAT_RGB`), not a direct file download, so `load_image_dataset`
> fails with `FileNotFoundError`. Download the dataset with the Kaggle API (or
> `kagglehub`) or use a direct archive link, then point `data_dir` at the extracted
> `EuroSAT_RGB` folder (locally, put it under [`data/`](data/)).

## Setup

### Google Colab (recommended)

Open [`notebooks/google_colab.ipynb`](notebooks/google_colab.ipynb) in Colab and run
the cells in order. A GPU runtime (*Runtime → Change runtime type → GPU*) speeds up
training.

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/google_colab.ipynb
```

## Repository structure

```
OpenGEO-AI/
├── configs/        Experiment / model configuration files
├── data/           Datasets (contents git-ignored)
├── docs/           Documentation
├── experiments/    Experiment runs and logs
├── models/         Saved model weights (contents git-ignored)
├── notebooks/      Jupyter notebooks
│   └── google_colab.ipynb   Main Colab notebook (EuroSAT classification)
├── results/        Metrics, plots and predictions (contents git-ignored)
├── scripts/        Runnable entry points (train, evaluate, predict)
├── src/            Python package with reusable code
│   ├── data/         Dataset loading and preprocessing
│   ├── models/       Model definitions
│   ├── training/     Training loops
│   ├── evaluation/   Metrics and plots
│   └── utils/        Shared helpers
├── tests/          Tests
├── requirements.txt
├── self.txt        Running log of changes
└── README.md
```

Empty folders hold a `.gitkeep` file so they are kept in git. Most of them are
placeholders for now.

## Dependencies

Main libraries: PyTorch, torchvision, NumPy, pandas, scikit-learn, OpenCV,
GeoPandas, Shapely, Matplotlib, Hugging Face `datasets` / `huggingface_hub`, and
`geoai-py`. See [`requirements.txt`](requirements.txt).

## Change log

See [`self.txt`](self.txt) for a dated log of what has been done.
