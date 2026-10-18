# BrainScan AI

Classify brain MRI images as glioma, meningioma, no tumor, or pituitary tumor using the CNN from the implementation guide.

## Structure

```text
data/raw/                    # One folder per class
notebooks/brainscan_cnn.ipynb # Full guide, step by step
notebooks/eda.ipynb           # Original EDA notebook
src/config.py                # Paths, image size, batch size, seed
src/data_loader.py           # Clean, load, resize, shuffle
src/model.py                 # build_model()
src/train.py                 # Explore, split, normalize, train
src/evaluate.py              # Metrics, curves, confusion matrix, examples
src/predict.py               # predict_image(path)
models/                      # best_model.h5 and classes.json after training
reports/figures/             # Saved plots
app/streamlit_app.py          # MRI upload and prediction
```

## Setup and run

Use Python 3.12. From the project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the Graphviz system package for `plot_model`. Put images directly in `data/raw/glioma/`, `data/raw/meningioma/`, `data/raw/notumor/`, and `data/raw/pituitary/`.

```bash
python -m src.train
streamlit run app/streamlit_app.py
```

Training uses 224×224 RGB images, an 80/20 train/test split, training class weights, normalization by 255, and the guide's four convolution blocks. It runs for up to 30 epochs with batch size 32, `validation_split=0.1`, checkpointing, early stopping, and learning-rate reduction. Evaluation runs after training and saves plots and `reports/classification_report.txt`.

Alternatively, run `notebooks/brainscan_cnn.ipynb` from top to bottom. It contains the guide's code with Markdown explanations. Back up the dataset before calling `clean_dataset`, which deletes unsupported files.

```python
from src.predict import predict_image
label, confidence = predict_image("path/to/mri.jpg")
```

## Results and limitations

Full training has not been run. Report the actual test accuracy and per-class recall after training; do not infer results from the implementation. The default dataset needs several GB of RAM, and CPU training can take hours.

This is educational decision support, not a medical diagnosis. Image-level splits can leak information when slices from the same patient or duplicate images occur in different partitions. Review per-class recall and missed tumors, not accuracy alone. Fixed seeds support reproducibility.

## Jira planning

Use one-week sprints and the guide's six epics:

| Epic | Tickets | Acceptance criteria |
|---|---|---|
| Setup & Planning | Repository, environment, Jira board | Dependencies installed; board created |
| Data Pipeline | Cleaning, loading, EDA, balancing, splitting | RGB images load; labels align; only training data is balanced |
| Model Design | CNN, compilation, architecture plot | Architecture matches the guide; plot saved |
| Training & Evaluation | Callbacks, curves, metrics, error analysis | Checkpoint and plots saved; test accuracy and recall reported |
| Deployment | Prediction, Streamlit | MRI upload displays prediction and class probabilities |
| Documentation | Notebook explanations, README, final review | Setup and limitations documented |

This table is a plan; no external Jira board has been created.
