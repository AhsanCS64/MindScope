# MindScope

MindScope is an educational research application for exploring the OpenML Student Depression Dataset, running an existing five-member XGBoost ensemble, and inspecting its predictions with SHAP. The output represents the model's estimate of a recorded survey label. It is unsuitable for diagnosis, clinical screening, treatment, urgency assessment, or student eligibility decisions.

## Application

- **Overview** introduces the project, cohort, and ensemble.
- **Assessment** runs a hypothetical 15-feature profile and shows both label probabilities, the 0.56 decision threshold, and a local SHAP waterfall.
- **Explainability** shows population-level importance, a SHAP summary, feature dependence plots, and the last sample's waterfall.
- **Dataset** presents distributions, missingness, numeric summaries, and Pearson correlations.
- **Performance** shows the exported ensemble's held-out metrics, confusion matrix, calibration, notebook results before refit, and prediction parity.
- **Methodology** explains the dataset, preprocessing, ensemble, interpretation, and limitations.

## Local setup

Use Python 3.13 and Node.js 22 or newer. Run these commands from the project root in PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r backend/requirements-dev.txt
.\.venv\Scripts\python backend/prepare_dataset.py
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --no-access-log
```

Skip dataset preparation if the workspace already contains `data/raw/student_depression_openml_46874.csv`. Use `--arff dataset_` to prepare from the original verified file without downloading.

In a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). FastAPI publishes [interactive API documentation](http://127.0.0.1:8000/docs) and [a health endpoint](http://127.0.0.1:8000/api/health).

## Verification

From `backend/`, run `..\.venv\Scripts\python validate_model.py`, `..\.venv\Scripts\python -m pytest -q`, and `..\.venv\Scripts\python benchmark.py`. Use `generate_global_shap.py` to regenerate the precomputed global JSON and figures. From `frontend/`, run `npm run build`.

## Documentation

- [System architecture, API, model card, dataset, SHAP, and responsible use](docs/technical_reference.md)
- [Academic project report](docs/project_report.md)
- [Render and Vercel deployment](docs/deployment.md)

Dataset source and provenance are documented by [OpenML dataset 46874](https://www.openml.org/search?type=data&sort=runs&id=46874).

