<img src="frontend/public/assets/mindscope-orbit.png" width="72" alt="MindScope orbit emblem">

# MindScope

MindScope is an interactive machine learning project for exploring the OpenML Student Depression Dataset. Load a hypothetical student profile, change an input such as academic pressure or sleep duration, and see how a five-model XGBoost ensemble changes its estimate. Each prediction comes with a breakdown of the inputs that moved it toward or away from the survey’s recorded **Yes** label.

**[Open the live application](https://mind-scope-tau.vercel.app/#overview)** · [Training notebook](student-depression.ipynb) · [Project report](docs/project_report.md)

## Try a prediction

1. Open [Assessment](https://mind-scope-tau.vercel.app/#assessment), select **Load sample profile**, then **Run model**.
2. Read the estimated label and the Yes and No probabilities. The waterfall shows how each feature moves the estimate from the model’s baseline, in percentage points.
3. Change one input and run the model again. Compare the two estimates, then open **Explainability** to see which features influence predictions across a sample of profiles.

Use hypothetical profiles. MindScope was built for research and education. Its predictions describe patterns in survey labels. Clinical screening, diagnosis, and decisions about individual students fall outside its scope.

## How the model works

| Stage | Implementation |
| --- | --- |
| Dataset | The source contains 27,901 records. The modeling cohort contains 27,870 records and uses 15 demographic, academic, and lifestyle inputs. The self-harm-related survey item is excluded from model inputs and public analytics. |
| Preprocessing | Each ensemble member applies its fitted pipeline. Numeric values use median imputation. Categorical values use most-frequent imputation and one-hot encoding. |
| Prediction | Five XGBoost models trained with seeds 42–46 each estimate the probability of the recorded Yes label. MindScope averages their estimates and assigns Yes at a threshold of 0.56. |
| Explanation | TreeSHAP computes contributions in probability space for each model using a fixed background of 64 profiles. MindScope averages the contributions and groups encoded categories under their original feature names. |

[Original dataset on OpenML](https://www.openml.org/d/46874) · [Technical reference](docs/technical_reference.md)

## Held-out performance

The notebook used a stratified 60/20/20 split with seed 42. Validation macro-F1 determined the decision threshold. The exported ensemble was then refit on all 22,296 training and validation records and evaluated on the original **5,574 held-out records**.

| Accuracy | Macro-F1 | ROC-AUC | Average precision | Yes precision | Yes recall | Brier score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 79.15% | 0.7857 | 0.8714 | 0.8974 | 0.8251 | 0.8170 | 0.1416 |

| Recorded label | Predicted No | Predicted Yes |
| --- | ---: | ---: |
| No | 1,747 | 565 |
| Yes | 597 | 2,665 |

The Performance page includes the calibration curve and separate results for the earlier train-only ensemble. Across ten saved notebook predictions, the serving implementation reproduced the Yes probabilities within a maximum absolute difference of 4.89 × 10⁻⁷. [Evaluation artifact](artifacts/model_metrics.json)

## Reading the explanations

Each dot below represents one sampled profile for the feature on that row. Its horizontal position shows the signed contribution to the predicted Yes probability. A wider horizontal spread means the feature contributed differently across profiles. These contributions describe the fitted model’s behavior and cannot establish what causes depression.

![Global SHAP summary for 200 sampled student profiles](reports/figures/global_shap_summary.png)

## Run locally

Use Python 3.13 and Node.js 22 or newer. From a terminal, clone the repository and create a Python environment.

```sh
git clone https://github.com/AhsanCS64/MindScope.git
cd MindScope
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell, `source .venv/Scripts/activate` in Windows Git Bash, or `source .venv/bin/activate` on Linux and macOS. From the repository root, install the backend dependencies, prepare the dataset, and start FastAPI.

```sh
python -m pip install -r backend/requirements-dev.txt
python backend/prepare_dataset.py
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --no-access-log
```

Dataset preparation needs internet access for the initial download. If you have the source ARFF locally, use `python backend/prepare_dataset.py --arff path/to/source.arff` instead. The fitted model and preprocessing artifacts are included in the repository.

In a second terminal, start from the repository root.

```sh
cd frontend
npm ci
npm run dev
```

Open [the local app](http://localhost:5173). Vite forwards requests under `/api` to port 8000. Check [API health](http://127.0.0.1:8000/api/health) or explore the [interactive API reference](http://127.0.0.1:8000/docs).

## Verify the project

From the activated Python environment, run these commands in `backend/`.

```sh
python validate_model.py
python -m pytest -q
```

The validation script checks prediction parity and scores the fitted ensemble on the reconstructed holdout. To regenerate the global SHAP analysis, run `python generate_global_shap.py`. For inference timings, run `python benchmark.py`. In `frontend/`, run `npm run build` to type-check and build the interface.

## Deployment

The React and TypeScript frontend runs on Vercel. FastAPI runs on Render, loads the fitted artifacts on CPU, and serves predictions, dataset summaries, and explanations. The application does not store submitted profiles.

- **Backend** — Deploy the included blueprint. Its build step installs Python dependencies and prepares the dataset. Render checks readiness at `/api/health`. [Render configuration](render.yaml)
- **Frontend** — Set the Vercel project root to `frontend/`. Its build runs in Vercel mode, and its rewrite forwards same-origin `/api/*` requests to Render. For a separate deployment, update the rewrite destination and the backend’s `CORS_ORIGINS` value to your own service origins. [Vercel configuration](frontend/vercel.json)

## Developer

**H M Ahsan Habib Kushal** · Machine Learning Engineer  
Computer Science and Engineering, NITER, University of Dhaka

[Portfolio](https://ahsanhabibkushal.vercel.app/) · [GitHub](https://github.com/AhsanCS64) · [Hugging Face](https://huggingface.co/Archmage64)
