# MindScope

## Student Depression Dataset Analysis and Explainable XGBoost Inference

**Academic project report**  
H. M. Ahsan Habib Kushal  
Department of Computer Science and Engineering, NITER, University of Dhaka

### Abstract

MindScope integrates a previously trained five-member XGBoost ensemble into an educational research application for the OpenML Student Depression Dataset. The application presents cohort statistics, model-estimated Yes and No label probabilities, and SHAP explanations computed across the full ensemble. The portable model reproduced the saved forecast for a reference profile within an absolute probability difference of 1.26 × 10⁻⁷. Probability-space SHAP also reconstructed the ensemble output across a 200-profile sample, with a maximum residual of 1.43 × 10⁻⁷. The final export achieved 79.15% accuracy, macro-F1 0.7857, and ROC-AUC 0.8714 on the original held-out split. These results describe performance on the recorded label.

### 1. Research objective

The project explores whether a tabular machine-learning demonstration can expose its inputs, decision rule, dataset characteristics, and model explanations in one research application. The classification task uses the recorded `Depression` label as a binary outcome. Users can inspect hypothetical profiles while seeing which features contribute to the estimated Yes probability. The outputs are for education and research demonstrations.

### 2. Dataset and preprocessing

The OpenML v2 file, dataset 46874, contains 27,901 records and 17 columns. Dataset analytics use 27,870 records after filtering to student entries and deduplicating full rows. The model takes 15 features. They cover demographics, academic indicators, work and study hours, financial stress, sleep, dietary habits, and family history. A self-harm-related source column is excluded from inference and public analytics. Each member uses its fitted preprocessing pipeline, with median imputation for numeric inputs, most-frequent imputation for categories, and one-hot encoding.

### 3. Model and inference

The saved metadata identifies five XGBoost classifiers trained with seeds 42 through 46 and 1,912 boosting rounds per member. The serving application applies each member's corresponding preprocessor and evaluates each booster on a CPU. If pₖ is the estimated Yes probability from member k, the ensemble reports p = (p₁ + p₂ + p₃ + p₄ + p₅) / 5. It reports 1 − p for No and applies the exported decision threshold of 0.56 for Yes. This phase used the portable export and performed no new training.

### 4. Explainability

For each member, interventional TreeSHAP decomposes the Yes score into a background-dependent baseline and feature contributions. The application averages the five baselines and the five contribution vectors. Since the ensemble averages probabilities, the average baseline plus contributions should equal the ensemble probability within numerical tolerance. Contributions from one-hot columns are grouped under the corresponding input for the waterfall and global importance ranking.

| Global feature | Mean absolute contribution |
|---|---|
| Academic Pressure | 15.35 percentage points |
| Financial Stress | 11.80 percentage points |
| Age | 7.79 percentage points |

These values describe the model's use of features across 200 sampled profiles. Background composition, correlations, and the fitted model affect the ranking. The ranking provides no evidence that the features cause depression.

### 5. Verification and results

| Check | Observed result |
|---|---|
| OpenML ARFF conversion | All 27,901 × 17 values matched the existing CSV after parsing |
| Portable export integrity | All five booster JSON and fitted preprocessing pairs matched the source ZIP |
| Saved inference reference | 0.27652499 |
| Serving probability for the reference | 0.27652511596679685 |
| Additional parity checks | 10 source-order predictions and 5 final-refit wiring predictions matched the executed notebook within tolerance |
| Local explanation residual for reference | 3.92 × 10⁻⁸ |
| Global explanation worst residual | 1.43 × 10⁻⁷ over 200 profiles |
| Portable model size | 7,229,790 bytes for boosters and preprocessors |
| Frontend compilation | TypeScript and Vite production build passed |

Ten in-process API tests passed, including concurrent prediction, input validation, CORS, and local SHAP. The workspace environment prevented socket-based checks of the preview and verification on remote hosts.

A local Windows CPU run measured 19.95 s to initialize dependencies, the model and summaries, 154 ms median warm inference over 20 runs, 5.47 s for the first explanation, 479 ms median warm explanation over five runs, and 260.9 MiB peak process working set. These figures describe the local benchmark. Render timing and memory remain unmeasured.

#### Held-out evaluation

The executed `student-depression.ipynb` documents a stratified 60/20/20 split with seed 42. The notebook tuned the ensemble on 16,722 training rows and selected 0.56 by macro-F1 on 5,574 validation rows. The final exported ensemble was refit on all 22,296 development rows. We reran inference on the original 5,574 held-out rows without retraining.

| Measure at threshold 0.56 | Notebook train-only ensemble | Exported refit |
|---|---:|---:|
| Accuracy | 0.7910 | 0.7915 |
| Macro-F1 | 0.7851 | 0.7857 |
| ROC-AUC | 0.8705 | 0.8714 |
| PR-AUC (average precision) | 0.8967 | 0.8974 |
| Brier score | 0.1420 | 0.1416 |

The refit produced 1,747 true negatives, 565 false positives, 597 false negatives, and 2,665 true positives. The Performance page gives the corresponding Yes precision, recall, F1, and ten-bin calibration curve. Notebook confidence intervals and subgroup checks describe the earlier train-only ensemble.

### 6. Implementation

FastAPI serves the model from a single set of loaded artifacts. The React and TypeScript frontend generates the form from the model schema, displays uncertainty and limitations, and provides access to derived data. The explanation route checks additivity and limits simultaneous compute. Submitted profiles are neither retained nor included in validation errors. The release definition targets a single CPU worker on Render and a static Vite build on Vercel.

### 7. Limitations and conclusion

The source does not establish a clinical reference standard, and the study does not demonstrate generalization or calibrated personal probabilities. Potential cohort differences and feature correlations limit interpretation. The project demonstrates how to preserve prediction behavior when integrating a trained model, explain every member of a probability ensemble, and expose its evidence and limitations alongside its output.

### References

1. OpenML, *student_depression_dataset*, v2, ID 46874. [Dataset record](https://www.openml.org/search?type=data&sort=runs&id=46874)
2. MindScope portable model bundle, metadata, and saved reference fixture, included in the project source.

