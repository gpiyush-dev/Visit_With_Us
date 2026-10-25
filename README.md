# Visit with Us — Tourism Purchase Prediction (MLOps)

End-to-end machine-learning workflow to estimate whether a customer is likely to purchase a tourism package (`ProdTaken`). The workflow validates the dataset, creates stratified train/test splits, tunes a Random Forest classifier, records parameters and metrics, serializes the model, and exposes predictions through Streamlit.

## Repository structure

```text
.
├── .github/workflows/pipeline.yml
├── tourism_project/
│   ├── data/tourism.csv
│   ├── model_building/
│   │   ├── data_register.py
│   │   ├── prep.py
│   │   └── train.py
│   └── deployment/
│       ├── app.py
│       └── model.joblib                 # produced by training pipeline
├── reports/metrics.json                 # produced by training pipeline
├── artifacts/                           # temporary train/test split files
├── requirements.txt
└── README.md
```

## Run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
streamlit run tourism_project/deployment/app.py
```

The training script writes `tourism_project/deployment/model.joblib` and `reports/metrics.json`. MLflow records tuned parameters and metrics locally when installed; the JSON report also preserves them for auditability.

## GitHub Actions

Push this repository to a GitHub repository's `main` branch. The workflow runs dataset validation, creates train/test CSVs, transfers them between jobs using a workflow artifact, trains and evaluates the model, then commits updated model and metrics files to `main`. The model/metrics commit uses `[skip ci]` and is excluded from the workflow's path triggers to avoid recursive runs.

Ensure **Settings → Actions → General → Workflow permissions** permits read and write access to repository contents. Never put a personal access token in source code.

## Streamlit Community Cloud deployment

1. Push the repository to GitHub and wait for the workflow to commit `tourism_project/deployment/model.joblib`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, and select this repository and `main` branch.
3. Set **Main file path** to `tourism_project/deployment/app.py`.
4. Streamlit will install dependencies from the root `requirements.txt`.
5. After deployment, set the app to public if required by your assignment.

**Published links to fill after deployment**
- GitHub repository: `ADD_YOUR_PUBLIC_GITHUB_REPOSITORY_URL`
- Streamlit app: `ADD_YOUR_PUBLIC_STREAMLIT_APP_URL`

## Data notes and limitations

- `CustomerID` and the generated `Unnamed: 0` index are excluded from model features.
- The split is stratified and uses a fixed random seed for reproducibility.
- The app uses a 0.50 probability threshold by default. Select a threshold based on business costs and validation results before production use.
- Predictions are estimates, not guarantees. Monitor calibration, recall/precision, drift, fairness, and campaign outcomes over time.
