# 🚀 Hello MLOps: End-to-End Machine Learning Lifecycle

This project demonstrates a complete MLOps pipeline, transforming a simple "Titanic Survival Prediction" problem into a production-ready system. It covers the full lifecycle from data versioning to automated deployment.

## 🎯 Learning Objectives

We implemented the "Ops" in MLOps by building a pipeline that bridges the gap between **Experimentation** and **Production**.

| Stage | Goal | Tool Used | Key Learning |
| :--- | :--- | :--- | :--- |
| **1. Data Versioning** | Track large datasets without bloating Git. | **DVC** | Git is for code, DVC is for data. DVC pointers replace heavy files. |
| **2. Experimentation** | Track model performance and parameters. | **MLflow** | Never rely on `print()` for accuracy. Log experiments to compare runs systematically. |
| **3. Packaging** | Create a consistent runtime environment. | **Docker** | "It works on my machine" is solved by containerizing the model + API. |
| **4. Serving** | Expose the model as a real-time HTTP service. | **FastAPI** | Models need an interface (API) to be useful to applications. |
| **5. CI/CD** | Automate testing and deployment. | **GitHub Actions** | Automated pipelines ensure only high-quality models (Acc > 75%) reach production. |

---

## 🛠️ Project Structure

```bash
├── .github/workflows/mlops.yml  # CI/CD Pipeline configuration
├── .dvc/                        # DVC configuration
├── data/                        # Dataset (managed by DVC, tracked in Git for this demo)
├── model/                       # Trained model artifacts (generated)
├── src/
│   ├── preprocess.py           # Cleans raw data
│   ├── train.py                # Trains Random Forest & logs to MLflow
│   └── app.py                  # FastAPI server for predictions
├── Dockerfile                   # Blueprint for the application container
└── requirements.txt             # Python dependencies
```

---

## 🚦 How It Works

### 1. The Build Phase (Local Development)
Data scientists work locally to train and improve the model.
```bash
# Data Version Control
dvc init
dvc add data/Titanic-Dataset.csv

# Train & Track
python src/train.py
# Logs metrics (accuracy) and parameters (n_estimators) to MLflow
```

### 2. The Ops Phase (Automated CI/CD)
When code is pushed to the `main` branch, the **GitHub Actions Pipeline** triggers:

1.  **Train-and-Evaluate Job**:
    *   Sets up the environment.
    *   Runs training (`src/train.py`).
    *   **Quality Gate**: Checks if `accuracy > 75%`. If fails, the pipeline stops.
    *   Artifacts the trained model.
2.  **Build-and-Push Job**:
    *   Downloads the trained model.
    *   Builds a Docker image.
    *   Pushes the image to **GitHub Container Registry (GHCR)**.

---

## 🏃‍♂️ How to Run

### Prerequisite
*   Docker installed.

### Run the Docker Container
You can pull the latest production-ready model directly from the registry:

```bash
docker pull ghcr.io/<your-username>/hello_ml:latest
docker run -p 8000:8000 ghcr.io/<your-username>/hello_ml:latest
```

### Test the API
Send a prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "Pclass": 3,
           "Sex": "male",
           "Age": 22.0,
           "SibSp": 1,
           "Parch": 0,
           "Fare": 7.25,
           "Embarked": "S"
         }'
```

---

## 🐛 Troubleshooting & Fixes (Battle Scars)
During this project, we encountered and solved several real-world issues:

*   **MLflow Permissions**: GitHub Runners crashed when MLflow tried to write to default local paths. *Fix*: Explicitly set `mlflow.set_experiment("Name")` in the code.
*   **Docker Tag Casing**: Docker requires lowercase tags, but GitHub usernames can be mixed-case. *Fix*: Used shell commands (`tr`) in CI/CD to lowercase variables.
*   **CI/CD Artifacts**: Build jobs don't share files with Training jobs automatically. *Fix*: Used `upload-artifact` and `download-artifact` steps to pass the model.
