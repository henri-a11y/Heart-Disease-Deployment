# Heart Disease Deployment

A complete end-to-end Machine Learning project that predicts whether a patient has heart disease, compares multiple algorithms, automatically selects the best model, and deploys a Flask web application on [Render](https://render.com).

## Project Description

This project trains four classification models on the Heart Disease dataset, evaluates them using **Accuracy Score**, saves the best-performing model, and serves predictions through:

- A **modern Bootstrap 5 web UI** with form-based input
- A **REST API** at `POST /predict` for programmatic access

The application is production-ready for deployment on Render using Gunicorn.

## Dataset

**Source:** [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| age       | Age in years                                     |
| sex       | Sex (0 = female, 1 = male)                       |
| cp        | Chest pain type (0–3)                            |
| trestbps  | Resting blood pressure (mm Hg)                   |
| chol      | Serum cholesterol (mg/dl)                        |
| fbs       | Fasting blood sugar > 120 mg/dl (1 = yes)        |
| restecg   | Resting ECG results (0–2)                        |
| thalach   | Maximum heart rate achieved                      |
| exang     | Exercise induced angina (1 = yes)                |
| oldpeak   | ST depression induced by exercise                |
| slope     | Slope of peak exercise ST segment (0–2)          |
| ca        | Number of major vessels (0–3)                    |
| thal      | Thalassemia (1 = normal, 2 = fixed, 3 = reversible)|
| target    | Heart disease (0 = no, 1 = yes)                  |

## Libraries Used

| Library       | Purpose                              |
|---------------|--------------------------------------|
| pandas        | Data loading and manipulation        |
| scikit-learn  | Model training and evaluation        |
| joblib        | Model serialization (`model.pkl`)    |
| Flask         | Web application framework            |
| gunicorn      | Production WSGI server (Render)      |
| numpy         | Numerical operations                 |
| Bootstrap 5   | Responsive frontend UI               |

## Methodology

1. **Data Loading** — Load `heart.csv` and split features/target.
2. **Train/Test Split** — 80/20 stratified split (`random_state=42`).
3. **Model Training** — Train four classifiers on the training set.
4. **Evaluation** — Compare models using **Accuracy Score** on the test set.
5. **Model Selection** — Automatically select the highest-accuracy model.
6. **Serialization** — Save the best model, metadata, and feature list as `model.pkl`.
7. **Deployment** — Flask app loads `model.pkl` at startup and serves predictions.

## Algorithms Compared

| Algorithm           | Description                                      |
|---------------------|--------------------------------------------------|
| Logistic Regression | Linear classifier with StandardScaler pipeline   |
| Decision Tree       | Tree-based classifier                            |
| Random Forest       | Ensemble of 100 decision trees                   |
| SVM                 | Support Vector Machine (RBF kernel) with scaling |

The model with the **highest test accuracy** is automatically selected and saved.

## Selected Model

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 0.8098   |
| Decision Tree       | 0.9854   |
| **Random Forest**   | **1.0000** |
| SVM                 | 0.9268   |

**Random Forest** was automatically selected as the best model (100% test accuracy) and saved as `model.pkl`.

## Project Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask web application
├── train_model.py         # Model training and comparison script
├── model.pkl              # Trained best model (generated)
├── heart.csv              # Dataset
├── requirements.txt       # Python dependencies
├── Procfile               # Render start command
├── runtime.txt            # Python version for Render
├── README.md              # Project documentation
├── templates/
│   └── index.html         # Bootstrap 5 prediction form
└── static/
    └── style.css          # Custom styles
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/henri-a11y/Assignment-10.git
cd Assignment-10
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Run the application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## GitHub Deployment

1. Create a new repository on GitHub named `Assignment-10`.
2. Push the project (include `model.pkl` so Render can deploy without retraining):

```bash
git add .
git commit -m "Initial commit: Heart Disease ML deployment project"
git branch -M main
git remote add origin https://github.com/henri-a11y/Assignment-10.git
git push -u origin main
```

## Render Deployment

1. Go to [https://render.com](https://render.com) and sign in.
2. Click **New +** → **Web Service**.
3. Connect your GitHub account and select the `Assignment-10` repository.
4. Configure the service:

| Setting        | Value                    |
|----------------|--------------------------|
| Name           | heart-disease-predictor  |
| Environment    | Python 3                 |
| Build Command  | `pip install -r requirements.txt` |
| Start Command  | *(auto-detected from Procfile)* |

5. Render reads `runtime.txt` (`python-3.11.9`) and `Procfile` (`web: gunicorn app:app`) automatically.
6. Click **Create Web Service** and wait for deployment.

No code changes are required for Render deployment.

## API Endpoint

### `POST /predict`

Accepts JSON with all 13 patient feature fields and returns a JSON prediction.

**Request:**

```bash
curl -X POST https://YOUR-APP.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
  }'
```

**Response:**

```json
{
  "prediction": 0,
  "has_heart_disease": false,
  "result": "No Heart Disease",
  "probability": 0.2341,
  "model_used": "Random Forest"
}
```

### `GET /health`

Health check endpoint returning service status.

## Screenshots

<!-- Replace with actual screenshots after deployment -->

| Home Page | Prediction Result |
|-----------|-------------------|
| ![Home Page](screenshots/home.png) | ![Prediction](screenshots/prediction.png) |

## Live Demo

**Render URL:** [https://heart-disease-predictor.onrender.com](https://heart-disease-predictor.onrender.com)

*(Replace with your actual Render URL after deployment.)*

## License

This project is for educational purposes.
