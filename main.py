from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error

app = FastAPI(title="Prospex ML API")

# CORS - Allow all origins (GitHub Pages, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(df, target, features):
    """Preprocess dataframe: drop NaN, encode categoricals, scale features"""
    df = df.copy()
    df = df.dropna()

    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def get_classifiers():
    """Return list of classification models"""
    return [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("Neural Network", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
        ("Linear Model", LogisticRegression(max_iter=500, random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVM", SVC(probability=True, random_state=42)),
    ]


def get_regressors():
    """Return list of regression models"""
    return [
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Neural Network", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
        ("Linear Model", LinearRegression()),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ("SVM", SVR()),
    ]


@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Prospex ML API is running"}


@app.post("/columns")
async def get_columns(file: UploadFile = File(...)):
    """Get column names from uploaded CSV"""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        return {"columns": list(df.columns), "rows": len(df)}
    except Exception as e:
        raise HTTPException(400, f"Parse error: {str(e)}")


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target: str = Form(""),
    features: str = Form(""),
    task: str = Form("classification")
):
    """Train ML models on uploaded dataset"""
    # Validate required parameters
    if not target or target.strip() == "":
        raise HTTPException(400, "target is required")

    if not features or features.strip() == "":
        raise HTTPException(400, "features is required")

    # Parse feature list
    feature_list = [f.strip() for f in features.split(",") if f.strip()]

    # Load and parse CSV
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV file: {str(e)}")

    # Validate target column exists
    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' not found in dataset")

    # Validate all feature columns exist
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    # Preprocess data
    try:
        X, y = preprocess(df, target, feature_list)
    except Exception as e:
        raise HTTPException(400, f"Preprocessing failed: {str(e)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select models based on task type
    models = get_classifiers() if task == "classification" else get_regressors()

    results = []

    # Train each model
    for name, model in models:
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            elapsed = round(time.time() - t0, 3)

            if task == "classification":
                avg = "binary" if len(np.unique(y)) == 2 else "macro"
                result = {
                    "name": name,
                    "acc": round(float(accuracy_score(y_test, y_pred)), 4),
                    "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "f1": round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "time": elapsed
                }
            else:
                r2 = float(r2_score(y_test, y_pred))
                result = {
                    "name": name,
                    "acc": round(max(0, r2), 4),
                    "precision": round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "recall": round(float(mean_squared_error(y_test, y_pred)), 4),
                    "f1": round(max(0, r2), 4),
                    "time": elapsed
                }
            results.append(result)

        except Exception as e:
            results.append({
                "name": name,
                "error": str(e),
                "acc": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "time": 0
            })

    # Return results
    return JSONResponse({
        "task": task,
        "rows": len(df),
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
