from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io, time, traceback

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

app = FastAPI(title="ML Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HELPERS ────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target: str, features: list):
    df = df.copy()
    df = df.dropna()

    # Encode categorical columns
    encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    X = df[features].values
    y = df[target].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def get_classifiers():
    return [
        ("Random Forest",       RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting",   GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("Neural Network",      MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
        ("Linear Model",        LogisticRegression(max_iter=500, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Support Vector",      SVC(probability=True, random_state=42)),
    ]


def get_regressors():
    return [
        ("Random Forest",       RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Gradient Boosting",   GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("Neural Network",      MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)),
        ("Linear Model",        LinearRegression()),
        ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=5)),
        ("Support Vector",      SVR()),
    ]


def feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        scores = np.abs(coef[0] if coef.ndim > 1 else coef)
        scores = scores / scores.sum() if scores.sum() > 0 else scores
    else:
        scores = np.ones(len(feature_names)) / len(feature_names)

    return {name: round(float(s), 4) for name, s in zip(feature_names, scores)}


def correlation_matrix(df, features):
    sub = df[features].select_dtypes(include=[np.number])
    corr = sub.corr().round(3)
    return {
        "columns": list(corr.columns),
        "matrix": corr.values.tolist()
    }


# ── ROUTES ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "ML Model API is running"}


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target: str = "",
    features: str = "",
    task: str = "classification"
):
    if not target:
        raise HTTPException(400, "target is required")
    if not features:
        raise HTTPException(400, "features is required")

    feature_list = [f.strip() for f in features.split(",") if f.strip()]

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not parse CSV file")

    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' not found in CSV")

    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

    try:
        X, y = preprocess(df, target, feature_list)
    except Exception as e:
        raise HTTPException(400, f"Preprocessing failed: {str(e)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_classifiers() if task == "classification" else get_regressors()
    results = []

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
                    "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
                    "precision": round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "recall":    round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "f1":        round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)), 4),
                    "time":      elapsed,
                    "importance": feature_importance(model, feature_list)
                }
            else:
                r2 = float(r2_score(y_test, y_pred))
                result = {
                    "name": name,
                    "accuracy":  round(max(0, r2), 4),  # R² as accuracy proxy
                    "precision": round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "recall":    round(float(mean_squared_error(y_test, y_pred)), 4),
                    "f1":        round(max(0, r2), 4),
                    "time":      elapsed,
                    "importance": feature_importance(model, feature_list)
                }

            results.append(result)

        except Exception as e:
            results.append({
                "name": name,
                "error": str(e),
                "accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "time": 0,
                "importance": {}
            })

    corr = {}
    try:
        numeric_df = df[feature_list].copy()
        for col in numeric_df.columns:
            if numeric_df[col].dtype == object:
                numeric_df[col] = LabelEncoder().fit_transform(numeric_df[col].astype(str))
        corr = correlation_matrix(numeric_df, feature_list)
    except Exception:
        pass

    return JSONResponse({
        "task": task,
        "rows": len(df),
        "columns": list(df.columns),
        "results": results,
        "correlation": corr
    })


@app.get("/columns")
async def get_columns(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    return {"columns": list(df.columns), "rows": len(df)}
