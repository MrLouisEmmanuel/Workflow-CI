"""
modelling.py - MLflow Project Version
=======================================
Script untuk melatih model RandomForestRegressor pada dataset California Housing.
Versi ini disesuaikan untuk dijalankan melalui 'mlflow run' sebagai MLflow Project.
Menggunakan MLflow autolog untuk pencatatan otomatis.

Author: Louis-Hutapea (loemanohan)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ============================================================
# 1. Konfigurasi MLflow
# ============================================================
# Untuk MLflow Project, tracking URI bisa otomatis dari environment
# atau kita set secara eksplisit ke folder lokal
# mlflow.set_tracking_uri("mlruns/")
# mlflow.set_experiment("california-housing-experiment")

# ============================================================
# 2. Load Dataset
# ============================================================
# Coba load dari file preprocessed yang ada di dalam MLProject
preprocessed_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "california_housing_preprocessing", "housing_preprocessed.csv"
)

if os.path.exists(preprocessed_path):
    print(f"[INFO] Memuat dataset dari: {preprocessed_path}")
    df = pd.read_csv(preprocessed_path)
    if 'MedHouseVal' in df.columns:
        X = df.drop(columns=['MedHouseVal'])
        y = df['MedHouseVal']
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
else:
    # Jika file tidak ditemukan, load langsung dari sklearn
    print("[INFO] File preprocessed tidak ditemukan. Memuat dataset dari sklearn...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = housing.data
    y = housing.target
    
    # Preprocessing sederhana - handle outlier
    print("[INFO] Melakukan preprocessing sederhana...")
    for col in X.columns:
        q1 = X[col].quantile(0.01)
        q99 = X[col].quantile(0.99)
        X[col] = X[col].clip(q1, q99)

print(f"[INFO] Ukuran dataset: {X.shape[0]} baris, {X.shape[1]} fitur")
print(f"[INFO] Fitur: {list(X.columns)}")

# ============================================================
# 3. Split Data
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[INFO] Training set: {X_train.shape[0]} sampel")
print(f"[INFO] Testing set: {X_test.shape[0]} sampel")

# ============================================================
# 4. Training Model dengan MLflow Autolog
# ============================================================
print("\n" + "=" * 60)
print("Memulai training model RandomForestRegressor...")
print("MLflow Project Mode")
print("=" * 60)

# Aktifkan autolog
mlflow.autolog()

with mlflow.start_run(run_name="mlproject-random-forest", nested=True):
    # Inisialisasi dan training model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Prediksi dan evaluasi
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Cetak hasil
    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL (MLflow Project)")
    print("=" * 60)
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print("=" * 60)
    
    # Feature importance
    feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
    importances = model.feature_importances_
    
    print("\nFeature Importance:")
    print("-" * 40)
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {imp:.4f}")
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n[INFO] MLflow Run ID: {run_id}")

print("\n[SELESAI] Training model via MLflow Project selesai!")
