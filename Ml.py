# src/forecasting/ml.py
import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error


def create_features(series, n_lags=12):
    """Cria features de lags + estatísticas + calendário."""
    df = pd.DataFrame({"y": series})
    df.index = pd.date_range(start="2000-01-01", periods=len(df), freq="M")

    # --- Lags ---
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # --- Rolling mean & std ---
    df["rolling_mean_3"] = df["y"].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df["y"].shift(1).rolling(window=6).mean()
    df["rolling_mean_12"] = df["y"].shift(1).rolling(window=12).mean()
    df["rolling_std_3"] = df["y"].shift(1).rolling(window=3).std()

    # --- Diferença mês a mês ---
    df["diff_1"] = df["y"].diff(1)

    # --- Rolling min/max ---
    df["rolling_min_3"] = df["y"].shift(1).rolling(window=3).min()
    df["rolling_max_3"] = df["y"].shift(1).rolling(window=3).max()

    # --- Features de calendário ---
    df["month"] = df.index.month
    df["year"] = df.index.year

    df.dropna(inplace=True)
    return df


def run_ml_models(series, horizon=12, train_size=0.8):
    """
    Roda modelos de ML (RF, XGB, LGBM) para previsão univariada com features enriquecidas.
    Retorna o melhor modelo baseado no MAPE.
    """
    series = pd.Series(series).reset_index(drop=True)

    # Caso 1: todos valores são zero
    if (series == 0).all():
        return {"best_model": "AllZero", "forecast": np.zeros(horizon), "mape": 0.0}

    # Caso 2: baixa variância
    if series.var() < 1e-8:
        return {"best_model": "LowVarianceMean", "forecast": np.full(horizon, series.mean()), "mape": None}

    # Criar features
    df_features = create_features(series, n_lags=max(12, horizon))

    # Split em porcentagem
    train_size_abs = int(len(df_features) * train_size)
    train_df = df_features.iloc[:train_size_abs]
    valid_df = df_features.iloc[train_size_abs:]

    X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
    X_valid, y_valid = valid_df.drop(columns=["y"]), valid_df["y"]

    results = {}

    # --- RANDOM FOREST ---
    def rf_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return mean_absolute_percentage_error(y_valid, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(rf_objective, n_trials=30, show_progress_bar=False)
    best_rf_params = study.best_params
    best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
    best_rf_model.fit(X_train, y_train)
    mape_rf = mean_absolute_percentage_error(y_valid, best_rf_model.predict(X_valid))
    results["RandomForest"] = (best_rf_model, mape_rf)

    # --- XGBOOST ---
    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return mean_absolute_percentage_error(y_valid, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(xgb_objective, n_trials=30, show_progress_bar=False)
    best_xgb_params = study.best_params
    best_xgb_model = XGBRegressor(**best_xgb_params, random_state=42)
    best_xgb_model.fit(X_train, y_train)
    mape_xgb = mean_absolute_percentage_error(y_valid, best_xgb_model.predict(X_valid))
    results["XGBoost"] = (best_xgb_model, mape_xgb)

    # --- LIGHTGBM ---
    def lgbm_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        return mean_absolute_percentage_error(y_valid, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(lgbm_objective, n_trials=30, show_progress_bar=False)
    best_lgbm_params = study.best_params
    best_lgbm_model = LGBMRegressor(**best_lgbm_params, random_state=42)
    best_lgbm_model.fit(X_train, y_train)
    mape_lgbm = mean_absolute_percentage_error(y_valid, best_lgbm_model.predict(X_valid))
    results["LightGBM"] = (best_lgbm_model, mape_lgbm)

    # --- Selecionar melhor ---
    best_model_name = min(results, key=lambda k: results[k][1])
    best_model, best_mape = results[best_model_name]

    # Forecast multi-step
    df_forecast = df_features.copy()
    forecast = []
    last_index = df_forecast.index[-1]

    for step in range(1, horizon + 1):
        next_date = last_index + pd.DateOffset(months=step)
        row = {}
        for lag in range(1, max(12, horizon) + 1):
            if lag <= len(df_forecast):
                row[f"lag_{lag}"] = df_forecast["y"].iloc[-lag]
            else:
                row[f"lag_{lag}"] = np.nan

        row["rolling_mean_3"] = pd.Series([row[f"lag_{i}"] for i in range(1, 4)]).mean()
        row["rolling_mean_6"] = pd.Series([row[f"lag_{i}"] for i in range(1, 7)]).mean()
        row["rolling_mean_12"] = pd.Series([row[f"lag_{i}"] for i in range(1, 13)]).mean()
        row["rolling_std_3"] = pd.Series([row[f"lag_{i}"] for i in range(1, 4)]).std()
        row["diff_1"] = row["lag_1"] - row["lag_2"] if not pd.isna(row["lag_2"]) else 0
        row["rolling_min_3"] = pd.Series([row[f"lag_{i}"] for i in range(1, 4)]).min()
        row["rolling_max_3"] = pd.Series([row[f"lag_{i}"] for i in range(1, 4)]).max()
        row["month"] = next_date.month
        row["year"] = next_date.year

        X_pred = pd.DataFrame([row])
        y_pred = best_model.predict(X_pred)[0]
        forecast.append(y_pred)

        df_forecast.loc[next_date] = [y_pred] + list(X_pred.iloc[0].values)

    return {"best_model": best_model_name, "forecast": np.array(forecast), "mape": best_mape}
