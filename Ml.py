# src/forecasting/ml.py
import numpy as np
import pandas as pd
import optuna
from loguru import logger

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error


# ==============================
# Utils
# ==============================
def create_features(series, n_lags=12):
    """Cria features de lags + estatísticas + calendário."""
    try:
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

    except Exception as e:
        logger.error(f"Erro ao criar features: {e}")
        raise


def optimize_model(model_class, objective_fn, X_train, y_train, X_valid, y_valid, n_trials=30):
    """Roda Optuna para encontrar melhores hiperparâmetros."""
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective_fn(trial, model_class, X_train, y_train, X_valid, y_valid),
                       n_trials=n_trials,
                       show_progress_bar=False)
        return study.best_params
    except Exception as e:
        logger.error(f"Erro na otimização do modelo {model_class.__name__}: {e}")
        raise


def forecast_with_model(model, df_features, horizon):
    """Gera previsões multi-step para horizon meses."""
    try:
        forecast = []
        df_forecast = df_features.copy()
        last_index = df_forecast.index[-1]

        for step in range(1, horizon + 1):
            next_date = last_index + pd.DateOffset(months=step)
            row = {}

            for lag in range(1, max(12, horizon) + 1):
                row[f"lag_{lag}"] = df_forecast["y"].iloc[-lag] if lag <= len(df_forecast) else np.nan

            # rolling features
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
            y_pred = model.predict(X_pred)[0]
            forecast.append(y_pred)

            df_forecast.loc[next_date] = [y_pred] + list(X_pred.iloc[0].values)

        forecast_index = pd.date_range(start=last_index + pd.DateOffset(months=1),
                                       periods=horizon,
                                       freq="M").strftime("%Y-%m-%d")

        forecast_series = pd.Series(forecast, index=pd.to_datetime(forecast_index))

        return forecast_series

    except Exception as e:
        logger.error(f"Erro ao gerar forecast: {e}")
        raise


# ==============================
# Modelagem principal
# ==============================
def run_ml_models(series, horizon=12, train_size=0.8):
    """
    Roda modelos de ML (RF, XGB, LGBM) para previsão univariada com features enriquecidas.
    Retorna o melhor modelo baseado no MAPE.
    """
    try:
        series = pd.Series(series).reset_index(drop=True)

        # --- Casos especiais ---
        if (series == 0).all():
            logger.info("Série com todos valores iguais a zero.")
            forecast_index = pd.date_range(start="2000-01-01", periods=horizon, freq="M").strftime("%Y-%m-%d")
            return {"best_model": "AllZero",
                    "forecast": pd.Series(np.zeros(horizon), index=pd.to_datetime(forecast_index)),
                    "mape": 0.0}

        if series.var() < 1e-8:
            logger.info("Série com baixa variância, retornando média.")
            forecast_index = pd.date_range(start="2000-01-01", periods=horizon, freq="M").strftime("%Y-%m-%d")
            return {"best_model": "LowVarianceMean",
                    "forecast": pd.Series(np.full(horizon, series.mean()), index=pd.to_datetime(forecast_index)),
                    "mape": None}

        # Criar features
        df_features = create_features(series, n_lags=max(12, horizon))

        # Split
        train_size_abs = int(len(df_features) * train_size)
        train_df = df_features.iloc[:train_size_abs]
        valid_df = df_features.iloc[train_size_abs:]

        X_train, y_train = train_df.drop(columns=["y"]), train_df["y"]
        X_valid, y_valid = valid_df.drop(columns=["y"]), valid_df["y"]

        results = {}

        # RANDOM FOREST
        def rf_objective(trial, model_class, X_train, y_train, X_valid, y_valid):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
            }
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return mean_absolute_percentage_error(y_valid, y_pred)

        best_rf_params = optimize_model(RandomForestRegressor, rf_objective, X_train, y_train, X_valid, y_valid)
        best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
        best_rf_model.fit(X_train, y_train)
        mape_rf = mean_absolute_percentage_error(y_valid, best_rf_model.predict(X_valid))
        results["RandomForest"] = (best_rf_model, mape_rf)

        # XGBOOST
        def xgb_objective(trial, model_class, X_train, y_train, X_valid, y_valid):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
            }
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return mean_absolute_percentage_error(y_valid, y_pred)

        best_xgb_params = optimize_model(XGBRegressor, xgb_objective, X_train, y_train, X_valid, y_valid)
        best_xgb_model = XGBRegressor(**best_xgb_params, random_state=42)
        best_xgb_model.fit(X_train, y_train)
        mape_xgb = mean_absolute_percentage_error(y_valid, best_xgb_model.predict(X_valid))
        results["XGBoost"] = (best_xgb_model, mape_xgb)

        # LIGHTGBM
        def lgbm_objective(trial, model_class, X_train, y_train, X_valid, y_valid):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", -1, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "random_state": 42,
            }
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            return mean_absolute_percentage_error(y_valid, y_pred)

        best_lgbm_params = optimize_model(LGBMRegressor, lgbm_objective, X_train, y_train, X_valid, y_valid)
        best_lgbm_model = LGBMRegressor(**best_lgbm_params, random_state=42)
        best_lgbm_model.fit(X_train, y_train)
        mape_lgbm = mean_absolute_percentage_error(y_valid, best_lgbm_model.predict(X_valid))
        results["LightGBM"] = (best_lgbm_model, mape_lgbm)

        # Melhor modelo
        best_model_name = min(results, key=lambda k: results[k][1])
        best_model, best_mape = results[best_model_name]

        # Forecast
        forecast_series = forecast_with_model(best_model, df_features, horizon)

        return {"best_model": best_model_name, "forecast": forecast_series, "mape": best_mape}

    except Exception as e:
        logger.error(f"Erro ao rodar modelos de ML: {e}")
        raise
