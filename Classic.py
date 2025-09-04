# src/forecasting/classical.py
import numpy as np
import pandas as pd
import optuna
from loguru import logger
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.forecasting.theta import ThetaModel

# Prophet é opcional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# =========================================================
# Utilitários
# =========================================================

def _prepare_series(series: pd.Series | pd.DataFrame) -> pd.Series:
    """Converte DataFrame em Series e garante frequência mensal."""
    if isinstance(series, pd.DataFrame):
        if {"data", "valor"}.issubset(series.columns):
            series = series.set_index("data")["valor"]
        else:
            logger.error("DataFrame inválido: precisa ter colunas 'data' e 'valor'.")
            raise ValueError("O DataFrame deve ter coluna 'valor' e 'data'.")
    return series.asfreq("MS")


def _safe_mape(y_true, y_pred) -> float:
    """MAPE com fallback em caso de erro."""
    try:
        return mean_absolute_percentage_error(y_true, y_pred)
    except Exception as e:
        logger.warning(f"Erro ao calcular MAPE: {e}")
        return np.inf


# =========================================================
# Modelos básicos / especiais
# =========================================================

def _fit_zero_forecast(train, valid):
    if not (train == 0).all():
        return None
    forecast = pd.Series([0] * len(valid), index=valid.index)
    return "ZeroForecast", {}, _safe_mape(valid, forecast), forecast


def _fit_mean_constante(train, valid):
    if train.std() >= 1e-6:
        return None
    forecast = pd.Series([train.mean()] * len(valid), index=valid.index)
    return "MediaConstante", {}, _safe_mape(valid, forecast), forecast


def _fit_naive(train, valid):
    forecast = pd.Series([train.iloc[-1]] * len(valid), index=valid.index)
    return "Naive", {}, _safe_mape(valid, forecast), forecast


def _fit_mean(train, valid):
    forecast = pd.Series([train.mean()] * len(valid), index=valid.index)
    return "MeanNaive", {}, _safe_mape(valid, forecast), forecast


def _fit_seasonal_naive(train, valid):
    if len(train) <= 12:
        return None
    season_vals = list(train.iloc[-12:].values)
    reps = int(np.ceil(len(valid) / 12))
    forecast = pd.Series((season_vals * reps)[:len(valid)], index=valid.index)
    return "SeasonalNaive", {}, _safe_mape(valid, forecast), forecast


# =========================================================
# Modelos mais complexos
# =========================================================

def _fit_moving_average(train, valid):
    def objective(trial):
        window = trial.suggest_int("window", 2, min(24, len(train) // 2))
        forecast_vals = [train.rolling(window).mean().iloc[-1]] * len(valid)
        return _safe_mape(valid, forecast_vals)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_window = study.best_params["window"]
    forecast = pd.Series([train.rolling(best_window).mean().iloc[-1]] * len(valid), index=valid.index)
    return "MovingAverage", {"window": best_window}, study.best_value, forecast


def _fit_holt_winters(train, valid):
    def objective(trial):
        trend = trial.suggest_categorical("trend", [None, "add", "mul"])
        seasonal = trial.suggest_categorical("seasonal", [None, "add", "mul"])
        seasonal_periods = trial.suggest_int("seasonal_periods", 0, 12)
        try:
            model = ExponentialSmoothing(
                train, trend=trend, seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None
            )
            fit = model.fit(optimized=True)
            forecast_vals = fit.forecast(len(valid))
            return _safe_mape(valid, forecast_vals)
        except Exception as e:
            logger.debug(f"Erro em Holt-Winters trial: {e}")
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    params = study.best_params
    model = ExponentialSmoothing(
        train, trend=params["trend"], seasonal=params["seasonal"],
        seasonal_periods=params["seasonal_periods"] if params["seasonal"] else None
    )
    fit = model.fit(optimized=True)
    forecast = fit.forecast(len(valid))
    return "HoltWinters", params, study.best_value, forecast


def _fit_sarima(train, valid):
    def objective(trial):
        try:
            p, d, q = trial.suggest_int("p", 0, 3), trial.suggest_int("d", 0, 2), trial.suggest_int("q", 0, 3)
            P, D, Q = trial.suggest_int("P", 0, 2), trial.suggest_int("D", 0, 1), trial.suggest_int("Q", 0, 2)
            m = trial.suggest_categorical("m", [0, 6, 12])
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            return _safe_mape(valid, fit.forecast(len(valid)))
        except Exception as e:
            logger.debug(f"Erro em SARIMA trial: {e}")
            return np.inf

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40, show_progress_bar=False)

    params = study.best_params
    model = SARIMAX(
        train,
        order=(params["p"], params["d"], params["q"]),
        seasonal_order=(params["P"], params["D"], params["Q"], params["m"]),
        enforce_stationarity=False, enforce_invertibility=False
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(len(valid))
    return "SARIMA", params, study.best_value, forecast


def _fit_theta(train, valid):
    try:
        theta_model = ThetaModel(train, period=12)
        fit = theta_model.fit()
        forecast = fit.forecast(len(valid))
        return "Theta", {}, _safe_mape(valid, forecast), forecast
    except Exception as e:
        logger.warning(f"Erro ao treinar Theta: {e}")
        return None


def _fit_linear_regression(train, valid):
    try:
        X_train = np.arange(len(train)).reshape(-1, 1)
        reg = LinearRegression().fit(X_train, train.values)
        X_valid = np.arange(len(train), len(train) + len(valid)).reshape(-1, 1)
        forecast = reg.predict(X_valid)
        return "LinearTrend", {}, _safe_mape(valid, forecast), pd.Series(forecast, index=valid.index)
    except Exception as e:
        logger.warning(f"Erro ao treinar LinearTrend: {e}")
        return None


def _fit_prophet(train, valid):
    if not PROPHET_AVAILABLE:
        return None
    try:
        df_train = train.reset_index().rename(columns={"data": "ds", "valor": "y"})
        df_train.columns = ["ds", "y"]

        model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model.fit(df_train)

        future = model.make_future_dataframe(periods=len(valid), freq="MS")
        forecast_vals = model.predict(future).iloc[-len(valid):]["yhat"].values
        return "Prophet", {}, _safe_mape(valid, forecast_vals), pd.Series(forecast_vals, index=valid.index)
    except Exception as e:
        logger.warning(f"Erro ao treinar Prophet: {e}")
        return None


# =========================================================
# Pipeline principal
# =========================================================

def run_classical_models(series, horizon=12, valid_size=0.2):
    logger.info("Iniciando treinamento de modelos clássicos...")

    series = _prepare_series(series)

    # Split treino/validação
    split_idx = int(len(series) * (1 - valid_size))
    train, valid = series.iloc[:split_idx], series.iloc[split_idx:]

    results = []
    for fit_func in [
        _fit_zero_forecast, _fit_mean_constante,  # <- viraram modelos!
        _fit_naive, _fit_mean, _fit_seasonal_naive,
        _fit_moving_average, _fit_holt_winters,
        _fit_sarima, _fit_theta,
        _fit_linear_regression, _fit_prophet
    ]:
        try:
            res = fit_func(train, valid)
            if res:
                results.append(res)
        except Exception as e:
            logger.error(f"Erro em {fit_func.__name__}: {e}")

    if not results:
        logger.error("Nenhum modelo conseguiu gerar previsão.")
        raise RuntimeError("Falha em todos os modelos.")

    # Melhor modelo
    best_model = min(results, key=lambda x: x[2])
    nome_modelo, melhores_params, _, _ = best_model
    logger.success(f"Melhor modelo: {nome_modelo} | Params: {melhores_params}")

    # TODO: implementar treinamento final no dataset completo
    return {
        "melhor_modelo": nome_modelo,
        "melhores_parametros": melhores_params,
        "forecast": None
    }
