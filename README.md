# Test
# -*- coding: utf-8 -*-
"""
Benchmark clássico de séries temporais com backtest + Optuna (Otimização Bayesiana).

Modelos:
- Naive, Seasonal Naive, Moving Average (baseline; MA otimiza janela)
- Holt (Suavização Exponencial com tendência)
- ETS / Holt-Winters (aditivo/multiplicativo)
- SARIMA (via SARIMAX)
- Prophet (opcional, se instalado)
- Theta (implementação clássica simples)
- STL + ARIMA (decomposição + ARIMA no componente ajustado sazonalmente)

Validação:
- Backtest rolling (expanding window), com horizonte fixo e n_splits.
- Métrica padrão da otimização: RMSE (minimização).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error

import optuna

# Statsmodels
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

# Prophet (opcional)
_HAS_PROPHET = False
try:
    from prophet import Prophet  # pip install prophet
    _HAS_PROPHET = True
except Exception:
    try:
        from fbprophet import Prophet  # fallback antigo
        _HAS_PROPHET = True
    except Exception:
        _HAS_PROPHET = False


# =========================================================
# Métricas
# =========================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred))
    }


# =========================================================
# Baselines
# =========================================================
def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], horizon)

def seasonal_naive_forecast(train: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    reps = int(np.ceil(horizon / season_length))
    return np.tile(train.iloc[-season_length:], reps)[:horizon]

def moving_average_forecast(train: pd.Series, horizon: int, window: int = 6) -> np.ndarray:
    window = max(1, min(window, len(train)))
    return np.repeat(train.iloc[-window:].mean(), horizon)


# =========================================================
# Modelos Clássicos
# =========================================================
def holt_forecast(train: pd.Series, horizon: int, smoothing_level: Optional[float]=None,
                  smoothing_trend: Optional[float]=None, damped_trend: bool=True) -> np.ndarray:
    model = Holt(train, damped_trend=damped_trend, initialization_method="estimated")
    fit = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend, optimized=(smoothing_level is None or smoothing_trend is None))
    return fit.forecast(horizon).values

def ets_forecast(train: pd.Series, horizon: int, trend: Optional[str]="add",
                 seasonal: Optional[str]="add", seasonal_periods: int=12,
                 damped_trend: bool=False, use_boxcox: Optional[bool]=None) -> np.ndarray:
    model = ExponentialSmoothing(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
        initialization_method="estimated",
        use_boxcox=use_boxcox
    )
    fit = model.fit(optimized=True)
    return fit.forecast(horizon).values

def sarima_forecast(train: pd.Series, horizon: int,
                    order: Tuple[int,int,int]=(1,1,1),
                    seasonal_order: Tuple[int,int,int,int]=(0,0,0,0),
                    enforce_stationarity: bool=True,
                    enforce_invertibility: bool=True) -> np.ndarray:
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=enforce_stationarity,
                    enforce_invertibility=enforce_invertibility)
    fit = model.fit(disp=False)
    return fit.forecast(steps=horizon).values

def prophet_forecast(train: pd.Series, horizon: int,
                     yearly_seasonality: str="auto",
                     seasonality_mode: str="additive",
                     changepoint_prior_scale: float=0.05,
                     seasonality_prior_scale: float=10.0,
                     holidays_prior_scale: float=10.0) -> np.ndarray:
    if not _HAS_PROPHET:
        raise RuntimeError("Prophet não está instalado.")
    df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        weekly_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq=pd.infer_freq(train.index) or "M")
    fcst = m.predict(future)
    return fcst["yhat"].iloc[-horizon:].values

def theta_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """
    Implementação simples do método Theta (Assimakopoulos & Nikolopoulos, 2000):
    - Decompõe a série em duas "theta lines": theta=0 (equivale a tendência linear)
      e theta=2 (equivale à suavização exponencial simples com alpha ~ 1).
    - Combina previsões (média) e adiciona drift.
    Esta versão funciona bem para muitas séries sem dependências externas.
    """
    y = train.values.astype(float)
    n = len(y)
    t = np.arange(1, n+1, dtype=float)

    # Regressão linear simples para tendência
    # y ~ a + b * t
    x = np.vstack([np.ones(n), t]).T
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    a, b = beta[0], beta[1]

    # Serie sem tendência (detrended)
    trend = a + b * t
    y_detrended = y - trend

    # SES (suavização exponencial simples) na série sem tendência
    # Escolhe alpha por minimização simples do SSE (grid curto) para robustez/velocidade
    best_alpha, best_sse = 0.2, np.inf
    for alpha in np.linspace(0.05, 0.95, 19):
        level = y_detrended[0]
        sse = 0.0
        for i in range(1, n):
            level = alpha * y_detrended[i-1] + (1 - alpha) * level
            err = y_detrended[i] - level
            sse += err * err
        if sse < best_sse:
            best_sse, best_alpha = sse, alpha

    # Reestima nível com melhor alpha e faz forecast
    level = y_detrended[0]
    for i in range(1, n):
        level = best_alpha * y_detrended[i-1] + (1 - best_alpha) * level
    # Forecast da parte sem tendência é constante = último nível
    fcst_seasonfree = np.repeat(level, horizon)

    # Recoloca tendência (drift linear)
    t_future = np.arange(n+1, n+horizon+1, dtype=float)
    trend_future = a + b * t_future

    # Combinação: média simples (theta=0 e theta=2)
    yhat = fcst_seasonfree + trend_future
    return yhat

def stl_arima_forecast(train: pd.Series, horizon: int, season_length: int=12,
                       arima_order: Tuple[int,int,int]=(1,1,1)) -> np.ndarray:
    """
    Decomposição STL para retirar sazonalidade, ARIMA no componente ajustado,
    e depois recoloca sazonalidade via seasonal naive do STL.
    """
    stl = STL(train, period=season_length, robust=True).fit()
    season = stl.seasonal
    resid_plus_trend = train - season  # série dessazonalizada

    # ARIMA no dessazonalizado
    model = SARIMAX(resid_plus_trend, order=arima_order, seasonal_order=(0,0,0,0))
    fit = model.fit(disp=False)

    # Forecast dessazonalizado
    fcst_res_trend = fit.forecast(steps=horizon)

    # Forecast sazonal (repete último ciclo)
    season_fcst = seasonal_naive_forecast(season, horizon, season_length=season_length)

    return (fcst_res_trend.values + season_fcst)


# =========================================================
# Backtest (rolling origin / expanding window)
# =========================================================
def rolling_backtest(
    ts: pd.Series,
    horizon: int,
    n_splits: int = 3,
    min_train_size: Optional[int] = None,
    step: Optional[int] = None,
    forecaster: Optional[Callable[[pd.Series, int, dict], np.ndarray]] = None,
    forecaster_params: Optional[dict] = None,
    metric: str = "RMSE",
    season_length: int = 12
) -> float:
    """
    Executa backtest com janela crescente (expanding window).
    Em cada split, treina no histórico disponível e prevê 'horizon' passos à frente.

    Retorna a média da métrica escolhida.
    """
    assert horizon > 0
    step = step or horizon
    forecaster_params = forecaster_params or {}

    N = len(ts)
    if min_train_size is None:
        # garante espaço para n_splits * horizon
        min_train_size = max(season_length * 2, N - n_splits * horizon)

    # pontos de corte
    cutpoints = []
    start = min_train_size
    while start + horizon <= N and len(cutpoints) < n_splits:
        cutpoints.append(start)
        start += step

    if len(cutpoints) == 0:
        raise ValueError("Backtest impossível: aumente dados, reduza n_splits ou horizonte.")

    metrics = []
    for cp in cutpoints:
        train = ts.iloc[:cp]
        test = ts.iloc[cp: cp + horizon]

        try:
            pred = forecaster(train, horizon, forecaster_params)
            # Ajusta o comprimento se necessário
            pred = np.array(pred).reshape(-1)
            if len(pred) != len(test):
                # fallback para tentar ajustar
                pred = pred[:len(test)]
            m = evaluate_forecast(test.values, pred)
            metrics.append(m[metric])
        except Exception:
            # penaliza falha
            metrics.append(np.inf)

    score = float(np.nanmean(metrics))
    return score


# =========================================================
# Wrappers para Optuna
# =========================================================
def optimize_moving_average(ts, horizon, n_splits, season_length, n_trials=25):
    def forecaster(train, h, params):
        w = int(params["window"])
        return moving_average_forecast(train, h, window=w)

    def objective(trial):
        window = trial.suggest_int("window", 1, max(3, min(24, max(2, len(ts)//10))))
        params = {"window": window}
        return rolling_backtest(ts, horizon, n_splits, season_length*2, horizon,
                                forecaster, params, metric="RMSE", season_length=season_length)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def optimize_holt(ts, horizon, n_splits, season_length, n_trials=30):
    def forecaster(train, h, params):
        return holt_forecast(train, h,
                             smoothing_level=params.get("alpha"),
                             smoothing_trend=params.get("beta"),
                             damped_trend=params.get("damped"))

    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 0.99)
        beta  = trial.suggest_float("beta", 0.01, 0.99)
        damped = trial.suggest_categorical("damped", [True, False])
        params = {"alpha": alpha, "beta": beta, "damped": damped}
        return rolling_backtest(ts, horizon, n_splits, season_length*2, horizon,
                                forecaster, params, metric="RMSE", season_length=season_length)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def optimize_ets(ts, horizon, n_splits, season_length, n_trials=40):
    def forecaster(train, h, params):
        return ets_forecast(train, h,
                            trend=params["trend"],
                            seasonal=params["seasonal"],
                            seasonal_periods=season_length,
                            damped_trend=params["damped"],
                            use_boxcox=params["boxcox"])

    def objective(trial):
        trend = trial.suggest_categorical("trend", [None, "add", "mul"])
        seasonal = trial.suggest_categorical("seasonal", [None, "add", "mul"])
        damped = trial.suggest_categorical("damped", [True, False])
        boxcox = trial.suggest_categorical("boxcox", [None, True, False])
        # Evita combinações inválidas (ex.: seasonal sem period)
        if seasonal is None and trend is None and damped is True:
            # ainda é válido, apenas um viés; seguimos
            pass
        params = {"trend": trend, "seasonal": seasonal, "damped": damped, "boxcox": boxcox}
        return rolling_backtest(ts, horizon, n_splits, season_length*2, horizon,
                                forecaster, params, metric="RMSE", season_length=season_length)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def optimize_sarima(ts, horizon, n_splits, season_length, n_trials=60):
    def forecaster(train, h, params):
        return sarima_forecast(train, h,
                               order=params["order"],
                               seasonal_order=params["seasonal_order"],
                               enforce_stationarity=True,
                               enforce_invertibility=True)

    def objective(trial):
        # Espaço não gigante para performance
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)

        if season_length > 1:
            P = trial.suggest_int("P", 0, 2)
            D = trial.suggest_int("D", 0, 1)
            Q = trial.suggest_int("Q", 0, 2)
            m = season_length
        else:
            P=D=Q=0
            m=0

        params = {
            "order": (p,d,q),
            "seasonal_order": (P,D,Q,m)
        }
        return rolling_backtest(ts, horizon, n_splits, max(season_length*2, 24), horizon,
                                forecaster, params, metric="RMSE", season_length=season_length)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def optimize_prophet(ts, horizon, n_splits, season_length, n_trials=30):
    if not _HAS_PROPHET:
        return None

    def forecaster(train, h, params):
        return prophet_forecast(
            train, h,
            yearly_seasonality=params["yearly_seasonality"],
            seasonality_mode=params["seasonality_mode"],
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"]
        )

    def objective(trial):
        yearly_seasonality = trial.suggest_categorical("yearly_seasonality", ["auto", True, False])
        seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
        sps = trial.suggest_float("seasonality_prior_scale", 0.01, 50.0, log=True)
        hps = trial.suggest_float("holidays_prior_scale", 0.01, 50.0, log=True)
        params = {
            "yearly_seasonality": yearly_seasonality,
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "holidays_prior_scale": hps
        }
        return rolling_backtest(ts, horizon, n_splits, season_length*2, horizon,
                                forecaster, params, metric="RMSE", season_length=season_length)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

def optimize_stl_arima(ts, horizon, n_splits, season_length, n_trials=30):
    def forecaster(train, h, params):
        return stl_arima_forecast(train, h, season_length=season_length,
                                  arima_order=params["arima_order"])

    def objective(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        params = {"arima_order": (p,d,q)}
        return rolling_backtest(ts, horizon, n_splits, season_length*2, horizon,
                                forecaster, params, metric="RMSE", season_length=season_length)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# =========================================================
# Execução completa
# =========================================================
def run_all_models_with_optimization(
    ts: pd.Series,
    horizon: int = 12,
    season_length: int = 12,
    n_splits: int = 3,
    n_trials: Dict[str, int] = None,
    random_seed: int = 42,
    plot: bool = True
):
    """
    Faz:
    1) Split Train/Test final (holdout) usando 'horizon' como teste.
    2) Para cada modelo, roda Optuna (n_trials) com backtest (n_splits) no conjunto de treino.
    3) Avalia no holdout final e reporta métricas.
    """
    if n_trials is None:
        n_trials = {
            "MA": 25,
            "Holt": 30,
            "ETS": 40,
            "SARIMA": 60,
            "Prophet": 30,
            "Theta": 1,         # Theta não usa Optuna (sem hiperparâmetros nesta versão)
            "STL_ARIMA": 30
        }

    np.random.seed(random_seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train = ts.iloc[:-horizon]
    test = ts.iloc[-horizon:]

    results = {}
    forecasts = {}
    best_params = {}
    studies = {}

    # ==========================
    # Baselines (sem Optuna) + MA com Optuna
    # ==========================
    try:
        # Naive
        pred_naive = naive_forecast(train, horizon)
        results["Naive"] = evaluate_forecast(test, pred_naive)
        forecasts["Naive"] = pred_naive
    except Exception as e:
        results["Naive"] = {"error": str(e)}

    try:
        # Seasonal Naive
        pred_snaive = seasonal_naive_forecast(train, horizon, season_length=season_length)
        results["SeasonalNaive"] = evaluate_forecast(test, pred_snaive)
        forecasts["SeasonalNaive"] = pred_snaive
    except Exception as e:
        results["SeasonalNaive"] = {"error": str(e)}

    # Moving Average com Optuna
    try:
        study_ma = optimize_moving_average(train, horizon, n_splits, season_length, n_trials=n_trials.get("MA", 25))
        studies["MA"] = study_ma
        best_params["MA"] = study_ma.best_params
        pred_ma = moving_average_forecast(train, horizon, window=int(study_ma.best_params["window"]))
        results["MovingAverage"] = evaluate_forecast(test, pred_ma)
        forecasts["MovingAverage"] = pred_ma
    except Exception as e:
        results["MovingAverage"] = {"error": str(e)}

    # ==========================
    # Holt
    # ==========================
    try:
        study_holt = optimize_holt(train, horizon, n_splits, season_length, n_trials=n_trials.get("Holt", 30))
        studies["Holt"] = study_holt
        best_params["Holt"] = study_holt.best_params
        pred_holt = holt_forecast(train, horizon,
                                  smoothing_level=study_holt.best_params["alpha"],
                                  smoothing_trend=study_holt.best_params["beta"],
                                  damped_trend=study_holt.best_params["damped"])
        results["Holt"] = evaluate_forecast(test, pred_holt)
        forecasts["Holt"] = pred_holt
    except Exception as e:
        results["Holt"] = {"error": str(e)}

    # ==========================
    # ETS / Holt-Winters
    # ==========================
    try:
        study_ets = optimize_ets(train, horizon, n_splits, season_length, n_trials=n_trials.get("ETS", 40))
        studies["ETS"] = study_ets
        best_params["ETS"] = study_ets.best_params
        pred_ets = ets_forecast(train, horizon,
                                trend=study_ets.best_params["trend"],
                                seasonal=study_ets.best_params["seasonal"],
                                seasonal_periods=season_length,
                                damped_trend=study_ets.best_params["damped"],
                                use_boxcox=study_ets.best_params["boxcox"])
        results["ETS"] = evaluate_forecast(test, pred_ets)
        forecasts["ETS"] = pred_ets
    except Exception as e:
        results["ETS"] = {"error": str(e)}

    # ==========================
    # SARIMA
    # ==========================
    try:
        study_sarima = optimize_sarima(train, horizon, n_splits, season_length, n_trials=n_trials.get("SARIMA", 60))
        studies["SARIMA"] = study_sarima
        best_params["SARIMA"] = study_sarima.best_params
        pred_sarima = sarima_forecast(
            train, horizon,
            order=(study_sarima.best_params["p"], study_sarima.best_params["d"], study_sarima.best_params["q"]),
            seasonal_order=(
                study_sarima.best_params.get("P", 0),
                study_sarima.best_params.get("D", 0),
                study_sarima.best_params.get("Q", 0),
                season_length if season_length > 1 else 0
            ),
            enforce_stationarity=True,
            enforce_invertibility=True
        )
        results["SARIMA"] = evaluate_forecast(test, pred_sarima)
        forecasts["SARIMA"] = pred_sarima
    except Exception as e:
        results["SARIMA"] = {"error": str(e)}

    # ==========================
    # Prophet (opcional)
    # ==========================
    if _HAS_PROPHET:
        try:
            study_prophet = optimize_prophet(train, horizon, n_splits, season_length, n_trials=n_trials.get("Prophet", 30))
            if study_prophet is not None:
                studies["Prophet"] = study_prophet
                best_params["Prophet"] = study_prophet.best_params
                pred_prophet = prophet_forecast(
                    train, horizon,
                    yearly_seasonality=study_prophet.best_params["yearly_seasonality"],
                    seasonality_mode=study_prophet.best_params["seasonality_mode"],
                    changepoint_prior_scale=study_prophet.best_params["changepoint_prior_scale"],
                    seasonality_prior_scale=study_prophet.best_params["seasonality_prior_scale"],
                    holidays_prior_scale=study_prophet.best_params["holidays_prior_scale"]
                )
                results["Prophet"] = evaluate_forecast(test, pred_prophet)
                forecasts["Prophet"] = pred_prophet
        except Exception as e:
            results["Prophet"] = {"error": str(e)}
    else:
        results["Prophet"] = {"info": "Prophet não instalado; modelo ignorado."}

    # ==========================
    # Theta (sem Optuna nessa versão)
    # ==========================
    try:
        pred_theta = theta_forecast(train, horizon)
        results["Theta"] = evaluate_forecast(test, pred_theta)
        forecasts["Theta"] = pred_theta
    except Exception as e:
        results["Theta"] = {"error": str(e)}

    # ==========================
    # STL + ARIMA
    # ==========================
    try:
        study_stl = optimize_stl_arima(train, horizon, n_splits, season_length, n_trials=n_trials.get("STL_ARIMA", 30))
        studies["STL_ARIMA"] = study_stl
        best_params["STL_ARIMA"] = study_stl.best_params
        pred_stl = stl_arima_forecast(train, horizon, season_length=season_length,
                                      arima_order=(study_stl.best_params["p"], study_stl.best_params["d"], study_stl.best_params["q"]))
        results["STL_ARIMA"] = evaluate_forecast(test, pred_stl)
        forecasts["STL_ARIMA"] = pred_stl
    except Exception as e:
        results["STL_ARIMA"] = {"error": str(e)}

    # ==========================
    # Saída
    # ==========================
    print("\n=== Resultados (holdout) ===")
    df_results = []
    for model, res in results.items():
        if isinstance(res, dict) and "MAE" in res:
            df_results.append([model, res["MAE"], res["RMSE"], res["MAPE"]])
            print(f"{model:>12s} | MAE={res['MAE']:.3f} | RMSE={res['RMSE']:.3f} | MAPE={res['MAPE']:.2f}%")
        else:
            print(f"{model:>12s} | {res}")

    df_results = pd.DataFrame(df_results, columns=["Modelo","MAE","RMSE","MAPE"]).sort_values("RMSE")
    print("\n=== Top por RMSE ===")
    print(df_results.to_string(index=False))

    # Plot comparativo
    if plot:
        plt.figure(figsize=(14,7))
        plt.plot(ts.index, ts.values, label="Real", linewidth=2)
        for name, preds in forecasts.items():
            try:
                plt.plot(test.index, preds, label=name, alpha=0.8)
            except Exception:
                pass
        plt.title("Comparação de Modelos (Previsão no Holdout)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "results": results,
        "best_params": best_params,
        "studies": studies,
        "forecasts": forecasts,
        "summary": df_results
    }


# =========================================================
# Exemplo de uso
# =========================================================
if __name__ == "__main__":
    # Série mensal sintética com sazonalidade + ruído
    rng = pd.date_range(start="2005-01-01", periods=180, freq="M")
    y = 100 + 0.2*np.arange(len(rng)) \
        + 10*np.sin(2*np.pi*np.arange(len(rng))/12) \
        + np.random.normal(0, 2.0, size=len(rng))
    ts = pd.Series(y, index=rng)

    # Configurações
    H = 12
    SEASON = 12
    N_SPLITS = 3
    N_TRIALS = {
        "MA": 20,
        "Holt": 25,
        "ETS": 30,
        "SARIMA": 50,
        "Prophet": 25,
        "Theta": 1,
        "STL_ARIMA": 25
    }

    out = run_all_models_with_optimization(
        ts,
        horizon=H,
        season_length=SEASON,
        n_splits=N_SPLITS,
        n_trials=N_TRIALS,
        random_seed=42,
        plot=True
    )
