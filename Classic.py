# src/forecasting/classical.py
import numpy as np
import pandas as pd
import optuna
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


def run_classical_models(series, horizon=12, valid_size=0.2):
    """
    Treina modelos clássicos e naives com Optuna para previsão.

    Parâmetros:
        series (pd.Series ou pd.DataFrame): Série temporal (índice = datas, valores = métrica)
        horizon (int): Passos futuros a prever (default=12)
        valid_size (float): Percentual da série para validação (default=0.2)

    Retorna:
        dict: {
            "melhor_modelo": nome do modelo,
            "melhores_parametros": dict,
            "forecast": pd.Series com previsões futuras
        }
    """

    # Se vier DataFrame, converte para Series
    if isinstance(series, pd.DataFrame):
        if "valor" in series.columns:
            series = series.set_index("data")["valor"]
        else:
            raise ValueError("O DataFrame deve ter coluna 'valor' e 'data'.")

    series = series.asfreq("MS")  # Frequência mensal

    # Caso especial: todos zeros
    if (series == 0).all():
        forecast = pd.Series(
            [0] * horizon,
            index=pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(),
                                periods=horizon, freq="MS")
        )
        return {
            "melhor_modelo": "ZeroForecast",
            "melhores_parametros": {},
            "forecast": forecast
        }

    # Caso especial: variação muito baixa
    if series.std() < 1e-6:
        forecast = pd.Series(
            [series.mean()] * horizon,
            index=pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(),
                                periods=horizon, freq="MS")
        )
        return {
            "melhor_modelo": "MediaConstante",
            "melhores_parametros": {},
            "forecast": forecast
        }

    # Split treino/validação proporcional
    split_idx = int(len(series) * (1 - valid_size))
    train = series.iloc[:split_idx]
    valid = series.iloc[split_idx:]

    results = []

    # ----------------------------
    # Modelo Naive
    naive_forecast = pd.Series([train.iloc[-1]] * len(valid), index=valid.index)
    results.append(("Naive", {}, mean_absolute_percentage_error(valid, naive_forecast), naive_forecast))

    # Média constante
    mean_val = train.mean()
    mean_forecast = pd.Series([mean_val] * len(valid), index=valid.index)
    results.append(("MeanNaive", {}, mean_absolute_percentage_error(valid, mean_forecast), mean_forecast))

    # Seasonal Naive
    if len(train) > 12:
        season_naive_vals = list(train.iloc[-12:].values)
        reps = int(np.ceil(len(valid) / 12))
        season_naive_forecast = pd.Series((season_naive_vals * reps)[:len(valid)], index=valid.index)
        results.append(("SeasonalNaive", {}, mean_absolute_percentage_error(valid, season_naive_forecast), season_naive_forecast))

    # ----------------------------
    # Média móvel otimizada
    def objective_ma(trial):
        window = trial.suggest_int("window", 2, min(24, len(train)//2))
        forecast_vals = [train.rolling(window).mean().iloc[-1]] * len(valid)
        return mean_absolute_percentage_error(valid, forecast_vals)

    study_ma = optuna.create_study(direction="minimize")
    study_ma.optimize(objective_ma, n_trials=20, show_progress_bar=False)
    best_window = study_ma.best_params["window"]
    ma_forecast = pd.Series([train.rolling(best_window).mean().iloc[-1]] * len(valid), index=valid.index)
    results.append(("MovingAverage", {"window": best_window}, study_ma.best_value, ma_forecast))

    # ----------------------------
    # Holt-Winters (ETS)
    def objective_hw(trial):
        trend = trial.suggest_categorical("trend", [None, "add", "mul"])
        seasonal = trial.suggest_categorical("seasonal", [None, "add", "mul"])
        seasonal_periods = trial.suggest_int("seasonal_periods", 0, 12)
        try:
            model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                         seasonal_periods=seasonal_periods if seasonal else None)
            fit = model.fit(optimized=True)
            forecast_vals = fit.forecast(len(valid))
            return mean_absolute_percentage_error(valid, forecast_vals)
        except:
            return np.inf

    study_hw = optuna.create_study(direction="minimize")
    study_hw.optimize(objective_hw, n_trials=30, show_progress_bar=False)
    best_hw_params = study_hw.best_params
    model_hw = ExponentialSmoothing(train,
                                    trend=best_hw_params["trend"],
                                    seasonal=best_hw_params["seasonal"],
                                    seasonal_periods=best_hw_params["seasonal_periods"] if best_hw_params["seasonal"] else None)
    fit_hw = model_hw.fit(optimized=True)
    hw_forecast = fit_hw.forecast(len(valid))
    results.append(("HoltWinters", best_hw_params, study_hw.best_value, hw_forecast))

    # ----------------------------
    # SARIMA
    def objective_sarima(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        P = trial.suggest_int("P", 0, 2)
        D = trial.suggest_int("D", 0, 1)
        Q = trial.suggest_int("Q", 0, 2)
        m = trial.suggest_categorical("m", [0, 6, 12])
        try:
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            forecast_vals = fit.forecast(len(valid))
            return mean_absolute_percentage_error(valid, forecast_vals)
        except:
            return np.inf

    study_sarima = optuna.create_study(direction="minimize")
    study_sarima.optimize(objective_sarima, n_trials=40, show_progress_bar=False)
    best_sarima_params = study_sarima.best_params
    model_sarima = SARIMAX(train,
                           order=(best_sarima_params["p"], best_sarima_params["d"], best_sarima_params["q"]),
                           seasonal_order=(best_sarima_params["P"], best_sarima_params["D"], best_sarima_params["Q"], best_sarima_params["m"]),
                           enforce_stationarity=False, enforce_invertibility=False)
    fit_sarima = model_sarima.fit(disp=False)
    sarima_forecast = fit_sarima.forecast(len(valid))
    results.append(("SARIMA", best_sarima_params, study_sarima.best_value, sarima_forecast))

    # ----------------------------
    # Theta Model
    try:
        theta_model = ThetaModel(train, period=12)
        fit_theta = theta_model.fit()
        theta_forecast = fit_theta.forecast(len(valid))
        results.append(("Theta", {}, mean_absolute_percentage_error(valid, theta_forecast), theta_forecast))
    except:
        pass

    # ----------------------------
    # Linear Regression (tendência simples)
    try:
        X_train = np.arange(len(train)).reshape(-1, 1)
        y_train = train.values
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        X_valid = np.arange(len(train), len(train) + len(valid)).reshape(-1, 1)
        lr_forecast = reg.predict(X_valid)
        results.append(("LinearTrend", {}, mean_absolute_percentage_error(valid, lr_forecast),
                        pd.Series(lr_forecast, index=valid.index)))
    except:
        pass

    # ----------------------------
    # Prophet
    if PROPHET_AVAILABLE:
        try:
            df_train = train.reset_index().rename(columns={"data": "ds", "valor": "y"})
            df_train.columns = ["ds", "y"]
            df_valid = valid.reset_index().rename(columns={"data": "ds", "valor": "y"})
            df_valid.columns = ["ds", "y"]

            model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            model_prophet.fit(df_train)

            future = model_prophet.make_future_dataframe(periods=len(valid), freq="MS")
            forecast_prophet = model_prophet.predict(future).iloc[-len(valid):]["yhat"].values

            results.append(("Prophet", {}, mean_absolute_percentage_error(valid, forecast_prophet),
                            pd.Series(forecast_prophet, index=valid.index)))
        except:
            pass

    # ----------------------------
    # Seleciona o melhor modelo
    best_model = min(results, key=lambda x: x[2])
    nome_modelo, melhores_params, _, _ = best_model

    # Treinamento final no dataset completo
    future_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")

    if nome_modelo == "Naive":
        forecast_final = pd.Series([series.iloc[-1]] * horizon, index=future_index)
    elif nome_modelo == "MeanNaive":
        forecast_final = pd.Series([series.mean()] * horizon, index=future_index)
    elif nome_modelo == "SeasonalNaive":
        season_vals = list(series.iloc[-12:].values)
        reps = int(np.ceil(horizon / 12))
        forecast_final = pd.Series((season_vals * reps)[:horizon], index=future_index)
    elif nome_modelo == "MovingAverage":
        forecast_final = pd.Series([series.rolling(melhores_params["window"]).mean().iloc[-1]] * horizon, index=future_index)
    elif nome_modelo == "HoltWinters":
        model = ExponentialSmoothing(series,
                                     trend=melhores_params["trend"],
                                     seasonal=melhores_params["seasonal"],
                                     seasonal_periods=melhores_params["seasonal_periods"] if melhores_params["seasonal"] else None)
        fit = model.fit(optimized=True)
        forecast_final = fit.forecast(horizon)
    elif nome_modelo == "SARIMA":
        model = SARIMAX(series,
                        order=(melhores_params["p"], melhores_params["d"], melhores_params["q"]),
                        seasonal_order=(melhores_params["P"], melhores_params["D"], melhores_params["Q"], melhores_params["m"]),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast_final = fit.forecast(horizon)
    elif nome_modelo == "Theta":
        theta_model = ThetaModel(series, period=12)
        fit = theta_model.fit()
        forecast_final = fit.forecast(horizon)
    elif nome_modelo == "LinearTrend":
        X_full = np.arange(len(series)).reshape(-1, 1)
        y_full = series.values
        reg = LinearRegression()
        reg.fit(X_full, y_full)
        X_future = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
        forecast_final = pd.Series(reg.predict(X_future), index=future_index)
    elif nome_modelo == "Prophet" and PROPHET_AVAILABLE:
        df_series = series.reset_index().rename(columns={"data": "ds", "valor": "y"})
        df_series.columns = ["ds", "y"]
        model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        model_prophet.fit(df_series)
        future = model_prophet.make_future_dataframe(periods=horizon, freq="MS")
        forecast_prophet = model_prophet.predict(future).iloc[-horizon:]["yhat"].values
        forecast_final = pd.Series(forecast_prophet, index=future_index)

    return {
        "melhor_modelo": nome_modelo,
        "melhores_parametros": melhores_params,
        "forecast": forecast_final
    }
