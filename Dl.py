# src/forecasting/deep_learning.py
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_percentage_error
from loguru import logger

from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAPE


# -----------------------------
# Pré-processamento
# -----------------------------
def create_sliding_windows(series, window_size):
    """Transforma série em janelas (1 passo à frente)."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)


# -----------------------------
# Modelos Torch
# -----------------------------
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)


class RNNModel(nn.Module):
    def __init__(self, model_type, input_size, hidden_size, num_layers=1):
        super().__init__()
        rnn_cls = nn.LSTM if model_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# -----------------------------
# Treino e Forecast
# -----------------------------
def train_torch_model(model, X_train, y_train, epochs=50, lr=1e-3, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
    return model


def recursive_forecast(model, last_window, horizon, rnn=False):
    """Gera previsão multi-step recursivamente."""
    preds = []
    window = last_window.copy()
    for _ in range(horizon):
        x_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        if rnn:
            x_input = x_input.unsqueeze(-1)  # (batch, seq, features)
        with torch.no_grad():
            y_pred = model(x_input).item()
        preds.append(y_pred)
        window = np.append(window[1:], y_pred)  # shift janela
    return np.array(preds)


# -----------------------------
# Função principal
# -----------------------------
def run_dl_models(series, horizon=12, window_size=24, train_size=0.8, start_date="2000-01-01", freq="M"):
    try:
        series = pd.Series(series).reset_index(drop=True)

        # Casos especiais
        if (series == 0).all():
            logger.warning("Série com todos os valores zero.")
            forecast_idx = pd.date_range(start=start_date, periods=horizon, freq=freq)
            return {"best_model": "AllZero",
                    "forecast": pd.Series(np.zeros(horizon), index=forecast_idx.strftime("%Y-%m-%d")),
                    "mape": 0.0}
        if series.var() < 1e-8:
            logger.warning("Série com variância muito baixa.")
            forecast_idx = pd.date_range(start=start_date, periods=horizon, freq=freq)
            return {"best_model": "LowVarianceMean",
                    "forecast": pd.Series(np.full(horizon, series.mean()), index=forecast_idx.strftime("%Y-%m-%d")),
                    "mape": None}

        # Criar janelas
        X, y = create_sliding_windows(series.values, window_size)
        train_size_abs = int(len(X) * train_size)
        X_train, y_train = X[:train_size_abs], y[:train_size_abs]
        X_valid, y_valid = X[train_size_abs:], y[train_size_abs:]

        results = {}

        # -----------------------------
        # MLP
        # -----------------------------
        def mlp_objective(trial):
            hidden_size = trial.suggest_int("hidden_size", 16, 128)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = MLPModel(window_size, hidden_size)
            model = train_torch_model(model, X_train, y_train, epochs=50, lr=lr)
            preds = [model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item() for x in X_valid]
            return mean_absolute_percentage_error(y_valid, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(mlp_objective, n_trials=10, show_progress_bar=False)
        mlp_best = MLPModel(window_size, study.best_params["hidden_size"])
        mlp_best = train_torch_model(mlp_best, X_train, y_train, epochs=50, lr=study.best_params["lr"])
        mlp_forecast = recursive_forecast(mlp_best, series.values[-window_size:], horizon, rnn=False)
        results["MLP"] = (mlp_best, mean_absolute_percentage_error(series[-horizon:], mlp_forecast), mlp_forecast)

        # -----------------------------
        # LSTM
        # -----------------------------
        def lstm_objective(trial):
            hidden_size = trial.suggest_int("hidden_size", 16, 128)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = RNNModel("LSTM", 1, hidden_size)
            model = train_torch_model(model, X_train[..., None], y_train, epochs=50, lr=lr)
            preds = [model(torch.tensor(x[..., None], dtype=torch.float32).unsqueeze(0)).item() for x in X_valid]
            return mean_absolute_percentage_error(y_valid, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(lstm_objective, n_trials=10, show_progress_bar=False)
        lstm_best = RNNModel("LSTM", 1, study.best_params["hidden_size"])
        lstm_best = train_torch_model(lstm_best, X_train[..., None], y_train, epochs=50, lr=study.best_params["lr"])
        lstm_forecast = recursive_forecast(lstm_best, series.values[-window_size:], horizon, rnn=True)
        results["LSTM"] = (lstm_best, mean_absolute_percentage_error(series[-horizon:], lstm_forecast), lstm_forecast)

        # -----------------------------
        # GRU
        # -----------------------------
        def gru_objective(trial):
            hidden_size = trial.suggest_int("hidden_size", 16, 128)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = RNNModel("GRU", 1, hidden_size)
            model = train_torch_model(model, X_train[..., None], y_train, epochs=50, lr=lr)
            preds = [model(torch.tensor(x[..., None], dtype=torch.float32).unsqueeze(0)).item() for x in X_valid]
            return mean_absolute_percentage_error(y_valid, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(gru_objective, n_trials=10, show_progress_bar=False)
        gru_best = RNNModel("GRU", 1, study.best_params["hidden_size"])
        gru_best = train_torch_model(gru_best, X_train[..., None], y_train, epochs=50, lr=study.best_params["lr"])
        gru_forecast = recursive_forecast(gru_best, series.values[-window_size:], horizon, rnn=True)
        results["GRU"] = (gru_best, mean_absolute_percentage_error(series[-horizon:], gru_forecast), gru_forecast)

        # -----------------------------
        # N-HiTS
        # -----------------------------
        try:
            df_nhits = pd.DataFrame({
                "ds": pd.date_range(start=start_date, periods=len(series), freq=freq),
                "y": series,
                "unique_id": "serie"
            })
            nf = NeuralForecast(models=[NHITS(input_size=window_size, h=horizon, loss=MAPE(), max_steps=500)], freq=freq)
            nf.fit(df_nhits)
            fcst = nf.predict().y.values
            mape_nhits = mean_absolute_percentage_error(series[-horizon:], fcst[-horizon:])
            results["N-HiTS"] = (nf, mape_nhits, fcst[-horizon:])
        except Exception as e:
            logger.warning(f"N-HiTS não treinado: {e}")

        # Escolher melhor modelo
        best_model_name = min(results, key=lambda k: results[k][1])
        forecast_idx = pd.date_range(start=start_date, periods=horizon, freq=freq)
        forecast_series = pd.Series(results[best_model_name][2], index=forecast_idx.strftime("%Y-%m-%d"))

        return {
            "best_model": best_model_name,
            "forecast": forecast_series,
            "mape": results[best_model_name][1]
        }

    except Exception as e:
        logger.error(f"Erro ao rodar modelos de deep learning: {e}")
        raise
