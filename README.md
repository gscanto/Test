
project/
│
├── data/                     # Dados de entrada / saída (brutos e processados)
│   ├── raw/                  # CSV, XLSX originais
│   ├── processed/            # Dados tratados / features
│   └── forecasts/            # Forecasts gerados
│
├── src/
│   ├── config/               # Configurações (paths, parâmetros, API)
│   │   └── settings.py
│   │
│   ├── data_loading/         # Leitura e pré-processamento
│   │   ├── load_vendas.py
│   │   ├── load_producao.py
│   │   ├── load_defeitos.py
│   │   └── load_ativacao.py
│   │
│   ├── forecasting/          # Modelos e pipelines de previsão
│   │   ├── classical.py      # ARIMA, Holt-Winters, etc
│   │   ├── ml.py             # XGBoost, RandomForest, etc
│   │   ├── deep.py           # LSTM, N-BEATS, etc
│   │   └── forecast_runner.py
│   │
│   ├── kpi_calculation/      # Cálculo dos KPIs
│   │   ├── asr.py
│   │   ├── casr.py
│   │   ├── asr_a.py
│   │   ├── casr_a.py
│   │   ├── m2.py
│   │   └── m3.py
│   │
│   ├── utils/                # Funções auxiliares
│   │   ├── date_utils.py
│   │   ├── api_client.py     # Envio de resultados para backend
│   │   └── metrics.py        # Métricas de avaliação dos modelos
│   │
│   └── main.py               # Script principal orquestrador
│
├── requirements.txt
└── README.md
