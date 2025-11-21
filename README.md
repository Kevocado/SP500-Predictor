# SP500 Hourly Predictor

This project predicts the next hour's closing price of the S&P 500 (SPY) using intraday minute data and a LightGBM model.

## Project Structure

```
sp500-hourly-predictor/
├── model/                  # Trained model files
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── data_loader.py      # Data fetching (yfinance)
│   ├── feature_engineering.py # Feature pipeline
│   ├── model.py            # Model training and inference
│   └── utils.py            # Utilities
├── streamlit_app.py        # Streamlit dashboard
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd sp500-hourly-predictor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Locally

1.  **Train the model (optional, can be done via UI):**
    You can use the "Retrain Model" button in the Streamlit app, or run a script to train it.

2.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

## Deployment to Streamlit Cloud

1.  Push this code to a GitHub repository.
2.  Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Click "New app".
4.  Select your repository, branch, and the main file path (`streamlit_app.py`).
5.  Click "Deploy".

## Features

-   **Data Fetching:** Real-time intraday data from Yahoo Finance.
-   **Feature Engineering:** Technical indicators (RSI, MACD, Bollinger Bands) and time-based features.
-   **Modeling:** LightGBM regressor for rolling intraday forecasting.
-   **Dashboard:** Interactive UI to view live data and predictions.
