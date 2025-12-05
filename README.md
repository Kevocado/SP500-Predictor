# Prediction Market Edge Finder

**Prediction Market Edge Finder** is a professional-grade analytics dashboard designed to identify statistical edges in prediction markets (specifically Kalshi) for assets like SPX, Nasdaq, BTC, and ETH.

It combines real-time market data, AI-powered probability models (LightGBM), and a "Bloomberg Terminal" style interface to help traders find undervalued opportunities where the model's calculated probability significantly diverges from the market's implied probability.

## ⚡ Key Features

-   **Real-Time Market Scanner**: Automatically fetches and categorizes live Kalshi markets into "Hourly", "End of Day", and "Range" opportunities.
-   **AI-Driven Probability**:
    -   **Hourly Models**: LightGBM regressors trained on 1-minute intraday data to predict the next hour's closing price.
    -   **Daily Models**: Models trained on hourly data for end-of-day predictions.
    -   **Auto-Retraining**: System automatically detects feature mismatches or concept drift and retrains models on the fly.
-   **Opportunity Detection**:
    -   **Edge Calculation**: `Edge = Model Probability - Market Cost`. Positive edge indicates a +EV trade.
    -   **Moneyness Filtering**: Filters out "junk trades" to focus on competitive strike prices.
    -   **Alpha Deck**: Highlights top "Alpha Picks" with the highest statistical edge.
-   **Advanced UI/UX**:
    -   Dark-mode "Bloomberg" aesthetic.
    -   Quick asset switching via top navigation pills.
    -   Integrated PnL simulator to estimate potential returns based on wager size.
    -   Live market context (VIX, 10Y Yield, BTC Volume).

## 🏗 Architecture

-   **Frontend**: [Streamlit](https://streamlit.io/) for the interactive dashboard.
-   **Modeling**: [LightGBM](https://lightgbm.readthedocs.io/) for fast, efficient gradient boosting on time-series data.
-   **Data Sources**:
    -   **YFinance**: Real-time price history for SPX, QQQ, BTC-USD, ETH-USD.
    -   **Kalshi API**: Live prediction market data (bids, asks, expirations).
-   **Backend Logic**: `src/` modules handle data ingestion, feature engineering, and inference.

## 🚀 Setup & Installation

### Prerequisites
-   Python 3.9+
-   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory (see `.env.example` if available, or use the format below):
    ```env
    # Required for Kalshi API access
    KALSHI_API_KEY=your_kalshi_api_key_here
    
    # Optional: Azure Connection String if using cloud logging
    # AZURE_CONNECTION_STRING=...
    ```

## 🖥️ Usage

1.  **Run the Application:**
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Using the Dashboard:**
    -   **Select Asset**: Click "SPX", "Nasdaq", "BTC", or "ETH" at the top.
    -   **View Opportunities**: Check the "Hourly" or "End of Day" tabs for trade ideas.
    -   **Analyze Edge**: Look for high "Edge" values (Green bars).
        -   **Green**: Model Confidence > Market Price (Buy Signal).
        -   **Red**: Model Confidence < Market Price (Avoid or Short).
    -   **PnL Calculator**: Enter your wager amount in the sidebar to see potential profit for each trade card.

## 📂 Project Structure

```
.
├── streamlit_app.py        # Main Dashboard Entry Point
├── README.md               # Project Documentation
├── requirements.txt        # Python Dependencies
├── .env                    # Environment Variables (Gitignored)
├── src/
│   ├── data_loader.py      # YFinance Data Fetching
│   ├── kalshi_feed.py      # Kalshi API Integration
│   ├── feature_engineering.py # Technical Indicators (RSI, MACD, etc.)
│   ├── model.py            # LightGBM Hourly Model Logic
│   ├── model_daily.py      # Daily Model Logic
│   └── utils.py            # Helper Functions
├── model/                  # Saved .pkl models (Gitignored)
└── scripts/                # Utility scripts
```

## ⚠️ Disclaimer
This tool is for informational and educational purposes only. Prediction markets are high-risk. The "Edge" and "Probability" metrics are model outputs and do not guarantee future results. Trade at your own risk.
