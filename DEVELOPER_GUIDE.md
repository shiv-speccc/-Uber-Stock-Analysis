# 🛠️ Developer Guide — VS Code & GitHub Setup

Complete step-by-step instructions to run the project locally and publish it to GitHub.

---

## Part 1 — VS Code Local Setup

### Step 1: Install Python 3.12
1. Go to https://www.python.org/downloads/
2. Download **Python 3.12.x** for your OS
3. During install → ✅ check **"Add Python to PATH"**
4. Verify: open terminal → `python --version` → should show `Python 3.12.x`

### Step 2: Install VS Code
1. Download from https://code.visualstudio.com/
2. Install these extensions:
   - **Python** (by Microsoft)
   - **Pylance**
   - **Jupyter** (for notebook support)

### Step 3: Clone and open the project
```bash
# Clone
git clone https://github.com/yourusername/uber-stock-analysis.git
cd uber-stock-analysis

# Open in VS Code
code .
```

### Step 4: Create and activate virtual environment
```bash
# Create (do this once)
python -m venv venv

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate — Windows (CMD)
venv\Scripts\activate.bat

# Activate — macOS / Linux
source venv/bin/activate

# Confirm: you should see (venv) prefix in terminal
```

### Step 5: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> ⏱ This takes ~3–5 minutes (Prophet installs Stan compiler)

### Step 6: Add the dataset
```
Place uber_stock_data.csv in:  data/raw/uber_stock_data.csv
```

### Step 7: Run the full pipeline
```bash
# Full run (trains models + generates all plots)
python main.py

# Skip retraining (uses saved models from models/)
python main.py --skip-train

# Only regenerate plots
python main.py --plots-only
```

### Step 8: Open the Jupyter Notebook
```bash
jupyter notebook notebooks/Uber_Stock_Analysis_Complete.ipynb
```
Or in VS Code: open the `.ipynb` file directly and select the `venv` kernel.

### Step 9: Check outputs
```
outputs/plots/         → 10 PNG charts
outputs/reports/       → JSON evaluation report + CSV forecasts
models/                → Saved ARIMA and Prophet models
logs/pipeline.log      → Full execution log
```

---

## Part 2 — GitHub Repository Setup

### Step 1: Create a new GitHub repository
1. Go to https://github.com → **New repository**
2. Repository name: `uber-stock-analysis`
3. Description: `End-to-end time series EDA and forecasting of Uber (UBER) stock using ARIMA and Facebook Prophet`
4. Set to **Public**
5. Do NOT initialise with README (we already have one)
6. Click **Create repository**

### Step 2: Initialise Git locally
```bash
cd uber-stock-analysis   # make sure you're in the project root
git init
git add .
git commit -m "Initial commit: Uber Stock Analysis & Forecasting project"
```

### Step 3: Connect to GitHub and push
```bash
git remote add origin https://github.com/yourusername/uber-stock-analysis.git
git branch -M main
git push -u origin main
```

### Step 4: Add screenshots to README
1. Create a `screenshots/` folder in the project root
2. Copy your favourite 3–4 plots from `outputs/plots/` into `screenshots/`
3. Add them to README.md:
```markdown
![Closing Price](screenshots/01_closing_price_trend.png)
![Prophet Forecast](screenshots/09_prophet_forecast.png)
![Model Comparison](screenshots/10_model_comparison.png)
```
4. Commit and push:
```bash
git add screenshots/ README.md
git commit -m "Add screenshots to README"
git push
```

### Step 5: Add repository description and topics
On GitHub repository page:
- Click ⚙️ gear icon next to **About**
- Description: `End-to-end ARIMA & Facebook Prophet stock price forecasting on Uber (UBER) NYSE data (2019-2025) with EDA, visualisations, and 180-day future projections.`
- Website: (leave blank or add your portfolio URL)
- Topics: `python`, `time-series`, `stock-analysis`, `arima`, `prophet`, `forecasting`, `data-science`, `eda`, `matplotlib`, `pandas`

### Step 6: Create a release (optional but impressive)
1. On GitHub → **Releases** → **Create a new release**
2. Tag: `v1.0.0`
3. Title: `v1.0.0 — Initial Release`
4. Description: paste the README overview section

---

## Part 3 — Career Assets

### Resume Bullet Points (ATS-optimised)

```
• Built end-to-end time series forecasting pipeline on 1,444 days of Uber (UBER) NYSE stock data 
  using Python, ARIMA, and Facebook Prophet; achieved 6.88% MAPE with Prophet on 289-row test set

• Performed comprehensive EDA including rolling volatility analysis, MA crossover detection, 
  trading volume correlation, and monthly seasonality decomposition using Pandas, Matplotlib, and Plotly

• Engineered 9 derived features (daily returns, log returns, 30/90-day MAs, intraday range, 
  rolling volatility) and validated stationarity using Augmented Dickey-Fuller (ADF) test

• Automated ARIMA hyperparameter selection via pmdarima auto_arima (AIC minimisation); 
  selected ARIMA(2,1,2) with AIC 3,708 and confirmed white-noise residuals (Ljung-Box p=0.47)

• Compared two forecasting models across RMSE, MAE, MAPE, and R² metrics; Prophet outperformed 
  ARIMA with 39% lower RMSE (6.15 vs 10.15) and generated 180-day price projections to Aug 2025

• Built modular, production-quality Python codebase with config.yaml, logging, error handling, 
  model persistence (joblib/pickle), and 10 publication-ready visualisations
```

### LinkedIn Project Description

```
📈 Uber Stock Price Analysis & Forecasting (2019–2025)

Developed a complete end-to-end time series analysis and forecasting project on Uber 
Technologies (NYSE: UBER) covering 5.75 years of daily trading data from its IPO through 
February 2025.

🔍 What I built:
• Automated preprocessing pipeline with forward-fill imputation and 9 engineered features
• Full EDA: trend analysis, volatility regimes, volume spikes, MA crossovers, return distributions
• Dual forecasting: ARIMA(2,1,2) via stepwise AIC search + Facebook Prophet with 
  daily/weekly/yearly seasonality
• Rigorous evaluation: RMSE, MAE, MAPE, R² on a held-out test set (Dec 2023 → Feb 2025)
• 180-day future price projections with 95% confidence intervals
• 10 production-quality charts and a fully interactive Jupyter notebook

📊 Key result: Prophet outperformed ARIMA on all metrics (RMSE: 6.15 vs 10.15; MAPE: 6.88% vs 11.63%), 
projecting Uber stock at approximately $79 by August 2025.

🛠 Stack: Python 3.12 | Pandas | NumPy | Matplotlib | Seaborn | Plotly | Statsmodels | 
pmdarima | Facebook Prophet | scikit-learn

#DataScience #TimeSeries #StockForecasting #Python #MachineLearning #ARIMA #Prophet
```

### GitHub Repository Description (one-line)
```
End-to-end ARIMA & Facebook Prophet forecasting on Uber (UBER) NYSE stock (2019–2025): EDA, 
10 visualisations, model comparison, and 180-day price projections. Python 3.12.
```

### Skills Demonstrated
`Time Series Analysis` · `ARIMA` · `Facebook Prophet` · `Statsmodels` · `Pandas` · `NumPy` · 
`Matplotlib` · `Seaborn` · `Plotly` · `scikit-learn` · `EDA` · `Feature Engineering` · 
`Model Evaluation` · `Data Preprocessing` · `Jupyter Notebook` · `Python 3.12` · 
`Modular Code Design` · `Logging` · `Configuration Management` · `Git/GitHub`

---

## Part 4 — Interview Q&A

**Q1: Why did Prophet outperform ARIMA on this dataset?**  
A: Uber's price history contains several nonlinear structural breaks — the COVID crash (2020), 
the post-2022 recovery, and the 2024 ATH run. ARIMA assumes linearity and stationarity after 
differencing, which limits its ability to model these shifts. Prophet's changepoint detection 
algorithm automatically identifies and fits these regime changes using a piecewise linear trend, 
giving it a fundamental advantage. It also captures yearly seasonality (earnings cycles) which 
ARIMA ignores.

**Q2: What did the ADF test tell you, and what did you do about it?**  
A: The ADF test on the training series returned a p-value of 0.46, failing to reject the null 
hypothesis of a unit root — confirming the series is non-stationary. This is expected for 
stock prices. I passed this to auto_arima which applied first-order differencing (d=1), 
transforming price levels to daily price changes. After differencing, the series became stationary 
and suitable for ARIMA modelling.

**Q3: How did you prevent data leakage in your train/test split?**  
A: I used a strict chronological split — the most recent 20% of data (Dec 2023 to Feb 2025) was 
held out as the test set, and no data from this period was used during training, feature 
engineering, or model selection. All rolling averages and lag features were computed strictly 
forward in time (pandas rolling with min_periods=1) to avoid look-ahead bias.

**Q4: How would you extend this project to improve accuracy?**  
A: Several directions: (1) LSTM/GRU deep learning to capture long-range sequential dependencies 
in price time series; (2) add exogenous regressors to Prophet — Uber's quarterly earnings, 
oil prices (key input cost), or competitor stock performance; (3) ensemble the ARIMA and Prophet 
predictions with learned weights; (4) use XGBoost with lag features, calendar features, and 
technical indicators (RSI, MACD) for a third model comparison.

**Q5: What does negative R² mean in your evaluation?**  
A: R² is negative when the model performs worse than a naive baseline that always predicts the 
mean of the test set. For our ARIMA (R²=-1.97), this happened because ARIMA essentially 
predicted a flat line (close to the last known value) while the actual test set showed significant 
upward movement during 2024. Prophet (R²=-0.09) was much closer to 0, meaning it nearly matched 
baseline performance on this difficult test window. Negative R² is common in stock forecasting 
when the test period exhibits a strong directional trend not present in the training data.

**Q6: Walk me through your data preprocessing steps.**  
A: I followed six steps: (1) loaded the CSV and parsed the Date column to datetime, setting it 
as the index; (2) validated all 7 required columns were present; (3) applied forward-fill to 
any missing Volume values to preserve temporal continuity; (4) cast all price columns to float 
and removed rows where High < Low (data quality check); (5) removed duplicate date entries; 
(6) engineered 9 derived features including daily percentage returns, log returns, 30/90-day 
rolling moving averages, intraday range, 30-day rolling volatility, and calendar decomposition.
```
