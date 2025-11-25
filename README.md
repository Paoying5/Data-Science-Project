# Air quality prediction website using LSTM (Vietnam)

A technical, end-to-end README covering the full build—from data acquisition to deployed dashboard—for PM2.5 forecasting and AQI analytics in Vietnam.

---

## Overview
<img width="1000" height="667" alt="image" src="https://github.com/user-attachments/assets/bdafad19-bf76-4dae-b901-fc4d2b33706e" />

This project designs and builds a web system to monitor, analyze, and forecast air quality across Vietnam using time-series deep learning (LSTM) and statistical baselines (ARIMA). It ingests weather and air pollution signals, performs rigorous EDA, trains forecasting and classification models, and exposes predictions via a Flask web API and interactive dashboard.

- **Scope:** 63 provinces, daily/hourly signals from 2023-11-14 to 2024-11-14.
- **Targets:** PM2.5 forecasting; AQI level classification; comparative ARIMA baseline.
- **Key results:** LSTM shows strong fit on test data; AQI Random Forest reaches perfect test accuracy; ARIMA baseline underfits complex dynamics.

---

## Data and pipeline

#### Data sources and schema

- **OpenWeather API:** Weather and air-pollution endpoints; rotated API keys to avoid rate limits.
- **Signals captured:**
  - **Location:** Province, Latitude, Longitude
  - **Datetime:** Date
  - **Weather:** Temperature, Humidity, Pressure, Wind Speed, Weather Description
  - **Pollutants:** CO, NO, NO2, O3, SO2, PM2.5, PM10, NH3
  - **Indices:** AQI
- **Artifact:** merged_weather_air_pollution.csv (clean, integrated dataset aligned on Province/Date/coords).

#### ETL pipeline (A → Z)

- **Collection:**  
  - **Rotation:** Cycle through multiple OpenWeather API keys on 429 responses.  
  - **Granularity:** Province groups to pace calls; 30-second inter-group delay.
- **Integration:**  
  - **Join keys:** Province, Date, Latitude, Longitude; unify weather + pollution tables into one DataFrame.
- **Cleaning:**  
  - **Missing values:** Checked; dataset shows no NaN across numeric weather/pollution columns.  
  - **Typing:** Numeric columns as float64; text as object (Province, Weather Description).
- **Validation:**  
  - **Dataset size:** 26,057 rows, 18 columns; index 0..26056; all non-null confirmed.  
  - **Sampling:** Random row inspection verifies plausibility and distribution ranges.
<img width="718" height="355" alt="image" src="https://github.com/user-attachments/assets/f04666d0-dbb6-45af-82b3-ddc7a5039f19" />


#### Exploratory data analysis (EDA)

- **Correlation heatmaps:**  
  - **Strong:** PM2.5 ⇄ PM10; CO moderately correlates with PM metrics.  
  - **Co-occurrence:** NO2 and SO2 often rise together (industrial/traffic signature).
- **Distribution analyses:**  
  - **Right-skew:** Most pollutant histograms skew right (many low, few high extremes).  
  - **Multimodal:** O3 and NH3 show multi-peak behavior.  
  - **Event rarity:** PM2.5 extreme events (>100) are rare but impactful.
- **Scatter/Pair plots:**  
  - **AQI banding:** AQI appears in discrete levels (≈2,3,4,5) vs PM2.5 ranges.  
  - **PM2.5 vs PM10:** Strong linear relation up to PM2.5 ≈ 100.  
  - **NO vs NO2:** Positive relation at low ranges; higher variance at extremes.

---

## Modeling and evaluation

#### LSTM forecasting (PM2.5)
<img width="761" height="179" alt="image" src="https://github.com/user-attachments/assets/59010407-9e91-4a1c-a4a4-67053e51eba2" />


- **Preprocessing:**  
  - **Scaler:** MinMaxScaler into [0, 1].  
  - **Lookback:** 60 time steps (sequence length).
- **Dataset shaping:**  
  - **Train/Test split:** 80%/20%.  
  - **Shapes:** X_train = (20,797, 60, 1), X_test = (5,200, 60, 1).
- **Architecture:**  
  - **Layers:** 3 × LSTM (50 units each).  
  - **Sequence flags:** return_sequences=True for first two layers; False for last.  
  - **Regularization:** Dropout=0.2 between LSTM layers.  
  - **Head:** Dense(1) regression output.  
  - **Loss/Optimizer:** MSE + Adam.
- **Training:**  
  - **EarlyStopping:** monitor val_loss, patience=5, restore_best_weights=True.  
  - **Convergence:** Training and validation loss both decrease; minimal gap (no overfitting).
- **Metrics and fit:**  
  - **RMSE (test):** 0.295 (on inverse-scaled predictions).  
  - **R² (scatter fit):** 0.99 alignment vs ground truth.
<img width="631" height="331" alt="image" src="https://github.com/user-attachments/assets/40ebf5b3-51ae-4b37-8da9-839d33bc0e59" />
<img width="434" height="341" alt="image" src="https://github.com/user-attachments/assets/6ea9bea2-ea5c-4155-ba49-0819f736c56d" />


#### ARIMA baseline (PM2.5)

- **Config:** ARIMA(p=5, d=1, q=0) over PM2.5 series.  
- **Split:** 80% train; 20% test.  
- **Performance:** RMSE ≈ 35.47; forecast path near-flat vs volatile actuals (underfits nonlinearity and event spikes).
<img width="599" height="280" alt="image" src="https://github.com/user-attachments/assets/ed187f32-d7d5-4e69-ae13-ed633e9a15bf" />


#### AQI classification (Random Forest)

<img width="444" height="343" alt="image" src="https://github.com/user-attachments/assets/87770dd6-0df6-455d-8bab-e2cc572dd2e5" />


- **Target:** AQI categorical levels (e.g., Good, Moderate, Unhealthy).  
- **Result:** Accuracy = 1.00; F1-score = 1.00 across classes; confusion matrix shows zero misclassifications.  
- **Note:** Imbalanced distribution—Unhealthy is under-represented; handle with care for real-world deployment.

---

## Web app and API

#### Flask application

<img width="761" height="339" alt="image" src="https://github.com/user-attachments/assets/59e127ae-ebfe-4c1a-90ba-005cbfb70990" />


- **Model serving:** Load trained LSTM model and scaler at startup.
- **Routes:**
  - **“/”:**  
    - **Label:** Dashboard home.  
    - **Action:** Reads CSV; shows most recent 30 days; renders charts and tables.
  - **“/get_data/<int:num_days>”:**  
    - **Label:** Data access.  
    - **Action:** Return last num_days observations as JSON.
  - **“/predict/lstm”:**  
    - **Label:** PM2.5 forecast API.  
    - **Input:** num_days (window).  
    - **Action:** Normalize → reshape → LSTM predict → inverse-scale; returns forecast series and summary.
- **Dashboard features:**  
  - **PM2.5 forecast plot:** Hourly/daily curve; example peak ≈ 103.48 μg/m³ “Khống tốt cho sức khỏe”.  
  - **History log:** Timestamped predictions and health status.  
  - **AQI distribution bar chart; classification confusion matrix.**

---

## Reproducibility and setup

#### Tech stack

| Component | Purpose |
|---|---|
| Python (3.10+) | Core language |
| Flask | Web API and dashboard |
| TensorFlow/Keras | LSTM modeling |
| scikit-learn | Scaling, Random Forest, metrics |
| Pandas/NumPy | Data wrangling |
| Matplotlib/Seaborn | Visualization |

#### Prerequisites

- **Python:** 3.10+ recommended.
- **API keys:** OpenWeather—provide multiple keys for rotation.

#### Environment variables

- **OPENWEATHER_API_KEYS:** Comma-separated list of keys used for rotation.

Example .env:
```
OPENWEATHER_API_KEYS=key1,key2,key3
```

#### Installation

```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Data collection

- **Run collector:**  
  - **Label:** Fetch weather + air pollution for all provinces and date range.  
  - **Output:** weather_data.csv, air_pollution_data.csv, merged_weather_air_pollution.csv.

```
python scripts/collect_openweather.py --start 2023-11-14 --end 2024-11-14
python scripts/merge_datasets.py
```

#### Training

- **LSTM PM2.5:**
```
python scripts/train_lstm_pm25.py --data data/merged_weather_air_pollution.csv \
  --lookback 60 --model out/lstm_pm25.h5 --scaler out/scaler.pkl
```

- **ARIMA PM2.5:**
```
python scripts/train_arima_pm25.py --data data/merged_weather_air_pollution.csv \
  --order 5 1 0 --model out/arima_pm25.pkl
```

- **AQI classifier:**
```
python scripts/train_rf_aqi.py --data data/merged_weather_air_pollution.csv \
  --model out/rf_aqi.pkl
```

#### Running the web app

```
FLASK_APP=app.py FLASK_ENV=development flask run
# or
python app.py
```

- **Access:** http://localhost:5000  
- **Configured paths:** Ensure app loads out/lstm_pm25.h5 and out/scaler.pkl.

#### Suggested repo structure

```
.
├─ app.py
├─ requirements.txt
├─ data/
│  ├─ weather_data.csv
│  ├─ air_pollution_data.csv
│  └─ merged_weather_air_pollution.csv
├─ out/
│  ├─ lstm_pm25.h5
│  ├─ scaler.pkl
│  ├─ arima_pm25.pkl
│  └─ rf_aqi.pkl
├─ scripts/
│  ├─ collect_openweather.py
│  ├─ merge_datasets.py
│  ├─ train_lstm_pm25.py
│  ├─ train_arima_pm25.py
│  └─ train_rf_aqi.py
└─ notebooks/
   ├─ eda.ipynb
   └─ model_evaluation.ipynb
```

---

## Results and roadmap

#### Key outcomes

- **LSTM PM2.5:**  
  - **Fit quality:** RMSE ≈ 0.295; **R² ≈ 0.99**; prediction lines closely follow actuals.
- **ARIMA baseline:**  
  - **Underfit:** RMSE ≈ 35.47; forecast nearly flat vs volatile ground truth.
- **AQI Random Forest:**  
  - **Classification:** Accuracy/F1 = 1.00; note class imbalance for Unhealthy.

#### Limitations

- **Scope:** Focus on PM2.5 forecasting; PM10/NO2/SO2 forecasting not yet included.  
- **Horizon:** Forecast window limited to ~24 hours.  
- **Uncertainty:** No confidence intervals on forecasts.  
- **Imbalance:** AQI class distribution skewed.

#### Roadmap

- **Broaden targets:**  
  - **Multivariate forecasts:** PM10, NO2, SO2, O3 as separate or joint targets.
- **Longer horizons:**  
  - **48–168 hours** with seq-to-seq models (e.g., stacked LSTM/GRU, TCN, Transformer).
- **Uncertainty quantification:**  
  - **Intervals:** Quantile regression, MC Dropout, or conformal prediction on LSTM outputs.
- **Data enrichment:**  
  - **External features:** Traffic, emission inventories, satellite AOD, seasonal indices.  
  - **Spatial modeling:** Geo-features and graph-based models to capture regional spillovers.
- **Product features:**  
  - **Mobile-ready UI, notifications, multilingual support, health advisories** per AQI band.

---

If you want, I can tailor the README to your actual file paths, script names, and add badges (Python, Flask, TensorFlow, Made in Vietnam) plus CI instructions for a fully “production-grade” presentation.
