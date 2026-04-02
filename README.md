# Grid Frequency Micro-Signals for Predicting German Electricity Imbalance Prices

## Overview

Electricity systems must continuously balance supply and demand. When imbalances occur, the system operator settles the difference using **imbalance prices (reBAP)**. These prices can experience extreme spikes during periods of grid stress.

This project investigates whether **grid frequency micro-signals** contain predictive information about imbalance price behaviour in the German electricity market.

Grid frequency reflects the instantaneous balance between electricity supply and demand. Small deviations in frequency may therefore act as early indicators of system stress before imbalance prices react.

The goal of this project is to explore whether frequency-derived signals can help predict:

- imbalance price levels
- price spikes
- volatility regimes

using machine learning techniques.

---

## Data

The dataset combines:

- **German electricity imbalance prices (reBAP)**
- **Grid frequency measurements**
- Aggregated market features derived from these signals

The data is structured as a time-series dataset covering multiple years of observations.

Example features include:

- frequency deviation
- rate of change of frequency (RoCoF)
- lagged price values
- rolling volatility
- engineered features derived from grid behaviour

---

## Methodology

Three modelling tasks were explored.

### 1. Price Regression

A regression model predicts the imbalance price directly.

Models used:

- baseline time-series predictors
- gradient boosted trees (XGBoost)

Performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

### 2. Price Spike Classification

Extreme imbalance price events are rare but economically significant.

This task reframes the problem as a classification problem:

- **Normal prices**
- **Spike events**

Machine learning models are trained to detect conditions that precede spikes.

Evaluation metrics include:

- precision
- recall
- classification accuracy

---

### 3. Return Regression

A regression model predicts short-term price returns rather than raw price levels.

This formulation focuses on relative movement and aims to capture short-term changes in market behaviour more directly than price-level forecasting.

Performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

### 4. Return Spike Classification

Since extreme return movements may provide a clearer representation of abrupt market stress than raw price levels, a separate classification model is used to identify **return spikes**.

This task classifies whether the return over a given period exceeds a defined spike threshold.

Evaluation metrics include:

- precision
- recall
- F1-score
- classification accuracy

---

### 5. Volatility Regime Classification

Imbalance prices exhibit periods of different volatility regimes.

This model attempts to classify the market state into categories such as:

- low volatility
- medium volatility
- high volatility

This helps analyse how frequency signals relate to different market conditions.

---

## Results

Key findings include:

- Machine learning models outperform simple baseline forecasting methods.
- Frequency-derived features provide limited improvement for direct price prediction.
- Frequency signals show stronger relationships with volatility regimes and system stress events.
- Extreme price spikes remain difficult to predict due to their rarity.

These results suggest that grid frequency may be more useful for **detecting market stress conditions** than for precise price forecasting.

---

## Repository Structure

```
Grid Frequency Data/    Zipped grid frequency (1 second resolution)
Aggregated Data/        reBAP price data and 15-minute aligned grid/price data
models/                 Saved model files & risk analysis dashboard
Result Tables/          Excel spreadsheet with model metrics
Scripts/                All python scripts including data collection/cleaning and experiments
```

---

## Technologies Used

- Python  
- pandas  
- numpy  
- scikit-learn  
- XGBoost  
- matplotlib / seaborn  

---

## Future Work

Potential extensions include:

- incorporating additional grid stability indicators
- analysing higher frequency signals
- improving rare event detection for price spikes
- integrating additional market or generation data

---

## Author
Diogo Fernandes  

<img width="462" height="690" alt="image" src="https://github.com/user-attachments/assets/f16d366f-bd2f-41f7-a4a9-e3663a2fa7b9" />
