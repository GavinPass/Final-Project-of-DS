# 📊 Final Project of Data Science  
## Interactive Macro & Portfolio Analytics Dashboard (Python · Dash · Plotly)

This repository contains my final project for the Applied Data Science course.  
The project implements a fully interactive web-based analytics dashboard that combines:

- **Macro-economic indicators visualization**
- **Portfolio performance analysis**
- **Historical VaR & rolling VaR**
- **Daily portfolio weights & heatmaps**
- **ARIMA-based volatility forecasting**
- **Scenario stress testing using SPX return shocks**

The dashboard is built using **Plotly Dash**, with data processing in pandas, modeling in statsmodels, and modular helper functions defined in `customfunction.py` and `MacroFunction.py`.

## 📁 Project Structure

```
Final-Project-of-DS/
│
├── app.py                   # Main Dash application
├── customfunction.py        # Portfolio analytics & helper functions
├── MacroFunction.py         # Macro data retrieval functions
│
├── SPX_return.csv           # SPX daily returns for stress testing
├── Portfolio_prices.csv     # Sample portfolio dataset
├── ratesdata.xlsx           # Treasury rate data
├── stock_cache.xlsx         # Cached stock data
│
└── README.md                # Documentation
```

