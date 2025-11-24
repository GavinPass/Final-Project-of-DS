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

The dashboard has been deployed to Heroku by clicking this website: 

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

---

## 🖥 How to Run This Project

Running the dashboard locally is very simple.  
Just follow the steps below:

---

### **1. Download the Project (ZIP)**

1. Click the green **Code** button (top-right of the GitHub page)
2. Select **Download ZIP**
3. Unzip the downloaded file  
4. Open the unzipped folder — this folder should contain:
```
app.py
customfunction.py
MacroFunction.py
SPX_return.csv
Portfolio_prices.csv
ratesdata.xlsx
stock_cache.xlsx
requirements.txt
README.md
```

---

### **2. Install the required Python libraries**

Open a terminal **inside the unzipped folder**, then run:

```bash
pip install -r requirements.txt
```

This installs Dash, Plotly, Pandas, Statsmodels, and all other required packages.

---

### **3. Run the application**

```bash
python app.py
```

If successful, the terminal will display:

```
Dash is running on http://127.0.0.1:8050/
```

---

### **4. Open the dashboard in a web browser**

Go to:

```
http://127.0.0.1:8050/
```

You will now be able to explore:

- **Macro Dashboard**  
- **Portfolio Analytics**  
- **ARIMA Risk Forecast**  
- **Stress Testing**

---

### ✔ Notes

- All required datasets are already included in the ZIP.  
- No additional configuration is needed.  
- Full explanation of methodology and results is provided in the separate project report.
