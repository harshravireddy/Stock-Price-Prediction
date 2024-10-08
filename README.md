# Stock Price Prediction

## Overview
This project implements a stock price prediction model for Apple Inc. (AAPL) using three different machine learning approaches: Long Short-Term Memory (LSTM) networks, Random Forest regression, and AutoRegressive Integrated Moving Average (ARIMA). The model downloads historical stock price data, preprocesses it, trains the models, evaluates their performance, and visualizes the results.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Results](#results)
- [License](#license)

## Requirements
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `yfinance`
- `matplotlib`
- `scikit-learn`
- `pmdarima`
- `keras`

## Installation
You can install the required libraries using pip. Run the following command:
```bash pip install numpy pandas yfinance matplotlib scikit-learn pmdarima keras ```

## Data Sources
The stock price data used in this project is obtained from Yahoo Finance using the `yfinance` library. The data includes historical stock prices for multiple companies, including:

- **Apple Inc. (AAPL)**
- **Microsoft Corp. (MSFT)**
- **Alphabet Inc. (GOOGL)**
- **Amazon.com Inc. (AMZN)**
- **Meta Platforms Inc. (META)**
- **NVIDIA Corporation (NVDA)**
- **Tesla Inc. (TSLA)**
- **Alibaba Group (BABA)**
- **Baidu Inc. (BIDU)**
- **Bilibili Inc. (BILI)**
- **BioNTech SE (BNTX)**
- **Cisco Systems Inc. (CSCO)**
- **Walt Disney Co. (DIS)**
- **Ford Motor Company (F)**
- **General Electric Company (GE)**
- **General Motors Company (GM)**
- **Goldman Sachs Group Inc. (GS)**
- **International Business Machines Corp. (IBM)**
- **Intel Corporation (INTC)**
- **Johnson & Johnson (JNJ)**
- **JPMorgan Chase & Co. (JPM)**
- **Coca-Cola Company (KO)**
- **Eli Lilly and Company (LLY)**
- **McDonald's Corporation (MCD)**
- **Merck & Co. Inc. (MRK)**
- **Morgan Stanley (MS)**
- **Nike Inc. (NKE)**
- **Pfizer Inc. (PFE)**
- **Procter & Gamble Co. (PG)**
- **AT&T Inc. (T)**
- **Verizon Communications Inc. (VZ)**
- **Walmart Inc. (WMT)**
- **Exxon Mobil Corporation (XOM)**

### Data Characteristics
- **Date Range**: The data spans from January 1, 2014, to January 1, 2024.
- **Data Points**: The dataset includes daily stock prices, including Open, High, Low, Close, Volume, and Adjusted Close prices.

### Data Retrieval
The data is downloaded programmatically using the following code snippet:
```python
import yfinance as yf 

# Define the stock symbols
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 
          'BABA', 'BIDU', 'BILI', 'BNTX', 'CSCO', 'DIS', 'F', 
          'GE', 'GM', 'GS', 'IBM', 'INTC', 'JNJ', 'JPM', 
          'KO', 'LLY', 'MCD', 'MRK', 'MS', 'NKE', 
          'PFE', 'PG', 'T', 'VZ', 'WMT', 'XOM']

# Download stock data
df = yf.download(stocks, start='2014-01-01', end='2024-01-01')

```
## Usage
### Clone this repository

```bash
Copy code
git clone https://github.com/harshravireddy/stock-price-prediction.git
cd stock-price-prediction
```
Run the main script

```
bash
Copy code
python Predicting_stock_price.py
```
## Code Structure

- `Predicting_stock_price.py:` The main Python script containing the code for downloading data, preprocessing, model training, evaluation, and visualization.
- `stock_data.csv:` The CSV file containing the downloaded stock price data (optional, if you choose to save it).

Key Sections in the Code

- **Data Downloading:** Downloads historical stock price data using yfinance.
- **Data Preprocessing:** Handles missing values and calculates technical indicators.
- **Model Training:** Implements LSTM, Random Forest, and ARIMA models for prediction.
- **Model Evaluation:** Calculates RMSE for each model to assess prediction accuracy.
- **Visualization:** Plots actual and predicted stock prices for comparison.

## Model Evaluation

The performance of each model is evaluated using the Root Mean Squared Error (RMSE):

- LSTM RMSE:  `160.575`
- Random Forest RMSE: `10.620`
- ARIMA RMSE: `48.852`

## Visualization

The results of the predictions are visualized in a plot that compares actual prices with predicted prices from the LSTM, Random Forest, and ARIMA models.

## License 

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Additional Notes
 I have predicted only for the apple stock prices.



