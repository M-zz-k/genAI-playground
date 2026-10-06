# AR Model (AutoRegressive Model)

This folder contains a Jupyter Notebook (`AR_Model.ipynb`) that demonstrates how to build and evaluate an AutoRegressive (AR) model for time series forecasting using the `statsmodels` library in Python.

## Overview
The notebook performs the following steps:
1. **Data Loading**: Loads weather data (`Weather_data.xlsx`) and parses the dates.
2. **Stationarity Check**: Uses the Augmented Dickey-Fuller (ADF) test to check if the temperature time series is stationary.
3. **Differencing**: Applies first-order differencing to make the data stationary if required.
4. **ACF & PACF Plots**: Plots the AutoCorrelation Function (ACF) and Partial AutoCorrelation Function (PACF) to determine the appropriate lag (`p`) for the AR model.
5. **Model Training**: Splits the data into training (80%) and testing (20%) sets, and fits a `statsmodels.tsa.ar_model.AutoReg` model using the determined lag.
6. **Prediction**: Generates predictions for both the training and testing sets.

## Dependencies
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`
- `statsmodels`

## Troubleshooting
If you encounter a `ValueError` regarding broadcasting shapes when calling `model.fit().summary()`, this is a known `statsmodels` index alignment issue with integer-based Pandas Series. To fix it, either:
- Set your datetime column as the DataFrame's index (`df.set_index("Date", inplace=True)`) before differencing.
- Pass the NumPy array directly to the model (`AutoReg(train.values, lags=p)`).
