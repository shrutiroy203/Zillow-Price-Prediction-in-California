# Zillow-Price-Prediction-in-California

The Zillow dataset (modified) recorded Feb 2008- Dec 2015 monthly median sold price forhousing in California, Feb 2008-Dec 2016 monthly median mortgage rate, and Feb 2008-Dec2016 monthly unemployment rate.

The dataset we analyzed contains data from Zillow. This dataset has 3 different variables: MedianSoldPrice_AllHomes,MedianMortageRate,UnemploymentRate

We use the following methods:

1.**Autoregressive Integrated Moving Average Model (ARIMA)**: Time series model that uses differenced lagged
values and lagged prediction errors as inputs to make our prediction of our target variable, median price.

2.**Seasonal Autoregressive Integrated Moving Average Model (SARIMA)**: An ARIMA model with a
seasonality component taken into account. The model is formed by including additional seasonal terms that
involve backshifts of the seasonal periods.

3.**Univariate Prophet Model:** Additive time series model from Facebook that decomposes a series into trend,
yearly, weekly, and daily seasonality and holiday effects.

4.**Univariate Long Short-Term Memory Model:** Inspired by biological neural networks, artificial neural networks
are layered structures of connected perceptrons. A recurrent neural network is specific to temporal data which
retains information for a short period of time. LSTM is a particular artificial recurrent neural network that
maintains feedback connections which allows it to store information over a long period of time.

5.**Exponential Smoothing:** A univariate time series method that consists solely of a trend component, a seasonal
component and error. The forecasts are weighted averages of prior values where the weights decay exponentially.

6.**VAR Model** : A multivariate time series statistical model used to capture the relationship between multiple
quantities as they change over time. VAR is a type of stochastic process model. VAR models generalize the
single-variable autoregressive model by allowing for multivariate time series

7.**SARIMAX Multivariate Model:** Sarimax multivariate version of SARIMAX just adds exogenous
variables.These are parallel time series variates that are not modeled directly via AR, I, or MA processes, but are
made available as a weighted input to the mode

8.**Multivariate Long Short-Term Memory Model:** A multivariate LSTM model takes as input both exogenous
variables and historical endogenous variables. The model predicts with endogenous variables. Multivariate
LSTM could capture more information than Univariate LSTM. As a result, it could lead to overfitting
