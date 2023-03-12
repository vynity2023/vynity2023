import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import csv
import math
def future_stock_price(days, prices, model):
    '''
    Calculates future 1 day stock price based on provided data

    Parameters:
    days (list): List of integers representing days
    prices (list): List of floats representing stock prices on each day

    Returns:
    float: Future 1 day stock price
    '''

    # Reshape the data for scikit-learn
    X = pd.DataFrame(days).values.reshape(-1, 1)
    y = pd.to_numeric(pd.DataFrame(prices).stack(), errors='coerce').values.ravel()

    # Fit a model
    model.fit(X, y)

    # Calculate the future 1 day stock price using the linear regression model
    future_day = days[-1] + 1
    future_price = model.predict([[future_day]])[0]
    return future_price


def predict_stock_prices(file_path, model):
    '''
    Predicts next 30 days stock prices based on data in CSV file

    Parameters:
    file_path (str): File path of CSV file containing data

    Returns:
    list: List of floats representing predicted stock prices for next 30 days
    '''

    # Read the data from the CSV file
    data = pd.read_csv(file_path)

    # Convert the data to numeric values
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Date'] = pd.to_numeric(data['Date'], errors='coerce')

    # Drop any rows with NaN
    data.dropna(subset=['Close'], inplace=True)

    # Extract the days and prices
    days = data['Date'].tolist()
    prices = data['Close'].tolist()

    # Initialize the count and predicted prices lists
    count = 0
    predicted_prices = prices.copy()

    # Predict the next 30 days stock prices
    while count < 30:
        future_price = future_stock_price(days, predicted_prices, model)
        predicted_prices.append(future_price)
        days.append(days[-1] + 1)
        count += 1

    return predicted_prices



# Choose the best model
model = RandomForestRegressor() # For example

# Specify the file path of the data
file_path = 'btc-file-for-training.csv'

# Predict the next 30 days stock prices using the selected model
predicted_prices = predict_stock_prices(file_path, model)




predict_list = predicted_prices[-30:]

with open('real future 30.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    real_price_list = [float(row['Close']) for row in reader]

# Calculate MAPE for comparison
mape = sum(abs((real_price_list[i] - predict_list[i])/real_price_list[i]) for i in range(len(predict_list))) / len(predict_list)

# Print the two lists side by side for comparison, along with MAPE
for i, (real_price, predict_price) in enumerate(zip(real_price_list, predict_list)):
    print(f"Real vs Prediction Day {i}: {real_price:.2f}   {predict_price:.2f}   MAPE: {mape:.2%}")
