# Stock Market Prediction using LSTM

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock market prices.

## Project Overview

The project uses historical stock market data to predict future stock prices using a deep learning approach.

## Steps Breakdown

### 1. Data Preprocessing
- Import required libraries (pandas, numpy, sklearn, matplotlib)
- Load stock market data from CSV file
- Remove NaN values from the dataset
- Scale the data to range (0,1) using MinMaxScaler
- Split data into training (65%) and testing (35%) sets
- Create time series dataset with 100 time steps

### 2. Model Architecture
- Sequential LSTM model with following layers:
  - LSTM layer (50 units, return sequences=True)
  - LSTM layer (50 units, return sequences=True)
  - LSTM layer (50 units)
  - Dense layer (1 unit)
- Compiled with:
  - Loss function: Mean Squared Error
  - Optimizer: Adam

### 3. Training
- Model trained for 100 epochs
- Batch size of 64
- Validation using test data

### 4. Evaluation
- Calculate Root Mean Squared Error (RMSE) for training and testing
- Calculate Mean Absolute Error (MAE)
- Visualize predictions against actual values

### 5. Future Predictions
- Make predictions for next 30 days
- Visualize future predictions

## Key Files
- `Stock Market Prediction LSTM.ipynb`: Main Jupyter notebook containing the code
- `x_train.csv`: Training data file (required for running the model)

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Usage
1. Ensure all required libraries are installed
2. Place your stock market data CSV file in the appropriate location
3. Update the file path in the code
4. Run the notebook cells sequentially

## Model Performance
- Training RMSE: 0.0309 (Root Mean Squared Error for training data)
- Testing RMSE: 0.0205 (Root Mean Squared Error for testing data)
- Testing MAE: 0.0153 (Mean Absolute Error for test predictions)
- The model shows good prediction accuracy with low error rates
- Visualizations demonstrate close alignment between predicted and actual values
- Future predictions are plotted for the next 30 days, showing the model's capability to forecast future trends

## Notes
- The model uses a 100-day window to predict the next day's stock price
- Data scaling is essential for optimal LSTM performance
- The model can be further fine-tuned by adjusting hyperparameters
