import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import ta
import numpy as np
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot for plotting residuals
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Title and description
st.title("Asset Price Prediction with Multiple Models and Technical Indicators")

# Asset type selection
asset_type = st.selectbox("Select Asset Type:", ["Stocks", "Crypto", "Forex", "Gold"])

# Ticker input based on asset type
if asset_type == "Stocks":
    ticker = st.text_input("Enter a stock ticker:", "AAPL")
elif asset_type == "Crypto":
    ticker = st.text_input("Enter a cryptocurrency ticker:", "BTC-USD")
elif asset_type == "Forex":
    ticker = st.text_input("Enter a forex pair ticker:", "EURUSD=X")
elif asset_type == "Gold":
    ticker = st.text_input("Enter a gold ticker:", "GC=F")

# Load all available data
data = yf.download(ticker, period="max", interval="1d")

# Ensure data is loaded successfully
if not data.empty:
    # Feature Engineering with Technical Indicators
    data['Price Change'] = data['Close'] - data['Open']
    data['High-Low'] = data['High'] - data['Low']
    data['RSI'] = ta.momentum.rsi(data['Close'])
    data['MACD'] = ta.trend.macd(data['Close'])
    data['Bollinger High'] = ta.volatility.bollinger_hband(data['Close'])
    data['Bollinger Low'] = ta.volatility.bollinger_lband(data['Close'])
    data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
    data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
    data['EMA 50'] = ta.trend.ema_indicator(data['Close'], window=50)
    data['EMA 100'] = ta.trend.ema_indicator(data['Close'], window=100)
    data['EMA 200'] = ta.trend.ema_indicator(data['Close'], window=200)
    data['Stochastic'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    data['Target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    # Features and target
    X = data[['Price Change', 'High-Low', 'Volume', 'RSI', 'MACD', 'Bollinger High', 'Bollinger Low', 'SMA', 'EMA', 'EMA 50', 'EMA 100', 'EMA 200', 'Stochastic', 'ATR']]
    y = data['Target']

    # Split data into training and testing sets (67% train, 33% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model performance dictionary
    model_performance = {}

    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    model_performance['Linear Regression'] = {
        'MSE': mean_squared_error(y_test, y_pred_lr),
        'R2': r2_score(y_test, y_pred_lr)
    }
    data['Predicted Close (LR)'] = lr_model.predict(scaler.transform(X))

    # 2. Decision Tree
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train_scaled, y_train)
    y_pred_dt = dt_model.predict(X_test_scaled)
    model_performance['Decision Tree'] = {
        'MSE': mean_squared_error(y_test, y_pred_dt),
        'R2': r2_score(y_test, y_pred_dt)
    }
    data['Predicted Close (DT)'] = dt_model.predict(scaler.transform(X))

    # 3. Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    model_performance['Random Forest'] = {
        'MSE': mean_squared_error(y_test, y_pred_rf),
        'R2': r2_score(y_test, y_pred_rf)
    }
    data['Predicted Close (RF)'] = rf_model.predict(scaler.transform(X))

    # 4. Gradient Boosting
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    y_pred_gb = gb_model.predict(X_test_scaled)
    model_performance['Gradient Boosting'] = {
        'MSE': mean_squared_error(y_test, y_pred_gb),
        'R2': r2_score(y_test, y_pred_gb)
    }
    data['Predicted Close (GB)'] = gb_model.predict(scaler.transform(X))

    # 5. Deep Learning Model
    dl_model = Sequential()
    dl_model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    dl_model.add(Dense(32, activation='relu'))
    dl_model.add(Dense(16, activation='relu'))
    dl_model.add(Dense(1))  # Output layer for regression

    dl_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    dl_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    y_pred_dl = dl_model.predict(X_test_scaled)
    y_pred_dl = y_pred_dl.flatten()  # Flattening the predictions to 1D
    model_performance['Deep Learning'] = {
        'MSE': mean_squared_error(y_test, y_pred_dl),
        'R2': r2_score(y_test, y_pred_dl)
    }
    data['Predicted Close (DL)'] = dl_model.predict(scaler.transform(X)).flatten()

    # Sidebar for Model and Residual Plot Selection
    st.sidebar.title("Navigation")
    selected_model = st.sidebar.selectbox("Select Model to View Price Action", 
                                          ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Deep Learning"])

    st.sidebar.title("Residual Analysis")
    residual_model = st.sidebar.selectbox("Select Model for Residual Plot", 
                                          ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Deep Learning"])

    def plot_residuals(y_true, y_pred, model_name):
        # Residuals calculation
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals Plot for {model_name}')
        
        # Dynamic commentary
        commentary = f"""
        **How to Read the Residual Plot for {model_name}:**
        - The residuals are the differences between the actual and predicted values.
        - A good model typically has residuals scattered randomly around the horizontal axis (y=0 line).
        - If there's a pattern (e.g., a curve), it indicates the model may be biased or missing important variables.
        - Large residuals (far from 0) indicate instances where the model is performing poorly.

        **What to Expect from {model_name}:**
        - **Linear Regression**: Residuals should ideally be evenly distributed. A funnel shape might indicate heteroscedasticity.
        - **Decision Tree**: Expect to see a more scattered distribution due to overfitting on the training data.
        - **Random Forest**: Should have smaller and more evenly distributed residuals compared to a single decision tree.
        - **Gradient Boosting**: Typically has even smaller residuals with less variance.
        - **Deep Learning**: Depending on the network's complexity, residuals might show a slight pattern if the model overfits or underfits.
        """

        st.write(commentary)

        # Adding annotations
        plt.annotate('Ideal Random Scattering', xy=(min(y_pred), 0), xytext=(min(y_pred), max(residuals)),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     fontsize=12, color='green')
        
        plt.annotate('Potential Bias Detected', xy=(max(y_pred), min(residuals)), xytext=(max(y_pred) - 1, min(residuals) + 2),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     fontsize=12, color='red')
        
        st.pyplot(plt)

    # Display selected residual plot
    if residual_model == "Linear Regression":
        plot_residuals(y_test, y_pred_lr, 'Linear Regression')
    elif residual_model == "Decision Tree":
        plot_residuals(y_test, y_pred_dt, 'Decision Tree')
    elif residual_model == "Random Forest":
        plot_residuals(y_test, y_pred_rf, 'Random Forest')
    elif residual_model == "Gradient Boosting":
        plot_residuals(y_test, y_pred_gb, 'Gradient Boosting')
    elif residual_model == "Deep Learning":
        plot_residuals(y_test, y_pred_dl, 'Deep Learning')

    # Generate narrative summary
    st.write("## Model Performance Summary")
    best_model = min(model_performance, key=lambda k: model_performance[k]['MSE'])
    st.write(f"### Best Performing Model: {best_model}")
    st.write(f"The best performing model is **{best_model}** with the lowest Mean Squared Error (MSE) and highest R-Squared value.")
    st.write(f"The following table summarizes the performance of each model:")

    # Display model performance on the top right side
    with st.sidebar:
        st.write("### Model Performance Comparison")
        performance_df = pd.DataFrame(model_performance).T
        performance_df = performance_df.round(2).applymap('{:,.2f}'.format)
        st.dataframe(performance_df)

    # Calculate the 98th and 2nd percentiles for the Difference column
    for model in ['LR', 'DT', 'RF', 'GB', 'DL']:
        data[f'Difference ({model})'] = data[f'Predicted Close ({model})'] - data['Target']
        upper_threshold = np.percentile(data[f'Difference ({model})'].values, 98)
        lower_threshold = np.percentile(data[f'Difference ({model})'].values, 2)

        # Label accuracy based on these thresholds
        def label_accuracy(diff):
            if diff >= upper_threshold:
                return 'High Overestimation'
            elif diff <= lower_threshold:
                return 'High Underestimation'
            elif lower_threshold < diff < upper_threshold:
                return 'Accurate'
            else:
                return 'Moderate Error'

        data[f'Accuracy ({model})'] = data[f'Difference ({model})'].apply(label_accuracy)

    # Round and format the values in the table for display
    display_data = data.copy().round(2)
    
    # Sort data by date in descending order
    display_data = display_data.sort_index(ascending=False)

    # Display a dropdown to select which model to visualize
    st.write("### Visualize Predictions")
    selected_model_to_display = st.selectbox("Select the model to display:", ["LR", "DT", "RF", "GB", "DL"])

    st.write(f"#### Predictions for {selected_model_to_display} Model")
    metrics_table = display_data[['Open', 'High', 'Low', 'Close', 'Volume', f'Predicted Close ({selected_model_to_display})', f'Difference ({selected_model_to_display})', f'Accuracy ({selected_model_to_display})', 'Target']]
    
    # Highlight different accuracy levels
    def highlight_accuracy(val):
        if val == 'High Overestimation':
            color = 'red'
        elif val == 'High Underestimation':
            color = 'orange'
        elif val == 'Accurate':
            color = 'green'
        else:
            color = 'yellow'
        return f'background-color: {color}'
    
    styled_table = metrics_table.style.applymap(highlight_accuracy, subset=[f'Accuracy ({selected_model_to_display})'])
    st.dataframe(styled_table, use_container_width=True)

    # Add a candlestick chart for the last 7 days with a line for predicted prices
    st.write(f"#### Projected Price Action for {selected_model_to_display} Model")
    last_seven_days = display_data[-7:].index
    last_seven_days_data = display_data.loc[last_seven_days]
    
    # Prepare predicted data for the next 4 days in hourly intervals
    last_date = last_seven_days_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=4 * 24, freq='H')
    
    future_predictions = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': np.nan,
    })
    future_predictions['Predicted Close'] = display_data[f'Predicted Close ({selected_model_to_display})'].iloc[-4:].values[0]
    future_predictions.set_index('Date', inplace=True)

    # Plot candlesticks for the last 7 days and line for the next 4 days
    fig = go.Figure()

    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=last_seven_days_data.index,
        open=last_seven_days_data['Open'],
        high=last_seven_days_data['High'],
        low=last_seven_days_data['Low'],
        close=last_seven_days_data['Close'],
        name='Candlestick'
    ))

    # Add line trace for future predictions
    fig.add_trace(go.Scatter(
        x=future_predictions.index,
        y=future_predictions['Predicted Close'],
        mode='lines',
        name='Predicted Close',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title=f"Price Action for {selected_model_to_display} Model",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("No data available for the selected ticker.")
