import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import datetime as dt
import plotly.express as px

data_frame = pd.read_csv("tsla_raw_data.csv")
data_frame.dropna(inplace=True)
feature_names = data_frame.columns.tolist()



def load_models(file = "xgb_stock_models.pkl"):
    xgb_reg,grid_xgb,le = joblib.load(file)
    return xgb_reg,grid_xgb,le

def preprocessing(data):
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract date-related features
    data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day.astype(int)  # Convert to integer
    data['quarter'] = data['Date'].dt.quarter
    data['day_of_year'] = data['Date'].dt.dayofyear
    data['weekday'] = data['Date'].dt.weekday


    # cyclic encoding for month and day ( important for capturing seasonality)
    data['month_sin'] = np.sin(2*np.pi*data['month']/12)
    data['month_cos'] = np.cos(2*np.pi*data['month']/12)
    data['day_sin'] = np.sin(2*np.pi*data['day']/31)
    data['day_cos'] = np.cos(2*np.pi*data['day']/31)

    # creating lag for stock price 
    data['lag_1'] = data['close'].shift(1)
    data['lag_7'] = data['close'].shift(7)
    data['lag_31'] = data['close'].shift(31)

    data['date'] = data['Date'].astype(int) // 10**9
    features = ['open', 'high', 'low', 'volume', 'adjusted_close', 'change_percent', 'avg_vol_20d',
                 'year', 'month', 'weekday', 'quarter', 'day_of_year', 'day', 'month_sin', 'month_cos', 'day_sin', 
                'day_cos', 'lag_1', 'lag_7', 'lag_31']


    # Drop the original Date column
    data = data.drop(columns=['Date'])
    data = data[features]



    return data

def predict(data,model):
    if model == xgb_reg:
        prediction = xgb_reg.predict(data,enable_categorical=True)

    else:
        prediction = grid_xgb.predict(data,enable_categorical=True) 

    return prediction

def main():
    st.title("📈 Tesla Stock Price Prediction")
    st.write("This is a simple web application that uses **XGBoost** to predict Tesla's stock price.")

    xgb_reg,grid_xgb,le = load_models()

    st.subheader("Enter Stock Features for Prediction")
    date = st.date_input("📅 Select Date:")
    open_price = st.number_input("📌 Open Price:", min_value=0)
    high_price = st.number_input("📌 High Price:", min_value=0)
    low_price = st.number_input("📌 Low Price:", min_value=0)
    close_price = st.number_input("📌 Close Price:", min_value=0)
    volume = st.number_input("📌 Volume:", min_value=0, format="%d")
    adjusted_close = st.number_input("📌 Adjusted Close Price:", min_value=0)
    change_percent = st.number_input("📌 Change Percentage:", min_value=-100, max_value=100)
    avg_vol_20d = st.number_input("📌 20-Day Average Volume:", min_value=0)
    model = st.selectbox("choose you want to xgboost model or xgboost with cross-validation",options = ["xgboost","xgboost with CV"])

    # Convert user input into a DataFrame
    user_data = pd.DataFrame({
        "Date": [date],
        "open": [open_price],
        "high": [high_price],
        "low": [low_price],
        "close": [close_price],
        "volume": [volume],
        "adjusted_close": [adjusted_close],
        "change_percent": [change_percent],
        "avg_vol_20d": [avg_vol_20d]
    })

    user_data = preprocessing(user_data)

    st.subheader("📊Predicted Stock Prices")
    
    @st.cache_data
    def show_data(df):
        return df.sample(n=10,random_state=1)

    st.sidebar.write("📊 Checking Data for Plot:")
    st.sidebar.dataframe(show_data(data_frame))


    X =st.sidebar.selectbox("choose X axis",feature_names)
    Y =st.sidebar.selectbox("choose Y axis",feature_names)
    if st.sidebar.button("plot"):
        if X in data_frame.columns and Y in data_frame.columns:
            fig, ax = plt.subplots()
            ax.plot(data_frame[X], data_frame[Y], marker="o", linestyle="-", color='#008000')  # ✅ Pass X and Y as positional arguments

            # Labels and Title
            ax.set_xlabel(X)
            ax.set_ylabel(Y)
            ax.set_title(f"{Y} vs. {X}")

            # ✅ Display Plot in Streamlit
            st.sidebar.pyplot(fig, use_container_width=True)
        else:
            st.sidebar.error("⚠️ Selected columns not found in DataFrame!")
    else:
        pass

    if st.button("predict"):
        if model == "xgboost with CV":
            prediction = grid_xgb.predict(user_data)
        else:
            prediction = xgb_reg.predict(user_data)

        st.success(f"Predicted price : {prediction}")   


if __name__ == "__main__":
    main()

            
               





