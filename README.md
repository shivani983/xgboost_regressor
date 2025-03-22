📜 README.md
# 🚀 Tesla Stock Price Prediction using XGBoost

This project implements an **XGBoost-based Machine Learning model** to predict Tesla stock prices using historical data. The model leverages **time-series features**, lag values, and cyclic encodings to improve predictive accuracy.

## 📌 Features
- Uses **XGBoost Regressor** for stock price prediction
- Implements **feature engineering** (lag features, cyclic encoding)
- **Streamlit Web App** for interactive predictions
- Supports **actual vs. predicted price visualization**
- **Random 10-row display in the sidebar** for quick data exploration

---

## 🔧 Installation & Environment Setup

Follow these steps to set up the project locally:

### **1️⃣ Create a Virtual Environment**

# Create a new virtual environment
python -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate

# MacOS/Linux
source venv/bin/activate
2️⃣ Install Required Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App

# streamlit run app.py
📊 Data Preprocessing
The model extracts time-series features from Tesla stock price data:

Feature Name	Description
open, high, low, close	Stock price attributes
volume, adjusted_close	Trading volume and adjusted close price
year, month, day, weekday, quarter	Extracted from date
month_sin, month_cos, day_sin, day_cos	Cyclic encoding for seasonality
lag_1, lag_7, lag_31	Previous stock prices to capture trends
🎯 Model Explanation
We use XGBoost Regressor, a tree-based ensemble learning algorithm that works well with structured data.

Target Variable (y) → Next day's close price

Features (X) → Historical stock data, date-based cyclic encodings, and lag features

Loss Function → Mean Squared Error (MSE)

Evaluation Metrics:

RMSE (Root Mean Squared Error)

R² Score (Model Accuracy)

🔹 Model Training
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load preprocessed data
X = df.drop(columns=["close"])  # Features
y = df["close"].shift(-1)  # Target: Next day's closing price

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.05)
xgb_reg.fit(X_train, y_train)
📈 How to Use the Streamlit App
1️⃣ Launch the App using:
 streamlit run app.py
2️⃣ Select stock price parameters in the sidebar
3️⃣ Click "Predict" to see the forecasted stock price
4️⃣ View the Graphs 


🤝 Contributing
Fork this repository

Create a new branch (git checkout -b feature-branch)

Make changes and commit (git commit -m "Added new feature")

Push the branch (git push origin feature-branch)

Submit a Pull Request 🚀

📜 License
This project is open-source and available under the MIT License.

🚀 Future Improvements
🔹 Add LSTM deep learning model for time-series forecasting
🔹 Integrate real-time stock price data (Yahoo Finance API)
🔹 Improve hyperparameter tuning with GridSearchCV

📬 Need Help? Contact Me
📧 Email: shivani_2312res617@iitp.ac.in
📌 LinkedIn: https://www.linkedin.com/in/shivani-virang-8748602b6/
📢 GitHub: shivani983

🔹 Star ⭐ this repository if you found it useful!
🔹 Contributions are welcome! 🚀

---bash
