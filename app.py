import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

st.set_page_config(page_title="Dự báo giá cổ phiếu bằng LSTM", page_icon="📈", layout="wide")

st.markdown("""
# 📈 Dự báo giá cổ phiếu Việt Nam bằng LSTM
Chọn sàn, tìm mã cổ phiếu hoặc nhập mã Yahoo Finance để dự báo giá đóng cửa.
""")

st.sidebar.header("⚙️ Tùy chọn mô hình")
ticker_input = st.sidebar.text_input("Nhập mã cổ phiếu (VD: VNM.VN, ACB.VN, FPT.VN):", value="VNM.VN")
start_date = st.sidebar.date_input("Từ ngày", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Đến ngày", value=pd.to_datetime("2026-10-14"))

epochs = st.sidebar.slider("Số vòng lặp (Epochs)", min_value=5, max_value=50, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

run_button = st.sidebar.button("🚀 Chạy dự báo & Huấn luyện AI")

if run_button:
    with st.spinner(f"Đang tải dữ liệu và huấn luyện mô hình LSTM cho mã {ticker_input}..."):
        df = yf.download(ticker_input, start=str(start_date), end=str(end_date))
        
        if df.empty or len(df) < 100:
            st.error("Không tìm thấy dữ liệu hoặc dữ liệu quá ít. Vui lòng thử lại với mã khác (VD: VNM.VN, ACB.VN).")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Số phiên có dữ liệu", f"{len(df):,}")
            latest_close = float(df['Close'].iloc[-1])
            col2.metric("Giá đóng cửa gần nhất", f"{latest_close:,.2f}")
            col3.metric("Ngày giao dịch gần nhất", df.index[-1].strftime('%d/%m/%Y'))

            st.markdown(f"### Biểu đồ giá lịch sử — {ticker_input}")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            ax_hist.plot(df.index, df['Close'], color='blue', linewidth=1.2)
            st.pyplot(fig_hist)

            data = df[['Close']].values
            train_size = int(0.8 * len(data))
            train_data = data[:train_size]
            test_data = data[train_size:]

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data)
            test_scaled = scaler.transform(test_data)

            def create_sequences(data, window_size=60):
                X, y = [], []
                for i in range(window_size, len(data)):
                    X.append(data[i-window_size:i])
                    y.append(data[i])
                return np.array(X), np.array(y)

            window_size = 60
            X_train, y_train = create_sequences(train_scaled, window_size)
            
            test_combined = np.vstack((train_scaled[-window_size:], test_scaled))
            X_test, y_test = create_sequences(test_combined, window_size)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            val_size = int(0.1 * len(X_train))
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train_final = X_train[:-val_size]
            y_train_final = y_train[:-val_size]

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = model.fit(
                X_train_final, y_train_final,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            test_pred = model.predict(X_test)
            test_pred = scaler.inverse_transform(test_pred)
            y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

            rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
            mae = mean_absolute_error(y_test_real, test_pred)
            r2 = r2_score(y_test_real, test_pred)

            st.markdown("### 📊 Đánh giá mô hình trên tập Test")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE", f"{rmse:.2f}")
            m2.metric("MAE", f"{mae:.2f}")
            m3.metric("R² Score", f"{r2:.4f}")

            st.markdown("### Biểu đồ Loss Train vs Validation")
            fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
            ax_loss.plot(history.history['loss'], label='Train Loss')
            ax_loss.plot(history.history['val_loss'], label='Validation Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)

            st.markdown("### Biểu đồ so sánh Giá thật và Dự báo (Test)")
            fig_test, ax_test = plt.subplots(figsize=(10, 5))
            ax_test.plot(y_test_real, label="Giá thật", color='blue')
            ax_test.plot(test_pred, label="Dự báo test", color='orange')
            ax_test.legend()
            st.pyplot(fig_test)

            last_60_days = np.vstack((train_scaled, test_scaled))[-window_size:]
            last_60_days = last_60_days.reshape(1, window_size, 1)
            future_predictions = []

            for i in range(10):
                next_pred = model.predict(last_60_days)
                future_predictions.append(next_pred[0, 0])
                last_60_days = np.append(
                    last_60_days[:, 1:, :],
                    next_pred.reshape(1, 1, 1),
                    axis=1
                )

            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)

            st.markdown("### 🔮 Dự báo giá đóng cửa 10 ngày tiếp theo:")
            future_df = pd.DataFrame({
                "Ngày thứ": [f"Ngày +{i+1}" for i in range(10)],
                "Giá dự báo": future_predictions.flatten()
            })
            st.dataframe(future_df.style.format({"Giá dự báo": "{:,.2f}"}))
else:
    st.info("Chọn mã cổ phiếu ở menu bên trái và nhấn **Chạy dự báo & Huấn luyện AI** để bắt đầu.")
