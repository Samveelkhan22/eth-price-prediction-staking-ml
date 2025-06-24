# 🔮 Ethereum Price Prediction Using On-Chain Validator & Transaction Data

This project explores the use of machine learning models—including Random Forests and LSTMs—to predict future Ethereum (ETH) prices using full node data and validator behavior proxies. Our goal is to model price dynamics 1–2 weeks into the future, leveraging Ethereum’s transition from Proof of Work (PoW) to Proof of Stake (PoS) and its impact on staking behavior and transaction fees.

---

## 📈 Objective

- **Primary Goal**: Predict ETH price 7 days into the future using validator-related on-chain data.
- **Secondary Goals**:
  - Analyze how validator staking trends correlate with ETH volatility.
  - Investigate how gas fees and transaction activity affect short- and long-term price movements.

---

## 🧾 Data Sources

1. **Block Info Data** (`17000000to17249999_Block_Info.csv`)
2. **Transaction Data** (`tx_a_170.csv`, `tx_a_172.csv`, `tx_a_175.csv`, `tx_a_177.csv`)
3. **ETH Price Data** (`ETH-USD.csv`)

Each dataset was preprocessed, aligned by date, and merged into a final time-series DataFrame.

---

## 🛠️ Features

- **Block Metrics**: gasLimit, gasUsed, difficulty, size, timestamp
- **Transaction Metrics**: tx_count, gasPrice, baseFee, unique_senders/recipients
- **Engineered Features**: 
  - `fee_to_reward_ratio = avg_gasPrice / avg_maxPriorityFeePerGas`
  - `price_volatility = High - Low`
- **Target Variable**: 
  - `future_price_7d = Close (t + 7 days)`

---

## 📊 Visualizations

- Spearman correlation heatmap of key features
- Time series plots of ETH price, volume, and tx_count
- PCA decomposition of on-chain metrics

---

## 🤖 Models

### Random Forest Regressor
- Ensemble of 100 trees
- Handles non-linearity and multicollinearity
- MAE: ~75.22 USD | R²: ~0.891

### LSTM (PyTorch Lightning)
- 64 hidden units + dropout + dense layer
- Captures time-dependent behavior
- MAE: ~68.41 USD | R²: ~0.912

---

## 📈 Performance

| Model          | MAE     | RMSE    | R²     |
|----------------|---------|---------|--------|
| Random Forest  | 75.22   | 79.99   | 0.891  |
| LSTM           | 68.41   | 72.83   | 0.912  |

---

## 📚 Dependencies
- pandas, numpy, matplotlib, seaborn
- scikit-learn, PyTorch, PyTorch Lightning
- tqdm, PCA, scipy

---

## 📌 Limitations
- Limited post-Merge data window
- Validator count is inferred from sender behavior (not directly sourced)
- Assumes short-term stationarity in ETH transaction dynamics

