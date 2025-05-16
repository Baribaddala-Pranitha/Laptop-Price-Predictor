# 💻 Laptop Price Prediction using Machine Learning

This project predicts laptop prices using a variety of features such as brand, type, screen size, resolution, hardware specs, and operating system. Multiple regression models are trained and compared to identify the best-performing model.

---

## 📁 Dataset

The dataset (`laptop_data.csv`) contains 1303 entries with the following key features:
- `Company`
- `TypeName`
- `Inches`
- `ScreenResolution`
- `Cpu`
- `Ram`
- `Memory`
- `Gpu`
- `OpSys`
- `Weight`
- `Price`

---

## 🧹 Data Preprocessing

Key preprocessing steps include:

- Extracting binary columns from `ScreenResolution`:
  - `TouchScreen`
  - `IPS`
- Parsing screen resolution to get `X_res` and `Y_res`.
- Calculating **PPI (Pixels Per Inch)**:
  \[
  \text{PPI} = \frac{\sqrt{X\_res^2 + Y\_res^2}}{\text{Inches}}
  \]
- Parsing `Memory` into separate `HDD` and `SSD` columns.
- Extracting `CPU Brand` and `GPU Brand`.
- Converting categorical features using OneHotEncoding.

---

## ✂️ Train/Test Split

- Used `train_test_split()` to split the data:
  - **85%** for training
  - **15%** for testing

---

## 🤖 Models Trained

| Model                 | R² Score | MAE     |
|----------------------|----------|---------|
| Linear Regression     | 0.807    | 0.210   |
| K-Nearest Neighbors   | 0.801    | 0.194   |
| Decision Tree         | 0.848    | 0.180   |
| Support Vector Regressor | 0.808 | 0.202   |
| **Random Forest**         | **0.887**| **0.159**|
| Extra Trees           | 0.875    | 0.160   |
| AdaBoost              | 0.779    | 0.240   |

✅ **Best Model:** RandomForestRegressor

---

## 🧪 Evaluation Metrics

- **R² Score**: Measures how well predictions approximate actual values.
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.

---

## 💾 Model Export

The trained model and processed dataset are saved using `pickle`:
- `df.pkl`: Processed DataFrame
- `pipe.pkl`: Final model pipeline (includes preprocessing + trained model)

---

## 🔍 Predicting Price

You can load and use the model like this:

```python
import pickle
import numpy as np

# Load the trained pipeline
model = pickle.load(open('pipe.pkl', 'rb'))

# Sample input
sample = np.array(['HP', 'Notebook', 8, 2.2, 0, 1, 141.21, 'Intel Core i5', 1000, 128, 'Intel', 'Windows']).reshape(1, -1)

# Predict log price and exponentiate
predicted_log_price = model.predict(sample)[0]
predicted_price = np.exp(predicted_log_price)

print(f"Predicted Laptop Price: ₹{predicted_price:.2f}")
