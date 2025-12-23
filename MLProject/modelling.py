import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# 1. Load Dataset
data_path = "churnmodelling_preprocessing.csv"
df = pd.read_csv(data_path)

print("Dataset preprocessing berhasil dimuat")
print("Shape dataset:", df.shape)

# 2. Pisahkan fitur dan target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-test split selesai")

# 4. MLflow autolog
mlflow.set_experiment("Customer Churn Modelling")
mlflow.sklearn.autolog()

# 5. Train model (TANPA start_run)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Training selesai")

# 6. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# 7. Simpan model untuk MLflow Serve
model_dir = "model"  
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

mlflow.sklearn.save_model(model, model_dir)
print(f"Model tersimpan di folder: {model_dir}")