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

# 4. MLflow setup
mlflow.set_experiment("Customer Churn Modelling")

# 5. Train model
try:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log manual (run sudah dimulai oleh mlflow run)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print("Training selesai")
    print("Accuracy:", accuracy)
except Exception as e:
    print(f"Error during training: {e}")
    raise
