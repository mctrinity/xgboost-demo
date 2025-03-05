import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Generate synthetic structured data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Simulated unstructured text data (e.g., customer feedback)
feedback = [
    "Great service, highly recommend!",
    "Terrible experience, won't use again.",
    "Average quality, decent pricing.",
    "Very satisfied, fast delivery!",
    "Support was unhelpful and rude.",
]

# Initialize LangChain LLM for chat models
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

# Extract sentiment scores using LLM
sentiment_scores = []
for review in feedback:
    response = llm.invoke(
        f"Analyze sentiment of this text: '{review}'. Return a number from -1 (negative) to 1 (positive)."
    )
    sentiment_scores.append(float(response.content))  # Extract content from AIMessage

# Ensure sentiment scores match data shape
sentiment_scores = np.array(sentiment_scores).reshape(-1, 1)

# Extend structured data with sentiment scores
num_samples = X_train.shape[0]  # Ensure we match the number of training samples
sentiment_train = np.random.choice(sentiment_scores.flatten(), num_samples).reshape(
    -1, 1
)
num_samples_val = X_val.shape[0]
sentiment_val = np.random.choice(sentiment_scores.flatten(), num_samples_val).reshape(
    -1, 1
)

X_train_extended = np.hstack((X_train, sentiment_train))
X_val_extended = np.hstack((X_val, sentiment_val))

# Convert structured + text-enhanced data into DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train_extended, label=y_train)
dval = xgb.DMatrix(X_val_extended, label=y_val)

# Define XGBoost parameters to prevent overfitting
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "max_depth": 4,
    "lambda": 10.0,  # L2 Regularization
    "alpha": 2.0,  # L1 Regularization
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Train XGBoost with early stopping and enable evals_result
evals = [(dtrain, "train"), (dval, "val")]
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=50,
)

# Make predictions
y_pred = model.predict(dval)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Evaluate accuracy
accuracy = accuracy_score(y_val, y_pred_binary)
print(f"Final Model Accuracy: {accuracy:.2f}")

# Extract training history
train_loss = evals_result["train"]["logloss"]
val_loss = evals_result["val"]["logloss"]

# Create a directory for saving plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Plot Training vs Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(val_loss, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "training_vs_validation_loss.png"))
plt.show()

# Plot Feature Importance
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.savefig(os.path.join(plot_dir, "feature_importance.png"))
plt.show()

# Plot Prediction Distribution
plt.figure(figsize=(8, 5))
plt.hist(y_pred, bins=20, alpha=0.7, label="Predicted Probabilities")
plt.axvline(0.5, color="red", linestyle="dashed", label="Decision Boundary")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Prediction Distribution")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "prediction_distribution.png"))
plt.show()
