import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Define parameters to prevent overfitting
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,  # Lower LR for better generalization
    "max_depth": 4,  # Reduce depth to prevent complex trees
    "lambda": 10.0,  # Strong L2 regularization
    "alpha": 2.0,  # Strong L1 regularization
    "min_child_weight": 5,  # Prevents small splits
    "subsample": 0.8,  # Randomly select 80% of data per tree
    "colsample_bytree": 0.8,  # Use 80% of features per tree
}

# Train XGBoost with early stopping
evals = [(dtrain, "train"), (dval, "val")]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=20,
    verbose_eval=50,
)

# Make predictions
y_pred = model.predict(dval)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

# Evaluate accuracy
accuracy = accuracy_score(y_val, y_pred_binary)
print(f"Final Model Accuracy: {accuracy:.2f}")
