# XGBoost Demo 🚀  

A machine learning project demonstrating **XGBoost** for classification tasks, with support for structured and unstructured data (via LangChain). Now includes visualization features to analyze model performance.  

## 📌 Features  
- **XGBoost Model** for classification  
- **Overfitting Prevention** using early stopping & regularization  
- **Feature Engineering** from structured & unstructured data  
- **LangChain Integration** for text feature extraction  
- **Data Visualization** with saved plots  

---

## 📂 Project Structure  
```
xgboost-demo/
│── xgboost_demo.py        # Main script for training XGBoost on structured data
│── xgboost_langchain.py   # XGBoost with LangChain-enhanced text processing
│── xgboost_plots.py       # XGBoost with data visualization and saved plots
│── requirements.txt       # Dependencies
│── .gitignore             # Files to ignore in Git
│── README.md              # Project documentation
│── data/                  # (Optional) Folder for datasets
│── models/                # (Optional) Folder to save trained models
│── plots/                 # Folder where plots are saved
```

---

## 📦 Installation  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-repo/xgboost-demo.git
cd xgboost-demo
pip install -r requirements.txt
```

---

## 🚀 Usage  

### **Run the Standard XGBoost Model (Structured Data Only):**  
```bash
python xgboost_demo.py
```
#### **Example Output for XGBoost Model Only:**  
```
[0]     train-logloss:0.66716   val-logloss:0.67121
[50]    train-logloss:0.28482   val-logloss:0.35612
[100]   train-logloss:0.21781   val-logloss:0.32199
[150]   train-logloss:0.18901   val-logloss:0.31459
[200]   train-logloss:0.17055   val-logloss:0.31356
[214]   train-logloss:0.16546   val-logloss:0.31316
Final Model Accuracy: 0.89
```

### **Run XGBoost with LangChain for Text Processing:**  
```bash
python xgboost_langchain.py
```
#### **Example Output for XGBoost + LangChain:**  
```
[0]     train-logloss:0.66716   val-logloss:0.67121
[50]    train-logloss:0.28422   val-logloss:0.35469
[100]   train-logloss:0.21863   val-logloss:0.32441
[150]   train-logloss:0.18764   val-logloss:0.31516
[190]   train-logloss:0.17194   val-logloss:0.31493
Final Model Accuracy: 0.90
```

### **Run XGBoost with Visualization & Saved Plots:**  
```bash
python xgboost_plots.py
```
#### **Generated Plots (Saved in `plots/` Directory):**  
- 📊 `training_vs_validation_loss.png` → Training vs Validation Loss Curve  
- 📊 `feature_importance.png` → Feature Importance Plot  
- 📊 `prediction_distribution.png` → Prediction Probability Distribution  

---

## ⚙️ Configuration  
Modify `xgboost_plots.py` to adjust:  
- `learning_rate`
- `max_depth`
- `n_estimators`
- Feature engineering techniques  
- Plot customizations (colors, labels, save paths)  

---

## 🔥 LangChain Integration  
This project includes **LangChain** to process text data and extract numerical features for XGBoost. The integration uses:  
✅ **Sentiment Analysis** – Convert feedback into sentiment scores  
✅ **Topic Extraction** – Categorize text into relevant topics  
✅ **LLM-based Feature Engineering** – Extract key insights from text  

Example usage in `xgboost_langchain.py`:  
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
response = llm.invoke("Analyze sentiment of this text: 'Great service!' Return a number from -1 to 1.")
sentiment_score = float(response.content)
```
These scores are then used as input features for XGBoost.  

---

## 📊 Feature Importance  
To visualize the most important features:  
```python
import xgboost as xgb
import matplotlib.pyplot as plt

xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()
```

---

## 🛠 Future Enhancements  
🔹 Optimize hyperparameters using **GridSearchCV**  
🔹 Add model **explainability (SHAP values)**  
🔹 Expand text processing with **BERT or OpenAI embeddings**  
🔹 More customizable plot styles and additional metrics  

---

## 🏆 Contributing  
Feel free to fork, modify, and submit a pull request! 🚀  

---

## 📜 License  
MIT License © 2025 Maki Dizon

