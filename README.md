# 📦 Late Delivery Risk Prediction Using Machine Learning

*A Machine Learning Project for Supply Chain Optimization*

## 📘 Overview

Late delivery is a recurring challenge in the eCommerce industry and is one of the major causes of customer dissatisfaction, operational inefficiency, and significant financial loss. The purpose of this project is to **predict whether an order will be delivered late** using supervised machine learning models trained on the **DataCo Smart Supply Chain Dataset**.

This project uses real-world eCommerce supply chain data containing customer, product, order, and shipment details to build predictive models capable of identifying potential late deliveries before they occur. Such predictions can support proactive planning, improve logistics operations, and enhance overall supply chain performance.

---

## 🎯 Objectives

The main objectives of this project are:

* To analyze the DataCo supply chain dataset and extract meaningful patterns related to delivery performance.
* To build multiple machine learning models (individual and ensemble) for predicting late deliveries.
* To evaluate these models using key performance metrics such as accuracy, precision, recall, and F1-score.
* To compare the performance of individual models vs. ensemble models.
* To identify the most influential factors affecting delivery delays.
* To provide a model that can help businesses reduce operational risks and improve customer satisfaction.

---

## ❓ Research Question

**“Can machine learning algorithms accurately predict late deliveries in an eCommerce supply chain, and which model performs best for this classification problem?”**

---

## 📁 Files in This Repository

This repository contains the following essential files:

```
📂 Late-Delivery-Risk-Prediction
│── DataCoSupplyChainDataset.csv
│── late_delivery_prediction.ipynb
│── late_delivery_prediction_model.ipynb
│── README.md   (this file)
```

### **📄 File Descriptions**

* **DataCoSupplyChainDataset.csv**
  The dataset used for this project containing customer, order, product, and shipping details.

* **late_delivery_prediction.ipynb**
  The main Jupyter Notebook containing exploratory data analysis (EDA), preprocessing, feature engineering, and initial model trials.

* **late_delivery_prediction_model.ipynb**
  The notebook focused on building final machine learning models, evaluating their performance, and comparing metrics across algorithms.

---

## 📊 Dataset Summary

* **Records:** ~180,000
* **Type:** Real-world eCommerce supply chain data
* **Target Variable:** `Delivery Status` (On time / Late)
* **Important Features:**

  * Days for shipping (real & scheduled)
  * Shipping mode
  * Order region and state
  * Customer segment
  * Product category

The dataset contains a moderate class imbalance but not severe enough to distort model performance.

---

## 🔧 Methodology

### **✔ Data Preprocessing**

* Removed irrelevant columns (customer zip, order zip, product description).
* Handled missing values.
* Converted categorical variables using label encoding.
* Selected numerical and non-problematic features for model training.
* Train-test splitting.

### **✔ Exploratory Data Analysis (EDA)**

* Identified distribution of late vs. on-time deliveries.
* Explored relationships between shipping mode, region, category, and delays.
* Identified key driver features influencing late delivery risk.

### **✔ Machine Learning Models Used**

#### **Individual Models**

* Logistic Regression
* Naïve Bayes
* KNN
* Decision Tree

#### **Ensemble Models**

* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Cross-validation accuracy

---

## 🏆 Results Summary

Ensemble techniques provided superior performance compared to traditional models.
Below is a simplified version of the comparative results (illustrative summary):

| Model               | Accuracy | Notes                                                 |
| ------------------- | -------- | ----------------------------------------------------- |
| Logistic Regression | ~97%     | Strong baseline performance                           |
| Random Forest       | ~98%     | One of the best models                                |
| XGBoost             | ~98%     | Best performer in most metrics                        |
| Naïve Bayes         | ~48%     | Performed poorly due to complex feature distributions |

### **Key Findings**

* Ensemble models such as Random Forest and XGBoost delivered the highest performance.
* Naïve Bayes struggled significantly due to data complexity.
* Shipping mode, region, and category name were strong predictors of late delivery.

---

## 🚀 How to Run This Project

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### **2. Install Dependencies**

You can install required libraries manually:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn
```

### **3. Run the Notebooks**

Open Jupyter Notebook or VS Code and run:

* `late_delivery_prediction.ipynb`
* `late_delivery_prediction_model.ipynb`

---

## 🛠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Matplotlib, Seaborn
* Jupyter Notebook

---

## 📌 Key Insights

* ML can accurately predict late deliveries in supply chains.
* Ensemble models outperform individual algorithms.
* Identifying risk early can help businesses improve logistics planning.
* Feature importance insights guide operational decisions for reducing delays.

---

## 📚 Acknowledgements

Dataset Source: *DataCo SMART Supply Chain Dataset*

---
## **👨‍💻 Author**

**Kunal — Aspiring Data Scientist**
Master’s Student in Applied Data Science
Focused on ML, analytics, and data-driven decision-making.
---
