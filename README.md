# Task-6
# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

This project demonstrates how to classify iris flower species using the **K-Nearest Neighbors (KNN)** algorithm, applied to the classic **Iris dataset**.

---

## 📌 Objective

- Understand and implement the KNN classification algorithm.
- Evaluate model performance using accuracy and confusion matrix.
- Visualize model performance for different values of **K**.

---

## 🛠️ Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

---

## 📁 Dataset

- **Name**: Iris.csv
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Target**: Species (Iris-setosa, Iris-versicolor, Iris-virginica)

---

## 🔍 Steps Performed

1. Loaded and explored the dataset.
2. Dropped unnecessary columns.
3. Normalized feature values using `StandardScaler`.
4. Encoded target labels using `LabelEncoder`.
5. Split data into training and test sets.
6. Trained KNN classifier for K values from 1 to 10.
7. Selected best K value based on test accuracy.
8. Evaluated final model using:
   - Accuracy score
   - Confusion matrix
   - Classification report
9. Visualized:
   - Accuracy vs K plot
   - Confusion matrix heatmap

---

## 📊 Results

- **Best K**: `2`
- **Test Accuracy**: `100%`
- **Model Performance**:
  Perfect classification on test set.

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python knn_iris_classifier.py
