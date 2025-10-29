# Hate-speech-classifier
Machine learning and deep learning models to detect online cyberbullying on social media.

# 🧠 Detecting Online Cyberbullying using Machine Learning & Deep Learning

## 📋 Overview
This project focuses on detecting **online cyberbullying** (specifically *racism* and *sexism*) using various machine learning and deep learning models.  
It leverages a labeled Twitter dataset and compares traditional algorithms like **Random Forest** with a **Convolutional Neural Network (CNN)** model.

---

## 📂 Dataset
- **Source:** Combined Twitter datasets labeled as `racism`, `sexism`, and `none`.
- **Final distribution:**
  - 🗣️ Racism: **3,940**
  - 👩 Sexism: **3,377**
  - ⚪ None: **23,005**

---

## 🧹 Data Preprocessing
Steps performed:
- Removal of URLs, mentions, hashtags, punctuation, digits, and emojis.
- Expansion of contractions (“can’t” → “cannot”).
- Stopword removal.
- Lemmatization using WordNetLemmatizer.
- Feature extraction using **TF-IDF Vectorizer (max_features=5000)**.
- Train-test split (80/20), stratified by label.

---

## 🧠 Models Implemented
| Model Type | Algorithm |
|-------------|------------|
| Classical ML | K-Nearest Neighbors (KNN) |
| Classical ML | Decision Tree |
| Classical ML | Support Vector Machine (SVM) |
| Classical ML | Random Forest |
| Deep Learning | Convolutional Neural Network (CNN) |

---

## 📊 Model Performance (Before SMOTE)

| Model | Accuracy | Precision | Recall | F1 Score |
|:--|:--:|:--:|:--:|:--:|
| KNN | 0.8425 | 0.8382 | 0.8425 | 0.8212 |
| Decision Tree | 0.9200 | 0.9159 | 0.9200 | 0.9150 |
| SVM | 0.8811 | 0.8787 | 0.8811 | 0.8727 |
| **Random Forest** | 🟢 **0.9294** | 🟢 **0.9270** | 🟢 **0.9294** | 🟢 **0.9235** |
| CNN | 0.9070 | 0.9033 | 0.9070 | 0.9014 |

---

## 🔁 Model Performance (After SMOTE)

| Model | Accuracy | Precision | Recall | F1 Score |
|:--|:--:|:--:|:--:|:--:|
| KNN | 0.7799 | 0.8306 | 0.7799 | 0.7927 |
| Decision Tree | 0.9237 | 0.9200 | 0.9237 | 0.9199 |
| SVM | 0.8087 | 0.8538 | 0.8087 | 0.8221 |
| **Random Forest** | 🟢 **0.9334** | 🟢 **0.9305** | 🟢 **0.9334** | 🟢 **0.9301** |

---

## 📈 Results Visualization
Visualizations make it easier to compare model performance and understand classification behavior.

### 1. Sampled Model Comparison
![Sampled_model](https://github.com/Scvrfvce/Hate-speech-classifier/blob/2946df115deeab49efdd465493b0c474b01c64a4/Model%20training%20perfomance%20sampled.png)  
*shows the perfomance of all models after SMOTE.*

### 2. Unsampled Model Comparison
![Unsampled_model](https://github.com/Scvrfvce/Hate-speech-classifier/blob/2946df115deeab49efdd465493b0c474b01c64a4/Model%20training%20perfomance%20Unsampled.png)  
*shows perfomance of all models before SMOTE .*

### 3. Confusion Matrix for Sampled Random Forest model
![Confusion Matrix](https://github.com/Scvrfvce/Hate-speech-classifier/blob/2946df115deeab49efdd465493b0c474b01c64a4/Sampled%20Random%20forest%20model.png))  
*Shows how well the Random Forest model predicts each class.*


## 🚀 Recommendations
- Use **Random Forest** for production deployment.
- Perform **hyperparameter tuning** (`n_estimators`, `max_depth`, etc.) for better accuracy.
- Try **embedding-based CNN or transformer models (BERT, DistilBERT)** for richer context understanding.
- Visualize **feature importance** for interpretability.

---

## 🧰 Technologies Used
- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- TensorFlow / Keras
- Matplotlib, Seaborn (for visualization)


## 👨‍🔬 Author
**[Scvrfvce]**  
*Machine Learning Researcher / Data Science Enthusiast*  


## 📝 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
