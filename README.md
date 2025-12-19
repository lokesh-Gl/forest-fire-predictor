# Forest Fire Prediction App

This project is a **machine learning-powered web application** built with **Streamlit** that predicts the likelihood of a forest fire based on weather indices and time-based features via CSV upload.

---

##  Features

-  **Predict forest fire** probability using Random Forest model  
-  **Feature importance** visualization   
-  Year-wise **fire trend analysis**  
-  **Map** with marker to show static prediction location  
-  Upload **CSV file** for batch predictions  

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit app |
| `random_forest_fire_model.pkl` | Trained Random Forest model |
| `forest_fire_updated.csv` | Dataset used for training & trend visualizations |
| `Forest_Fire_Prediction.ipynb` | Notebook used to train and save model |
| `README.md` | You are here ✔️ |

---

##  Model Information

- **Model**: Random Forest Classifier  
- **Trained on features**:
  - `ISI`, `FFMC`, `FWI`, `DMC`, `day`, `month`, `year`
- **Target**: `Classes` (`0` = No Fire, `1` = Fire)

---

## ⚙️ How to Run

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/yourusername/forest-fire-predictor.git
cd forest-fire-predictor
pip install -r requirements.txt
```
