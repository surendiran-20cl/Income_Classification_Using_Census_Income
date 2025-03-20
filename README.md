# Income Classification Using Census Income Dataset

## Project Overview
This project focuses on classifying individuals based on income levels using the Census Income dataset from the UCI Machine Learning Repository. The goal is to predict whether an individual's income exceeds $50,000 per year based on demographic and employment-related features.

## Dataset
The dataset used in this project is the **Census Income Dataset** (also known as the Adult dataset), which contains 48,842 records collected from the 1994 U.S. Census. It includes various attributes such as age, education, occupation, and hours worked per week.

- Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income)
- Features: 14 categorical and numerical attributes
- Target Variable: Binary classification (`<=50K` or `>50K`)

## Project Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling
2. **Exploratory Data Analysis (EDA)**
   - Visualizing distributions of numerical and categorical variables
   - Identifying correlations between features
3. **Model Training & Evaluation**
   - Models used: Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM
   - Performance metrics: Accuracy, Precision, Recall, F1-score
   - Hyperparameter tuning for best performance
4. **Deployment**
   - The final model (LightGBM) is saved as a pickle file (`lightgbm_income_classifier.pkl`).
   - A **Flask API** is built for real-time income classification.

## Model Deployment
The deployment is handled using **Flask**, providing a REST API for income classification.

### Running the API
1. Install dependencies:
   ```bash
   pip install flask pandas numpy lightgbm
   ```
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. API Endpoints:
   - **Home Route:** `http://127.0.0.1:5000/` (Returns a welcome message)
   - **Prediction Route:** `http://127.0.0.1:5000/predict` (Accepts JSON input and returns a prediction)

### Example API Request
```json
{
  "age": 39,
  "education-num": 13,
  "hours-per-week": 40,
  "occupation": "Exec-managerial",
  "marital-status": "Married-civ-spouse"
}
```

### Example Response
```json
{
  "prediction": 1  # 1 indicates income >50K, 0 indicates income <=50K
}
```

## Repository Structure
```
|-- data/
|   |-- census-income.csv  # Processed dataset
|
|-- models/
|   |-- lightgbm_income_classifier.pkl  # Trained model
|
|-- notebooks/
|   |-- EDA_and_Modeling.ipynb  # Exploratory Data Analysis & Model Training
|
|-- app.py  # Flask API for deployment
|-- README.md  # Project documentation
```

## Conclusion
This project demonstrates a complete pipeline from data preprocessing to model deployment. The LightGBM model performed the best, and the Flask API allows real-time predictions. Future improvements may include deploying the model using **FastAPI** or **Docker** for scalability.

## Acknowledgment
This project was developed as part of the **Intellipaat Advanced Data Science & AI Program**.

## Contact
For any queries or feedback, feel free to reach out via GitHub issues.
