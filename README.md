# Autism Prediction Using Machine Learning

## Project Overview
This project aims to build a machine learning model to predict whether an individual has Autism Spectrum Disorder (ASD) based on various factors such as AQ-10 scores and demographic data. The goal is to create an early screening tool for ASD, providing a valuable tool for healthcare professionals.

## Dataset

The dataset consists of the following columns:

- **ID**: Unique identifier for each patient.
- **A1_Score to A10_Score**: Scores from the Autism Spectrum Quotient (AQ-10) screening tool.
- **age**: Age of the patient in years.
- **gender**: Gender of the patient.
- **ethnicity**: Ethnicity of the patient.
- **jaundice**: Whether the patient had jaundice at the time of birth.
- **autism**: Whether a family member has been diagnosed with autism.
- **contry_of_res**: Country of residence of the patient.
- **used_app_before**: Whether the patient has used a screening app before.
- **result**: AQ-10 screening result.
- **age_desc**: Describes the patient's age range.
- **relation**: Relation of the patient who completed the test.
- **Class/ASD**: Target variable indicating whether the patient has ASD (1) or not (0).

## Objective
The objective of this project is to build and evaluate multiple machine learning models (Decision Tree, Random Forest, XGBoost) to predict the presence of ASD based on the features provided. The best-performing model will be selected for final deployment.

## Technologies Used

- Python
- Libraries: 
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`
  - `imblearn`
  - `pickle`

## Approach

1. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical features using `LabelEncoder`.
   - Normalize and scale numerical features if necessary.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize feature distributions.
   - Investigate correlations between features using a heatmap.

3. **Model Training**:
   - Train and evaluate three classifiers: **Decision Tree**, **Random Forest**, and **XGBoost**.
   - Perform hyperparameter tuning using `RandomizedSearchCV`.

4. **Model Evaluation**:
   - Evaluate models based on **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**.
   - Cross-validation to ensure generalization.

5. **Best Model**:
   - The **Random Forest Classifier** with tuned hyperparameters performed the best, achieving **93%** cross-validation accuracy.

6. **Model Saving**:
   - The final model is saved using **Pickle** for future use and deployment.

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/autism-prediction.git
cd autism-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. **Load the Dataset**:  
   Load your dataset into a Pandas DataFrame.

2. **Preprocess the Data**:  
   Apply necessary preprocessing steps (e.g., missing value handling, encoding).

3. **Train the Model**:  
   Choose a model (Decision Tree, Random Forest, or XGBoost) and train it on the preprocessed dataset.

4. **Evaluate the Model**:  
   Use metrics like accuracy, precision, recall, and confusion matrix to evaluate the model performance.

5. **Save the Model**:  
   After selecting the best-performing model, use **Pickle** to save it for future use.

## Final Outcome

- The best model achieved **93% accuracy** with Random Forest Classifier after tuning.
- The model was evaluated using a confusion matrix and classification report, achieving a balanced performance for both ASD (1) and non-ASD (0) classes.

## Contributing

Feel free to fork the repository, submit issues, or contribute improvements to the project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the contributors of the dataset.
- Inspired by the need for early ASD detection to improve intervention and support.
