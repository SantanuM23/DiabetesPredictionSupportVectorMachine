# Diabetes Prediction Model

This project implements a Diabetes Prediction Model using the **Support Vector Machine (SVM)** algorithm. The model predicts whether an individual is diabetic or non-diabetic based on medical attributes provided in a dataset.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training](#model-training)
- [Testing the Model](#testing-the-model)
- [Results](#results)

## Dataset
The model uses the **PIMA Indian Diabetes Dataset** available as `diabetes.csv`. This dataset includes the following features:
- **Pregnancies**
- **Glucose**
- **BloodPressure**
- **SkinThickness**
- **Insulin**
- **BMI**
- **DiabetesPedigreeFunction**
- **Age**

The target column is:
- **Outcome**: 0 for Non-Diabetic, 1 for Diabetic.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.x
- NumPy
- pandas
- scikit-learn

Install dependencies using:
```bash
pip install numpy pandas scikit-learn
```

## Usage
1. Clone the repository or download the script.
2. Place the `diabetes.csv` file in the same directory as the script.
3. Run the script:
   ```bash
   python diabetes_prediction.py
   ```

## Model Training
The following steps are performed to train the model:
1. **Data Preprocessing**:
   - Features are scaled using `StandardScaler` to standardize the input data.
2. **Data Splitting**:
   - The dataset is split into training (80%) and testing (20%) sets.
3. **Model Training**:
   - The SVM model with a linear kernel is trained using the training data.

## Testing the Model
After training, the model's accuracy is evaluated on both training and testing datasets.

To test a single input:
1. Define an input array (example):
   ```python
   input = (8, 99, 84, 0, 0, 35.4, 0.388, 50)
   ```
2. The input is reshaped and standardized.
3. The trained model predicts whether the input corresponds to a diabetic or non-diabetic individual.

## Results
- **Training Accuracy**: `77.20%`
- **Testing Accuracy**: `76.62%`

### Example Input and Output
#### Input:
```python
input = (8, 99, 84, 0, 0, 35.4, 0.388, 50)
```
#### Output:
```
Non Diabetic
```

## Notes
- Ensure the dataset is clean and has no missing values.
- Experiment with different SVM kernels for better performance.
