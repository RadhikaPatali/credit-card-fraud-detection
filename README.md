# Credit Card Fraud Detection

This code implements a credit card fraud detection system using machine learning techniques. It uses a dataset containing credit card transaction information and builds a model to predict whether a transaction is fraudulent or not.

## Dataset
The dataset used in this code can be found at [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection). It consists of two files:
- `fraudTrain.csv`: Training dataset
- `fraudTest.csv`: Test dataset

Make sure to download the dataset files and place them in the same directory as the code file.

## Dependencies
This code requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `imblearn`
- `sklearn`
- `xgboost`
- `datetime`
- `hyperopt`

You can install these libraries using pip:

```
pip install <package_name>
```
Replace <package_name> with the name of the package that needs to be installed.

## Instructions
1. Make sure to have the dataset files (`fraudTrain.csv` and `fraudTest.csv`) in the same directory as the code file.

2. Open the code file and run it using a Python IDE or Jupyter Notebook.

3. The code performs the following steps:

    a. Data analysis and visualization: It provides information about the dataset, performs exploratory data analysis, and visualizes the data using heatmaps and bar plots.
    
    b. Pre-processing: It applies data pre-processing techniques such as downsampling, one-hot encoding, and label encoding to prepare the data for training the models.
    
    c. Data splitting: It splits the pre-processed data into training and test sets.
    
    d. Model training and evaluation: It trains several machine learning models, including Support Vector Machine (SVM), Decision Tree, Logistic Regression, and XGBoost. It evaluates the models using classification metrics such as accuracy, confusion matrix, and classification report.
    
    e. Hyperparameter tuning (optional): It demonstrates an example of hyperparameter tuning using XGBoost with the hyperopt library. This step is optional and can be skipped if not required.
    
4. The results, including classification reports and confusion matrices, will be displayed for each model. The best hyperparameters (if tuned) will also be shown.