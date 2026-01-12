# House Price Analysis & Prediction

This project focuses on analyzing housing data and predicting house prices
using data analytics and machine learning techniques.

Key steps included:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Regression-based machine learning model
- Model evaluation and insights generation

This project was completed as part of the Code With Harry Data Science Course.


## Kaggle Competition
- Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Model Score: 87.16% (R-squared score)

## File Structure
- `house_price_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and prediction.
- `submission.csv`: CSV file containing the predicted house prices for the test dataset.
- `gbr.pkl`: Pickle file containing the trained GradientBoostingRegressor model.

## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Data Loading and Analysis
- The training and test datasets are loaded from CSV files.
- Exploratory data analysis is performed to understand the structure and characteristics of the data.
- Data visualization techniques such as histograms, box plots, and heatmaps are used to analyze the distribution of features and identify missing values.

## Data Preprocessing
- Missing values are handled using appropriate techniques such as imputation or dropping columns.
- Categorical variables are encoded using one-hot encoding.
- Numerical features are standardized to ensure uniformity and improve model performance.

## Model Selection and Training
- Several regression models are considered, including Linear Regression, SVR, SGDRegressor, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor, and MLPRegressor.
- Cross-validation is used to evaluate each model's performance based on the R-squared score.
- The GradientBoostingRegressor model is selected based on its superior performance.

## Model Evaluation and Prediction
- The selected model is trained on the training dataset.
- The trained model is used to make predictions on the test dataset.
- The predictions are saved to a CSV file (`submission.csv`) for submission.


## Additional Notes
- The `submission.csv` file contains the predicted house prices for the test dataset.
- The trained model (`gbr.pkl`) is stored as a pickle file for future use or deployment.

For any further inquiries or improvements, feel free to reach out.


Author: Amaan Hasan Farooqi
