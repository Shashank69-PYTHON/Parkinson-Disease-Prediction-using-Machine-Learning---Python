# Parkinson-Disease-Prediction-using-Machine-Learning---Python
Parkinson's disease is a progressive neurological disorder that affects movement. Stiffening, tremors and slowing down of movements may be signs of Parkinson's disease. While there is no certain diagnostic test, but we can use machine learning in predicting whether a person has Parkinson's disease based on specific biomarkers. In this article, we will use machine learning models to predict Parkinson's disease.
1. Importing Libraries and Dataset
We will be usingPandas, Numpy, Matplotlib, Seaborn, Sckit-learn, XGBoost and Imblearn.
2. Importing Dataset
The dataset we are going to use here includes 755 columns and three observations for each patient. The value's in these columns are part of some other diagnostics which are generally used to capture the difference between a healthy and affected person.
3. Data Exploration and Cleaning
To gain a better understanding of the dataset, we utilize several built-in functions from the Pandas library. These tools help us examine the structure, data types, and overall characteristics of the data.

df.info(): This function gives a quick summary of the dataset, showing the number of rows and columns, data types of each column, and how many non-null (non-missing) values are present. It's useful for spotting missing data and understanding the data types you'll be working with.

df.describe().T: This provides a statistical overview of the numerical columns, including metrics like mean, standard deviation, minimum, and maximum values, along with the quartiles. Transposing the summary with .T makes it easier to read and compare features.

df.isnull().sum().sum(): This expression calculates the total number of missing values in the entire dataset. If the result is 0, the data has no missing entries. If missing values are found, data cleaning steps will be required to handle them appropriately.
4. Data Wrangling
Data wrangling involves reorganizing and transforming the dataset to make it suitable for analysis. In our case, since the dataset contains multiple entries for each patient (three observations per patient), we need to aggregate these records into a single, representative entry for each individual. The steps to achieve this are as follows:

df.groupby('id').mean().reset_index(): This groups the data by the "id" column (which represents patient ID) and calculates the average of the numerical features for each patient. This ensures that multiple records for the same patient are combined into a single entry.

df.drop('id', axis=1, inplace=True): After the aggregation, the "id" column is no longer needed, so we remove it from the dataset. This helps in simplifying the data and keeping only the relevant columns.

5. Feature Selection
Feature selection helps enhance model accuracy and decreases the computational load by identifying and retaining only the most relevant features. This process helps in removing unnecessary or less impactful variables, which makes the dataset more optimized for machine learning. The steps involved are as follows:

X = df.drop('class', axis=1): This step removes the target variable (class) and extracts only the feature set for further processing.

MinMaxScaler().fit_transform(X): This scales the features to a range between [0,1] using Min-Max Scaling, ensuring that each feature has an equal impact on the model’s performance.

SelectKBest(chi2, k=30): The Chi-Square test is applied here to select the top 30 features, based on their correlation with the target variable (class).

selector.fit(X_norm, df['class']): The feature selection model is then trained on the normalized feature data, using the target variable for reference.

filtered_columns = selector.get_support(): This step identifies the most important features based on the Chi-Square test results.

filtered_data = X.loc[:, filtered_columns]: Only the selected top 30 features are extracted from the original dataset.

filtered_data['class'] = df['class']: Finally, the target variable (class) is reattached to the dataset, forming a reduced set with only the most significant features.
