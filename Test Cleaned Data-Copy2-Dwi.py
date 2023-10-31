#!/usr/bin/env python
# coding: utf-8

# # Import All Useful Libs

# In[1]:
%pip install sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import seaborn as sns
from scipy import stats
from  scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


# # Read The Data

# In[2]:


df_train = pd.read_csv('input/insurance_data.csv')
df_train


# # Data Cleaning

# In[3]:


total = df_train.isnull().sum()
percent = (total / len(df_train)) * 100
missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_values)
df_train = df_train.drop('index',axis=1)


# In[4]:


mean_age = np.mean(df_train['age'])
# Fill null values in the 'age' column with the mean age
df_train['age'].fillna(mean_age, inplace=True)

# Check if there are any missing values in the 'age' column after filling
missing_values = df_train['age'].isnull().sum()
print("Missing values in 'age' column after filling:", missing_values)


# In[5]:


df_train.dropna(subset=['region'],inplace=True)

# Check if there are any missing values in the 'region' column after droping
missing_values = df_train['region'].isnull().sum()
print("Missing values in 'region' column after filling:", missing_values)


# In[6]:


df_train.describe()


# In[7]:


# Create dummy variables for the 'region' variable
region_dummies = pd.get_dummies(df_train['region'], prefix='region', drop_first=True)

# Encode the 'smoker' variable (0 for 'No' and 1 for 'Yes')
df_train['smoker'] = df_train['smoker'].map({'No': 0, 'Yes': 1})

# Show the first few rows of the encoded variables
region_dummies.head(), df_train[['smoker']].head()


# # Column Types

# Numerical : Patient ID, Age, BMI, Bloodpressure, Children, Claim
# Categorical : Gender, Diabetic, Smoker, Region

# In[8]:


df_train.sample(5)


# # Univariate Analysis

# ##### Conclusion: 
#     - Age has 5 missing values
#     - Data has two peaks and skewness of 0.11 so data is distributed biomodaly which means two age groups have claimed insurance most so one new column can be added to identify these age groups
#     - There are no outliers
#     - Max age is recorded 60 and mean age is 38 which shows very old people have not been insured

# In[9]:


df_train['age'].describe()


# In[10]:


sns.displot(df_train['age'],kde=True)


# In[11]:


df_train['age'].plot(kind='kde')


# In[12]:


print('skewness in age is ' ,df_train['age'].skew())


# In[13]:


df_train['age'].plot(kind='box')


# ###### BMI

# ##### Conclusion :
#     - Distribution is perfectly normal
#     - Skewness is very minimal
#     - Boxplot indicated some serious outlier, BMI above 50 shows very obese and its fatal

# In[14]:


sns.distplot(df_train['bmi'])


# In[15]:


df_train['bmi'].describe()


# In[16]:


df_train['bmi'].plot(kind='kde')


# In[17]:


df_train['bmi'].skew()


# In[18]:


df_train['bmi'].plot(kind='box')


# In[19]:


df_train[df_train['bmi']>35].count()


# ###### Conclusion:
#     1. Data is highly positively skewed with skewness of 1.48
#     2. Boxplot shows some serious outliers which is cause of positive skewness
#     3. BP higher than 120 is counted 55, so 55 people due to BP shootup

# In[20]:


sns.distplot(df_train['bloodpressure'])


# In[21]:


df_train['bloodpressure'].skew()


# In[22]:


df_train['bloodpressure'].plot(kind='kde')


# In[23]:


col=df_train[df_train['bloodpressure']>130]
col['bloodpressure'].plot(kind='hist')


# In[24]:


df_train['bloodpressure'].describe()


# In[25]:


df_train['bloodpressure'].plot(kind='box')


# In[26]:


df_train[df_train['bloodpressure']<120]


# In[27]:


df_train[df_train['bloodpressure']<120].count()


# In[28]:


sns.displot(df_train['claim'],kde=True)
plt.show()


# In[29]:


df_train['claim'].plot(kind='kde')


# In[30]:


df_train['claim'].skew()


# In[31]:


df_train['claim'].describe()


# In[32]:


df_train['claim'].plot(kind='box')


# In[33]:


df_train[df_train['claim']>30000].count()


# In[34]:


df_train[df_train['claim']<30000].count()


# In[35]:


stats.iqr(df_train['claim'])


# ### Gender

# ###### Conclusion:
#     1. Male and Female are almost in same quantity

# In[36]:


df_train['gender'].value_counts()


# In[37]:


df_train['gender'].value_counts().plot(kind='bar')


# In[38]:


df_train['gender'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# In[39]:


df_train['gender'].isnull().sum()


# In[40]:


df_train.head()


# ### Diabetic

# ###### Conclusion:
#     1. Most patients are non diabetic but there is no more difference

# In[41]:


df_train['diabetic'].value_counts()


# In[42]:


df_train['diabetic'].value_counts().plot(kind='bar')


# In[43]:


df_train['diabetic'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# ### Children

# ###### Conclusion:
#     1. Most people are children less and number of children is increasing up to 5

# In[44]:


df_train['children'].value_counts()


# In[45]:


df_train['children'].value_counts().plot(kind='bar')


# In[46]:


df_train['children'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# ### Smoker

# ###### Conclusion : 80% persons are not smokers

# In[47]:


df_train['smoker'].value_counts()


# In[48]:


df_train['smoker'].value_counts().plot(kind='bar')


# In[49]:


df_train['smoker'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# ### Region

# ###### Conclusion:
#     The most people are coming from:
#         a. Southeast
#         b. Northwest
#         c. Southwest
#         d. Northwest

# In[50]:


df_train['region'].value_counts()


# In[51]:


df_train['region'].value_counts().plot(kind='bar')


# In[52]:


df_train['region'].value_counts().plot(kind='pie',autopct='%0.1f%%')


# In[53]:


df_train.sample(5)


# ### Scatterplot Analysis

# In[54]:


plt.scatter(df_train['bloodpressure'],df_train['claim'])
plt.title('bloodpressure vs claim')
plt.show()


# In[55]:


correlation = df_train['claim'].corr(df_train['bloodpressure'])
correlation


# In[56]:


plt.scatter(df_train['age'],df_train['claim'])
plt.title('age vs claim')
plt.show()


# In[57]:


correlation = df_train['claim'].corr(df_train['age'])
correlation


# In[58]:


plt.scatter(df_train['bmi'],df_train['claim'])
plt.title('BMI vs claim')
plt.show()


# In[59]:


correlation = df_train['claim'].corr(df_train['bmi'])
correlation


# In[60]:


sns.barplot(x=df_train['bloodpressure'],y=df_train['claim'])
plt.xlabel('bloodpressure')
plt.ylabel('claim')
plt.show()


# ### LinearRegression Test

# In[61]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[62]:


df_train


# In[63]:


region_dummies = pd.get_dummies(df_train['region'])

df_train = pd.concat([df_train.drop('region', axis=1), region_dummies], axis=1)

df_train = df_train.drop(columns =['age', 'gender', 'diabetic', 'children', 'PatientID'])

# Encode smoker column 1 = Yes, 0 = No
label_encoder = LabelEncoder()
categorical_columns = ['smoker', 'northeast', 'northwest', 'southeast', 'southwest']
for column in categorical_columns:
    df_train[column] = label_encoder.fit_transform(df_train[column])


# In[64]:


df_train


# ### Correlation Coefficient & Linearity

# In[65]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Separate features (X) and target variable (y)
X = df_train.drop(columns=['claim'])  # Drop patientID and claim columns for features
y = df_train['claim']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate mean squared error to evaluate the model
mse = mean_squared_error(y_test, predictions)

# Print beautified output
print("\nMean Squared Error (MSE): {:.2f}".format(mse))
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print("{}: {:.2f}".format(feature, coef))
print("\nIntercept: {:.2f}".format(model.intercept_))

# Set the style of the heatmap
sns.set(style="white")  

# Create a figure and a set of subplots
plt.figure(figsize=(10, 8))

# Define your correlation matrix (assuming you have it defined somewhere in your code)
corr_matrix = df_train.corr()

# Customize the heatmap appearance
sns.heatmap(corr_matrix,
            annot=True,  # Annotate each cell with the numeric value
            cmap="coolwarm",  # Set the color map (you can choose any color map you prefer)
            fmt=".2f",  # Format the annotations to two decimal places
            linewidths=.5,  # Width of the lines that divide each cell
            vmin=-1,  # Set the minimum value of the color scale
            vmax=1,   # Set the maximum value of the color scale
            square=True,  # Make the cells square-shaped
            cbar_kws={"shrink": 0.8},  # Shrink the color bar size for better visibility
            annot_kws={"size": 12}  # Set the font size of the annotations
           )

# Set the title and labels for the axes
plt.title("Correlation Matrix", fontsize=16)
plt.xlabel("X Axis Label", fontsize=14)  # Replace "X Axis Label" with your actual x-axis label
plt.ylabel("Y Axis Label", fontsize=14)  # Replace "Y Axis Label" with your actual y-axis label

# Display the heatmap
plt.show()


# In[66]:


# Set the style of the heatmap
sns.set(style="white")  

# Create a figure and a set of subplots
plt.figure(figsize=(10, 8))

# Define your correlation matrix (assuming you have it defined somewhere in your code)
corr_matrix = df_train.corr()

# Customize the heatmap appearance
sns.heatmap(corr_matrix,
            annot=True,  # Annotate each cell with the numeric value
            cmap="coolwarm",  # Set the color map (you can choose any color map you prefer)
            fmt=".2f",  # Format the annotations to two decimal places
            linewidths=.5,  # Width of the lines that divide each cell
            vmin=-1,  # Set the minimum value of the color scale
            vmax=1,   # Set the maximum value of the color scale
            square=True,  # Make the cells square-shaped
            cbar_kws={"shrink": 0.8},  # Shrink the color bar size for better visibility
            annot_kws={"size": 12}  # Set the font size of the annotations
           )

# Set the title and labels for the axes
plt.title("Correlation Matrix", fontsize=16)
plt.xlabel("X Axis Label", fontsize=14)  # Replace "X Axis Label" with your actual x-axis label
plt.ylabel("Y Axis Label", fontsize=14)  # Replace "Y Axis Label" with your actual y-axis label

# Display the heatmap
plt.show()


# In[67]:


# Scatter plot for true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', edgecolors='k', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.show()


# ## Data Transformation
# Due to the suggestion above, doing the Data Transformation for stabilizing the variance and make the data more suitable for linear regression

# In[68]:


# Separate features (X) and target variable (y)
X = df_train.drop(columns=['claim'])  # Drop claim columns for features
y = np.log1p(df_train['claim'])  # Apply log transformation using log(1+x) to avoid log(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Revert log-transformed predictions to original scale for plotting
original_y_test = np.expm1(y_test)
original_predictions = np.expm1(predictions)

# Calculate mean squared error to evaluate the model
mse = mean_squared_error(original_y_test, original_predictions)

# Calculate R-squared
r_squared = r2_score(original_y_test, original_predictions)  # Calculate R-squared

# Scatter plot for true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(original_y_test, original_predictions, color='blue', edgecolors='k', alpha=0.6)
plt.plot([original_y_test.min(), original_y_test.max()], [original_y_test.min(), original_y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values after Log Transformation')
plt.grid(True)
plt.show()


# In[69]:


# Print beautified output
print("\nR Squared value: {:.2f}".format(r_squared))
print("Mean Squared Error (MSE) after log transformation: {:.2f}".format(mse))
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print("{}: {:.2f}".format(feature, coef))
print("\nIntercept: {:.2f}".format(model.intercept_))


# In[70]:


# Print the correlation matrix
corr_matrix = df_train.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# ## Independence
# Using the Durbin-Watsons statistics

# In[71]:


df_train


# In[72]:


from statsmodels.stats.stattools import durbin_watson

# Log transformation of the target variable
df_train['claim'] = np.log(df_train['claim'])

# Selecting features and target
X = df_train.drop(columns=['claim'])  # Drop patientID and claim columns for features
y = df_train['claim']

# Handling categorical variables (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the charges
y_pred = model.predict(X_test)

# Compute residuals
residuals = y_test - y_pred

# Calculate Durbin-Watson statistic
dw_statistic = durbin_watson(residuals)

print(f'Durbin-Watson Statistic: {dw_statistic}')


# #### The Explanation
# The Durbin-Watson statistic value of \(1.8686354311882818\) is close to 2, which is the ideal value suggesting no autocorrelation.
# 
# In more detail:
# 
# - A Durbin-Watson statistic value close to 2 suggests that there is no first-order linear autocorrelation in the residuals.
# - A value between 0 and 2 indicates positive autocorrelation (i.e., the presence of a systematic pattern where a positive error for one observation increases the chance of a positive error for another observation).
# - A value between 2 and 4 indicates negative autocorrelation (i.e., a positive error for one observation increases the chance of a negative error for another observation).
# 
# In your case, the Durbin-Watson statistic value of \(1.8686354311882818\) is very close to 2, indicating that the residuals from your regression model have little to no first-order linear autocorrelation. This is a good indication as it means the assumption of independent errors is nearly met, making the linear regression model more trustworthy.

# In[73]:


# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True)
plt.show()


# # Normality

# ### Chi-Square

# In[74]:


import pandas as pd
from scipy.stats import chisquare

# Assuming df_train has been loaded already

variables_to_test = df_train
for var in variables_to_test:
    observed_values = df_train[var].value_counts().sort_index().values
    num_unique_values = len(df_train[var].unique())
    expected_values = [len(df_train) / num_unique_values] * num_unique_values
    if len(observed_values) == len(expected_values):  # Ensure lengths match before testing
        chi2, p = chisquare(observed_values, expected_values)
        print(f'Variable: {var}')
        print(f'Chi-Square: {chi2}')
        print(f'P-value: {p}')
        print('---' * 10)
    else:
        print(f"Skipped {var} due to shape mismatch between observed and expected values.")


# ### Q-Q plot of standardized residuals

# In[75]:


from scipy.stats import probplot

variables = df_train
for var in variables:
    # Standardize the column values
    standardized_values = (df_train[var] - np.mean(df_train[var])) / np.std(df_train[var])
    
    # Create Q-Q plot
    plt.figure(figsize=(8, 6))
    probplot(standardized_values, dist='norm', plot=plt)
    plt.title(f"Q-Q Plot of Standardized {var}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.show()


# # Residual Homoscedasticity

# In[76]:


# Separate features (X) and target variable (y)
X = df_train.drop(columns=['claim'])  # Drop patientID and claim columns for features
y = np.log1p(df_train['claim'])  # Apply log transformation using log(1+x) to avoid log(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Revert log-transformed predictions to original scale for plotting
original_y_test = np.expm1(y_test)
original_predictions = np.expm1(predictions)

# Calculate mean squared error to evaluate the model
mse = mean_squared_error(original_y_test, original_predictions)

# Calculate residuals
residuals = y_test - predictions

# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='blue', edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True)
plt.show()


# # Removing Outliers

# ### Visualization

# In[77]:


# Reload the file to be analyzed for removing the outliers
df_train = pd.read_csv('input/insurance_data.csv')


# In[78]:


# Creating box plots to visually inspect outliers
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Box plot for 'claim' (target variable)
sns.boxplot(x=df_train['claim'], ax=axes[0])
axes[0].set_title('Box plot of Claim')

# Box plot for 'bmi' (independent variable)
sns.boxplot(x=df_train['bmi'], ax=axes[1])
axes[1].set_title('Box plot of BMI')

# Box plot for 'bloodpressure' (independent variable)
sns.boxplot(x=df_train['bloodpressure'], ax=axes[2])
axes[2].set_title('Box plot of Blood Pressure')

plt.tight_layout()
plt.show()


# In[79]:


def calculate_iqr_outliers(data):
    """
    Calculate the IQR and identify outliers for a given dataset.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    return outliers

# Applying the IQR method to identify outliers
outliers_claim = calculate_iqr_outliers(df_train['claim'])
outliers_bmi = calculate_iqr_outliers(df_train['bmi'])
outliers_bloodpressure = calculate_iqr_outliers(df_train['bloodpressure'])

# Summarizing the number of outliers for each variable
iqr_outliers_summary = pd.DataFrame({
    'Variable': ['Claim', 'BMI', 'Blood Pressure'],
    'Number of Outliers (IQR Method)': [outliers_claim.sum(), outliers_bmi.sum(), outliers_bloodpressure.sum()]
})

iqr_outliers_summary


# In[80]:


# Removing outliers based on Z-score method
outliers_combined = (outliers_claim | outliers_bmi | outliers_bloodpressure)
df_train_cleaned = df_train[~outliers_combined]

df_train_cleaned


# In[81]:


# Preparing the data again
region_dummies_no_outlier = pd.get_dummies(df_train_cleaned['region'])
df_train_cleaned = pd.concat([df_train_cleaned.drop('region', axis=1), region_dummies_no_outlier], axis=1)
df_train_cleaned = df_train_cleaned.drop(columns =['age', 'gender', 'diabetic', 'children', 'PatientID'])

# Encode smoker column 1 = Yes, 0 = No
label_encoder = LabelEncoder()
categorical_columns = ['smoker', 'northeast', 'northwest', 'southeast', 'southwest']
for column in categorical_columns:
    df_train_cleaned[column] = label_encoder.fit_transform(df_train_cleaned[column])


df_train_cleaned


# In[82]:


X_no_outliers = df_train_cleaned[['bloodpressure', 'smoker', 'bmi']].copy()
y_no_outliers = df_train_cleaned['claim']

# Splitting the data into training and testing sets
X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(
    X_no_outliers, y_no_outliers, test_size=0.2, random_state=42
)

# Fit the model on the training data without outliers
linear_regression_model_no_outliers = LinearRegression()
linear_regression_model_no_outliers.fit(X_train_no_outliers, y_train_no_outliers)

# Make predictions on the testing data
y_pred_no_outliers = linear_regression_model_no_outliers.predict(X_test_no_outliers)

# Calculate residuals
residuals_no_outliers = y_test_no_outliers - y_pred_no_outliers

# Create a residual plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_no_outliers, y=residuals_no_outliers)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Without Outliers)')
plt.grid(True)
plt.show()

# Evaluate the model
mse_no_outliers = mean_squared_error(y_test_no_outliers, y_pred_no_outliers)
r2_no_outliers = r2_score(y_test_no_outliers, y_pred_no_outliers)


print("\nMean Squared Error (MSE) no outliers: {:.2f}".format(mse_no_outliers))
print("R square no outliers: {:.2f}".format(r2_no_outliers))


# ### Correlation Coefficient & Linearity

# In[83]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Separate features (X) and target variable (y)
X = df_train_cleaned.drop(columns=['claim'])  # Drop patientID and claim columns for features
y = df_train_cleaned['claim']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate mean squared error to evaluate the model
mse = mean_squared_error(y_test, predictions)

# Print beautified output
print("\nMean Squared Error (MSE): {:.2f}".format(mse))
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print("{}: {:.2f}".format(feature, coef))
print("\nIntercept: {:.2f}".format(model.intercept_))

# Set the style of the heatmap
sns.set(style="white")  

# Create a figure and a set of subplots
plt.figure(figsize=(10, 8))

# Define your correlation matrix (assuming you have it defined somewhere in your code)
corr_matrix = df_train_cleaned.corr()

# Customize the heatmap appearance
sns.heatmap(corr_matrix,
            annot=True,  # Annotate each cell with the numeric value
            cmap="coolwarm",  # Set the color map (you can choose any color map you prefer)
            fmt=".2f",  # Format the annotations to two decimal places
            linewidths=.5,  # Width of the lines that divide each cell
            vmin=-1,  # Set the minimum value of the color scale
            vmax=1,   # Set the maximum value of the color scale
            square=True,  # Make the cells square-shaped
            cbar_kws={"shrink": 0.8},  # Shrink the color bar size for better visibility
            annot_kws={"size": 12}  # Set the font size of the annotations
           )

# Set the title and labels for the axes
plt.title("Correlation Matrix", fontsize=16)
plt.xlabel("X Axis Label", fontsize=14)  # Replace "X Axis Label" with your actual x-axis label
plt.ylabel("Y Axis Label", fontsize=14)  # Replace "Y Axis Label" with your actual y-axis label

# Display the heatmap
plt.show()


# In[84]:


# Set the style of the heatmap
sns.set(style="white")  

# Create a figure and a set of subplots
plt.figure(figsize=(10, 8))

# Define your correlation matrix (assuming you have it defined somewhere in your code)
corr_matrix = df_train_cleaned.corr()

# Customize the heatmap appearance
sns.heatmap(corr_matrix,
            annot=True,  # Annotate each cell with the numeric value
            cmap="coolwarm",  # Set the color map (you can choose any color map you prefer)
            fmt=".2f",  # Format the annotations to two decimal places
            linewidths=.5,  # Width of the lines that divide each cell
            vmin=-1,  # Set the minimum value of the color scale
            vmax=1,   # Set the maximum value of the color scale
            square=True,  # Make the cells square-shaped
            cbar_kws={"shrink": 0.8},  # Shrink the color bar size for better visibility
            annot_kws={"size": 12}  # Set the font size of the annotations
           )

# Set the title and labels for the axes
plt.title("Correlation Matrix", fontsize=16)
plt.xlabel("X Axis Label", fontsize=14)  # Replace "X Axis Label" with your actual x-axis label
plt.ylabel("Y Axis Label", fontsize=14)  # Replace "Y Axis Label" with your actual y-axis label

# Display the heatmap
plt.show()


# In[85]:


# Scatter plot for true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', edgecolors='k', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.show()


# ## Data Transformation
# Due to the suggestion above, doing the Data Transformation for stabilizing the variance and make the data more suitable for linear regression

# In[86]:


# Separate features (X) and target variable (y)
X = df_train_cleaned.drop(columns=['claim'])  # Drop claim columns for features
y = np.log1p(df_train_cleaned['claim'])  # Apply log transformation using log(1+x) to avoid log(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Revert log-transformed predictions to original scale for plotting
original_y_test = np.expm1(y_test)
original_predictions = np.expm1(predictions)

# Calculate mean squared error to evaluate the model
mse = mean_squared_error(original_y_test, original_predictions)

# Calculate R-squared
r_squared = r2_score(original_y_test, original_predictions)  # Calculate R-squared

# Scatter plot for true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(original_y_test, original_predictions, color='blue', edgecolors='k', alpha=0.6)
plt.plot([original_y_test.min(), original_y_test.max()], [original_y_test.min(), original_y_test.max()], 'k--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values after Log Transformation')
plt.grid(True)
plt.show()


# In[87]:


# Print beautified output
print("\nR Squared value: {:.2f}".format(r_squared))
print("Mean Squared Error (MSE) after log transformation: {:.2f}".format(mse))
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print("{}: {:.2f}".format(feature, coef))
print("\nIntercept: {:.2f}".format(model.intercept_))


# In[88]:


# Print the correlation matrix
corr_matrix = df_train_cleaned.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# ## Independence
# Using the Durbin-Watsons statistics

# In[89]:


df_train_cleaned


# In[90]:


from statsmodels.stats.stattools import durbin_watson

# Log transformation of the target variable
df_train_cleaned['claim'] = np.log(df_train_cleaned['claim'])

# Selecting features and target
X = df_train_cleaned.drop(columns=['claim'])  # Drop patientID and claim columns for features
y = df_train_cleaned['claim']

# Handling categorical variables (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the charges
y_pred = model.predict(X_test)

# Compute residuals
residuals = y_test - y_pred

# Calculate Durbin-Watson statistic
dw_statistic = durbin_watson(residuals)

print(f'Durbin-Watson Statistic: {dw_statistic}')


# #### The Explanation
# The Durbin-Watson statistic value of \(1.8686354311882818\) is close to 2, which is the ideal value suggesting no autocorrelation.
# 
# In more detail:
# 
# - A Durbin-Watson statistic value close to 2 suggests that there is no first-order linear autocorrelation in the residuals.
# - A value between 0 and 2 indicates positive autocorrelation (i.e., the presence of a systematic pattern where a positive error for one observation increases the chance of a positive error for another observation).
# - A value between 2 and 4 indicates negative autocorrelation (i.e., a positive error for one observation increases the chance of a negative error for another observation).
# 
# In your case, the Durbin-Watson statistic value of \(1.8686354311882818\) is very close to 2, indicating that the residuals from your regression model have little to no first-order linear autocorrelation. This is a good indication as it means the assumption of independent errors is nearly met, making the linear regression model more trustworthy.

# In[91]:


# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True)
plt.show()


# # Normality

# ### Chi-Square

# In[92]:


import pandas as pd
from scipy.stats import chisquare

# Assuming df_train has been loaded already

variables_to_test = df_train_cleaned
for var in variables_to_test:
    observed_values = df_train_cleaned[var].value_counts().sort_index().values
    num_unique_values = len(df_train_cleaned[var].unique())
    expected_values = [len(df_train_cleaned) / num_unique_values] * num_unique_values
    if len(observed_values) == len(expected_values):  # Ensure lengths match before testing
        chi2, p = chisquare(observed_values, expected_values)
        print(f'Variable: {var}')
        print(f'Chi-Square: {chi2}')
        print(f'P-value: {p}')
        print('---' * 10)
    else:
        print(f"Skipped {var} due to shape mismatch between observed and expected values.")


# ### Q-Q plot of standardized residuals

# In[93]:


from scipy.stats import probplot

variables = df_train_cleaned
for var in variables:
    # Standardize the column values
    standardized_values = (df_train_cleaned[var] - np.mean(df_train_cleaned[var])) / np.std(df_train_cleaned[var])
    
    # Create Q-Q plot
    plt.figure(figsize=(8, 6))
    probplot(standardized_values, dist='norm', plot=plt)
    plt.title(f"Q-Q Plot of Standardized {var}")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.show()


# # Residual Homoscedasticity

# In[94]:


# Separate features (X) and target variable (y)
X = df_train_cleaned.drop(columns=['claim'])  # Drop patientID and claim columns for features
y = np.log1p(df_train_cleaned['claim'])  # Apply log transformation using log(1+x) to avoid log(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Revert log-transformed predictions to original scale for plotting
original_y_test = np.expm1(y_test)
original_predictions = np.expm1(predictions)

# Calculate mean squared error to evaluate the model
mse = mean_squared_error(original_y_test, original_predictions)

# Calculate residuals
residuals = y_test - predictions

# Plot residuals against fitted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, color='blue', edgecolors='k', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True)
plt.show()


# ###### Remove Outliers and Save the Data

# In[95]:


# Read the CSV file into a DataFrame
df = pd.read_csv('input/insurance_data.csv')


# In[96]:


df


# In[97]:


total = df.isnull().sum()
percent = (total / len(df)) * 100
missing_values = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_values)
df_train = df.drop('index',axis=1)

mean_age = np.mean(df['age'])
# Fill null values in the 'age' column with the mean age
df['age'].fillna(mean_age, inplace=True)

# Check if there are any missing values in the 'age' column after filling
missing_values = df['age'].isnull().sum()
print("Missing values in 'age' column after filling:", missing_values)

df.dropna(subset=['region'],inplace=True)

# Check if there are any missing values in the 'region' column after droping
missing_values = df['region'].isnull().sum()
print("Missing values in 'region' column after filling:", missing_values)

df.describe()


# In[98]:


df


# In[99]:


region_dummies = pd.get_dummies(df['region'])

df = pd.concat([df.drop('region', axis=1), region_dummies], axis=1)

# Encode smoker column 1 = Yes, 0 = No
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'smoker', 'diabetic', 'northeast', 'northwest', 'southeast', 'southwest']
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])


# In[100]:


df


# In[101]:


def calculate_iqr_outliers(data):
    """
    Calculate the IQR and identify outliers for a given dataset.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    return outliers

# Applying the IQR method to identify outliers
outliers_claim = calculate_iqr_outliers(df['claim'])
outliers_bmi = calculate_iqr_outliers(df['bmi'])
outliers_bloodpressure = calculate_iqr_outliers(df['bloodpressure'])

# Summarizing the number of outliers for each variable
iqr_outliers_summary = pd.DataFrame({
    'Variable': ['Claim', 'BMI', 'Blood Pressure'],
    'Number of Outliers (IQR Method)': [outliers_claim.sum(), outliers_bmi.sum(), outliers_bloodpressure.sum()]
})

iqr_outliers_summary


# In[102]:


outliers_combined = (outliers_claim | outliers_bmi | outliers_bloodpressure)
df_train_cleaned = df[~outliers_combined]

df_train_cleaned


# In[103]:


# Save the cleaned data to a new CSV file
df_train_cleaned.to_csv('input/cleaned_data.csv', index=False)


# In[104]:


df = pd.read_csv('cleaned_data.csv')


# In[105]:


df

