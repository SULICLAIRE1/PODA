# %%
%pip install seaborn

# %%
%pip install matplotlib

# %%
# loading data
%pip install pandas
import pandas as pd

# Load the dataset
data = pd.read_csv('/Insurance Claim.csv')

# Display the first few rows of the dataset
data.head()


# %%
# For our multivariate linear regression analysis, we'll focus on the columns: bmi, bloodpressure, smoker, region, and claim.
# Now, let's proceed with the following steps:
# Remove any null values from the dataset.
# Remove outliers using the IQR method for the columns: bmi and bloodpressure (assuming we are only considering numerical columns for outlier removal).

# %%
# Remove rows with null values in the specified columns
data_cleaned = data[['bmi', 'bloodpressure', 'smoker', 'region', 'claim']].dropna()

# Display the shape of the original and cleaned data to see how many rows were removed
original_shape = data.shape[0]
cleaned_shape = data_cleaned.shape[0]

original_shape, cleaned_shape


# %%
def remove_outliers_iqr(df, column):
    """Remove outliers using IQR method for a specified column."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers for 'bmi' and 'bloodpressure' columns
data_cleaned = remove_outliers_iqr(data_cleaned, 'bmi')
data_cleaned = remove_outliers_iqr(data_cleaned, 'bloodpressure')

# Display the shape of the data after outlier removal
data_cleaned.shape[0]


# %%
%pip install statsmodels

# %%
import statsmodels.api as sm

# Convert categorical variables to dummy variables
data_encoded = pd.get_dummies(data_cleaned, dtype=float,drop_first=True)  # Drop first column to avoid multicollinearity

print(data_encoded)


# %%
# Define independent (X) and dependent (y) variables
import numpy as np
X = data_encoded.drop('claim', axis=1)
y = data_encoded['claim']

# Add a constant to the model (it's a best practice to include an intercept in the model)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
model.summary()

# %%
print(X.shape)
print(X)

# %%
from statsmodels.stats.diagnostic import het_goldfeldquandt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 3. Test for Homoscedasticity
f_stat, p_value, _ = het_goldfeldquandt(y, X)
homoscedasticity_p_value = p_value

# 4. Q-Q plot for Normality of Residuals
plt.figure(figsize=(8, 6))
qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# 5. Histogram for Normality of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.show()

# 6. Check for Multicollinearity
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_data, homoscedasticity_p_value



