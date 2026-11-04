# Step 1: Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# Step 2: Create dataset

data = {
    'Salary': [20000, 25000, 30000, 35000, 40000,
               45000, 50000, 55000, 60000, 500000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


# Step 3: Detect outliers using IQR

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_IQR = df[
    (df['Salary'] < lower) |
    (df['Salary'] > upper)
]

print("\nOutliers Using IQR:")
print(outliers_IQR)


# Step 4: Detect outliers using Z-Score

df['Z_score'] = zscore(df['Salary'])

outliers_Z = df[
    (df['Z_score'] > 3) |
    (df['Z_score'] < -3)
]

print("\nOutliers Using Z-Score:")
print(outliers_Z)


# Step 5: Treat outliers using IQR Capping

df['Treated_Salary'] = df['Salary'].clip(
    lower=lower,
    upper=upper
)

print("\nDataset After Outlier Treatment:")
print(df[['Salary', 'Treated_Salary']])


# Step 6: Visualize before and after treatment

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=df['Salary'])
plt.title("Before Outlier Treatment")

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Treated_Salary'])
plt.title("After Outlier Treatment")

plt.tight_layout()
plt.show()


# Step 7: Final result

print("\nOriginal Salary Statistics:")
print(df['Salary'].describe())

print("\nTreated Salary Statistics:")
print(df['Treated_Salary'].describe())

print("\nOutlier Detection and Treatment Completed!")
