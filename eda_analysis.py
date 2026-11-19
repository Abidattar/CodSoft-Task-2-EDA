
# CODSOFT DATA ANALYST INTERNSHIP
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# STEP 1: Load Dataset

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "cleaned_dataset.csv"

df = pd.read_csv(file_path)

print("\n--- DATASET LOADED SUCCESSFULLY ---")
print(df)

# STEP 2: Examine Dataset

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET INFORMATION ---")
df.info()

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# STEP 3: Descriptive Statistics

print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe())

print("\n--- CATEGORICAL SUMMARY ---")
print(df.describe(include='object'))

# STEP 4: Check Missing Values

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# STEP 5: Analyze Employee Age

print("\n--- AGE ANALYSIS ---")
print("Average Age:", round(df['Age'].mean(), 2))
print("Minimum Age:", df['Age'].min())
print("Maximum Age:", df['Age'].max())
print("Median Age:", df['Age'].median())

# STEP 6: Analyze Salary

print("\n--- SALARY ANALYSIS ---")
print("Average Salary:", round(df['Salary'].mean(), 2))
print("Minimum Salary:", df['Salary'].min())
print("Maximum Salary:", df['Salary'].max())
print("Median Salary:", df['Salary'].median())

# STEP 7: Employee Distribution by City

print("\n--- EMPLOYEES BY CITY ---")
city_count = df['City'].value_counts()
print(city_count)

# STEP 8: Average Salary by City

print("\n--- AVERAGE SALARY BY CITY ---")
city_salary = df.groupby('City')['Salary'].mean().sort_values(
    ascending=False
)
print(city_salary)

# STEP 9: Salary Analysis by Gender

print("\n--- SALARY BY GENDER ---")
gender_salary = df.groupby('Gender')['Salary'].agg(
    ['count', 'mean', 'median']
)
print(gender_salary)

# STEP 10: Identify Highest Paid Employee

highest_paid = df.loc[df['Salary'].idxmax()]

print("\n--- HIGHEST PAID EMPLOYEE ---")
print("Name:", highest_paid['Name'])
print("City:", highest_paid['City'])
print("Salary:", highest_paid['Salary'])

# STEP 11: Identify Outliers Using IQR

def find_outliers(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    print(f"\n--- OUTLIERS IN {column} ---")
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

    if outliers.empty:
        print("No outliers found.")
    else:
        print(outliers[['Name', column]])

    return outliers


age_outliers = find_outliers(df, 'Age')
salary_outliers = find_outliers(df, 'Salary')

# STEP 12: Relationship Between Age and Salary

print("\n--- AGE AND SALARY CORRELATION ---")

correlation = df['Age'].corr(df['Salary'])

print("Correlation:", round(correlation, 2))

# STEP 13: Prepare Output Folder

output_dir = BASE_DIR / "eda_outputs"
output_dir.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# STEP 14: Salary Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df['Salary'], bins=5, kde=True)

plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig(output_dir / "salary_distribution.png")
plt.show()

# STEP 15: Employee Distribution by City

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x='City')

plt.title("Employee Distribution by City")
plt.xlabel("City")
plt.ylabel("Number of Employees")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(output_dir / "employee_by_city.png")
plt.show()

# STEP 16: Average Salary by City

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x='City',
    y='Salary',
    errorbar=None
)

plt.title("Average Salary by City")
plt.xlabel("City")
plt.ylabel("Average Salary")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(output_dir / "average_salary_by_city.png")
plt.show()

# STEP 17: Age and Salary Relationship

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x='Age',
    y='Salary',
    hue='Gender',
    s=100
)

plt.title("Relationship Between Age and Salary")
plt.xlabel("Age")
plt.ylabel("Salary")

plt.tight_layout()
plt.savefig(output_dir / "age_salary_relationship.png")
plt.show()

# STEP 18: Correlation Heatmap

plt.figure(figsize=(7, 5))

correlation_matrix = df[['Age', 'Salary']].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Age and Salary Correlation Heatmap")

plt.tight_layout()
plt.savefig(output_dir / "correlation_heatmap.png")
plt.show()

# STEP 19: Business Findings

top_city = city_salary.idxmax()
top_city_salary = city_salary.max()

average_salary = df['Salary'].mean()
average_age = df['Age'].mean()

print("\n========== BUSINESS FINDINGS ==========")

print("Total Employees:", len(df))

print("Average Employee Age:", round(average_age, 2))

print("Average Employee Salary:", round(average_salary, 2))

print("Highest Average Salary City:", top_city)

print("Highest Average City Salary:", round(top_city_salary, 2))

print("Highest Paid Employee:", highest_paid['Name'])

print("Highest Individual Salary:", highest_paid['Salary'])

print("Age-Salary Correlation:", round(correlation, 2))

print("Salary Outliers:", len(salary_outliers))

# STEP 20: Generate EDA Report

report = f"""
CODSOFT DATA ANALYST INTERNSHIP
TASK 2: EXPLORATORY DATA ANALYSIS

1. DATASET OVERVIEW

Total Employees: {len(df)}
Total Columns: {len(df.columns)}

2. AGE ANALYSIS

Average Age: {average_age:.2f}
Minimum Age: {df['Age'].min()}
Maximum Age: {df['Age'].max()}

3. SALARY ANALYSIS

Average Salary: {average_salary:.2f}
Minimum Salary: {df['Salary'].min()}
Maximum Salary: {df['Salary'].max()}

4. CITY ANALYSIS

City with highest average salary: {top_city}
Average salary in that city: {top_city_salary:.2f}

Employee distribution by city:
{city_count.to_string()}

5. GENDER ANALYSIS

{gender_salary.to_string()}

6. OUTLIER ANALYSIS

Age outliers: {len(age_outliers)}
Salary outliers: {len(salary_outliers)}

Salary outlier records:
{salary_outliers[['Name', 'Salary']].to_string(index=False)}

7. RELATIONSHIP ANALYSIS

Correlation between Age and Salary: {correlation:.2f}

8. CONCLUSION

The dataset was examined using descriptive statistics,
visualizations, and outlier detection.

Salary distribution, employee distribution by city,
gender-based salary summaries, and the relationship
between age and salary were analyzed.

The analysis provides an overview of the employee
dataset and identifies unusual salary values.

Note: This dataset contains only a small sample of
employees. Findings should not be generalized to
a larger workforce.
"""

report_path = output_dir / "EDA_Report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

print("\nEDA REPORT SAVED SUCCESSFULLY!")

print("Report location:", report_path)

print("\nALL EDA TASKS COMPLETED SUCCESSFULLY!")
