import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load dataset
st.title("Medical Data Visualization")

covid_data = pd.read_csv('Covid_data.csv')

st.write("Dataset Preview:")
st.dataframe(covid_data.head())

st.write("EDA")
covid_data['ID'] = covid_data.index
covid_data['death'] = covid_data['DATE_DIED'].apply(lambda x: 1 if x != '9999-99-99' else 0)
st.write(covid_data.isnull().sum())
st.write(covid_data.duplicated().sum())
st.write("Basic Statistics:")
st.write(covid_data.describe())

# Age Distribution
st.write("### Age Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(covid_data['AGE'], bins=30, kde=True, color='blue', ax=ax)
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Death Rate
st.write("### Death Outcome Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Death Outcome Distribution')
ax.set_xlabel('Death (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
st.pyplot(fig)

# Sex Distribution
st.write("### Sex Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='SEX', data=covid_data, palette='Set1', ax=ax)
ax.set_title('Distribution of Sex')
ax.set_xlabel('Sex (1 = Male, 2 = Female)')
ax.set_ylabel('Count')
st.pyplot(fig)

# ICU vs Death
st.write("### ICU Admission vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='ICU', hue='death', data=covid_data, palette='coolwarm', ax=ax)
ax.set_title('ICU Admission vs Death Outcome')
ax.set_xlabel('ICU Admission (1 = Yes, 0 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
correlation_matrix = covid_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)

# Conditions Bar Plot
st.write("### Medical Conditions Distribution")
conditions = ['DIABETES', 'PNEUMONIA', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 
              'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']
covid_data_conditions = covid_data[conditions].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=covid_data_conditions.index, y=covid_data_conditions.values, palette='viridis', ax=ax)
ax.set_title('Distribution of Medical Conditions in Patients')
ax.set_xlabel('Condition')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Boxplot for Age vs Death
st.write("### Age vs Death")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='death', y='AGE', data=covid_data, palette='Set3', ax=ax)
ax.set_title('Age Distribution for Death vs Survival')
ax.set_xlabel('Death (0 = Survived, 1 = Died)')
ax.set_ylabel('Age')
st.pyplot(fig)

# Pregnancy vs Death
st.write("### Pregnancy vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PREGNANT', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Pregnancy vs Death Outcome')
ax.set_xlabel('Pregnant (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
st.pyplot(fig)

