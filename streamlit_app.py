import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Title for the app
st.title("Medical Data Visualization - COVID-19")

# Load dataset with caching for performance
@st.cache_data
def load_data():
    try:
        covid_data = pd.read_csv('Covid_data.csv')
        return covid_data
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        st.stop()

covid_data = load_data()

# Display dataset preview
st.write("### Dataset Preview")
st.dataframe(covid_data.head())

# Handle missing and incorrect data (converting date columns)
st.write("### EDA")
covid_data['ID'] = covid_data.index

# Handle 'DATE_DIED' column and create 'death' indicator
covid_data['DATE_DIED'] = pd.to_datetime(covid_data['DATE_DIED'], errors='coerce')
covid_data['death'] = covid_data['DATE_DIED'].apply(lambda x: 1 if pd.notnull(x) else 0)

# Display missing values and duplicated rows count
st.write("Missing Values in Each Column:")
st.write(covid_data.isnull().sum())

st.write("Number of Duplicated Rows:")
st.write(covid_data.duplicated().sum())

# Display basic statistics of the dataset
st.write("### Basic Statistics")
st.write(covid_data.describe())

# Create count plots for each feature

# PREGNANT
st.write("### Pregnancy vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PREGNANT', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Pregnancy vs Death Outcome')
ax.set_xlabel('Pregnant (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# DIABETES
st.write("### Diabetes vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='DIABETES', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Diabetes vs Death Outcome')
ax.set_xlabel('Diabetes (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# PNEUMONIA
st.write("### Pneumonia vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PNEUMONIA', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Pneumonia vs Death Outcome')
ax.set_xlabel('Pneumonia (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# COPD
st.write("### COPD vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='COPD', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('COPD vs Death Outcome')
ax.set_xlabel('COPD (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# ASTHMA
st.write("### Asthma vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='ASTHMA', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Asthma vs Death Outcome')
ax.set_xlabel('Asthma (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# IMMUNOSUPPRESSION (INMSUPR)
st.write("### Immunosuppression vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='INMSUPR', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Immunosuppression vs Death Outcome')
ax.set_xlabel('Immunosuppression (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# HYPERTENSION
st.write("### Hypertension vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='HIPERTENSION', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Hypertension vs Death Outcome')
ax.set_xlabel('Hypertension (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# CARDIOVASCULAR DISEASE
st.write("### Cardiovascular Disease vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='CARDIOVASCULAR', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Cardiovascular Disease vs Death Outcome')
ax.set_xlabel('Cardiovascular Disease (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# OBESITY
st.write("### Obesity vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='OBESITY', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Obesity vs Death Outcome')
ax.set_xlabel('Obesity (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# RENAL CHRONIC DISEASE
st.write("### Renal Chronic Disease vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='RENAL_CHRONIC', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Renal Chronic Disease vs Death Outcome')
ax.set_xlabel('Renal Chronic Disease (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# TOBACCO USE
st.write("### Tobacco Use vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='TOBACCO', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('Tobacco Use vs Death Outcome')
ax.set_xlabel('Tobacco Use (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)

# ICU Admission
st.write("### ICU Admission vs Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='ICU', hue='death', data=covid_data, palette='Set2', ax=ax)
ax.set_title('ICU Admission vs Death Outcome')
ax.set_xlabel('ICU Admission (1 = Yes, 2 = No)')
ax.set_ylabel('Count')
st.pyplot(fig)
