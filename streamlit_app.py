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

# Age Distribution
st.write("### Age Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(covid_data['AGE'].dropna(), bins=30, kde=True, color='blue', ax=ax)
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Death Outcome Distribution
st.write("### Death Outcome Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='death', hue='death', data=covid_data, palette='Set2', ax=ax, legend=False)
ax.set_title('Death Outcome Distribution')
ax.set_xlabel('Death (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
st.pyplot(fig)

# Sex Distribution
st.write("### Sex Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='death', hue='SEX', data=covid_data, palette='Set1', ax=ax, legend=False)
ax.set_title('Distribution of Sex')
ax.set_xlabel('Sex (1 = Male, 2 = Female)')
ax.set_ylabel('Count')
st.pyplot(fig)

# ICU Admission vs Death Outcome
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

# Medical Conditions Distribution
st.write("### Medical Conditions Distribution")
conditions = ['DIABETES', 'PNEUMONIA', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 
              'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO']

# Sum of conditions and sort
covid_data_conditions = covid_data[conditions].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=covid_data_conditions.index, y=covid_data_conditions.values, palette='viridis', ax=ax)
ax.set_title('Distribution of Medical Conditions in Patients')
ax.set_xlabel('Condition')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Age vs Death Boxplot
st.write("### Age Distribution by Death Outcome")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='death', y='AGE', data=covid_data, palette='Set3', ax=ax)
ax.set_title('Age Distribution for Death vs Survival')
ax.set_xlabel('Death (0 = Survived, 1 = Died)')
ax.set_ylabel('Age')
st.pyplot(fig)

# Pregnancy vs Death Outcome (Filtered by Sex)
st.write("### Pregnancy vs Death Outcome (Females Only)")
# Filter the data for females (SEX = 2)
female_data = covid_data[covid_data['SEX'] == 2]

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PREGNANT', hue='death', data=female_data, palette='Set2', ax=ax)
ax.set_title('Pregnancy vs Death Outcome (Females Only)')
ax.set_xlabel('Pregnant (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
st.pyplot(fig)

def plot_feature_death_rate(feature):
    feature_data = covid_data[[feature, 'death']].dropna()
    
    if set(feature_data[feature].unique()) == {0, 1}:  # Ensure binary data
        death_rate = feature_data.groupby(feature)['death'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=feature, y='death', data=death_rate, palette='coolwarm', ax=ax)
        ax.set_title(f'Death Rate by {feature}')
        ax.set_xlabel(f'{feature} (0 = No, 1 = Yes)')
        ax.set_ylabel('Death Rate')
        return fig
    else:
        st.write(f"Feature '{feature}' does not have binary values (0 or 1) and cannot be plotted.")
        return None

# Plotting death rate for each feature individually

# DIABETES
st.write("### Death Rate by DIABETES")
fig_diabetes = plot_feature_death_rate('DIABETES')
if fig_diabetes:
    st.pyplot(fig_diabetes)

# PNEUMONIA
st.write("### Death Rate by PNEUMONIA")
fig_pneumonia = plot_feature_death_rate('PNEUMONIA')
if fig_pneumonia:
    st.pyplot(fig_pneumonia)

# COPD
st.write("### Death Rate by COPD")
fig_copd = plot_feature_death_rate('COPD')
if fig_copd:
    st.pyplot(fig_copd)

# ASTHMA
st.write("### Death Rate by ASTHMA")
fig_asthma = plot_feature_death_rate('ASTHMA')
if fig_asthma:
    st.pyplot(fig_asthma)

# IMMUNOSUPPRESSION (INMSUPR)
st.write("### Death Rate by INMSUPR (Immunosuppression)")
fig_inmsupr = plot_feature_death_rate('INMSUPR')
if fig_inmsupr:
    st.pyplot(fig_inmsupr)

# HYPERTENSION
st.write("### Death Rate by HYPERTENSION")
fig_hypertension = plot_feature_death_rate('HIPERTENSION')
if fig_hypertension:
    st.pyplot(fig_hypertension)

# CARDIOVASCULAR DISEASE
st.write("### Death Rate by CARDIOVASCULAR Disease")
fig_cardiovascular = plot_feature_death_rate('CARDIOVASCULAR')
if fig_cardiovascular:
    st.pyplot(fig_cardiovascular)

# OBESITY
st.write("### Death Rate by OBESITY")
fig_obesity = plot_feature_death_rate('OBESITY')
if fig_obesity:
    st.pyplot(fig_obesity)

# RENAL CHRONIC DISEASE
st.write("### Death Rate by RENAL_CHRONIC (Chronic Kidney Disease)")
fig_renal_chronic = plot_feature_death_rate('RENAL_CHRONIC')
if fig_renal_chronic:
    st.pyplot(fig_renal_chronic)

# TOBACCO USE
st.write("### Death Rate by TOBACCO")
fig_tobacco = plot_feature_death_rate('TOBACCO')
if fig_tobacco:
    st.pyplot(fig_tobacco)

# ICU Admission
st.write("### Death Rate by ICU Admission")
fig_icu = plot_feature_death_rate('ICU')
if fig_icu:
    st.pyplot(fig_icu)

# PREGNANCY
st.write("### Death Rate by PREGNANCY")
fig_pregnant = plot_feature_death_rate('PREGNANT')
if fig_pregnant:
    st.pyplot(fig_pregnant)

        
# Pregnancy vs Death Outcome (Filtered by Sex)
st.write("### Pregnancy vs Death Outcome (Females Only)")
# Filter the data for females (SEX = 2)
female_data = covid_data[covid_data['SEX'] == 2]

fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='PREGNANT', hue='death', data=female_data, palette='Set2', ax=ax)
ax.set_title('Pregnancy vs Death Outcome (Females Only)')
ax.set_xlabel('Pregnant (0 = No, 1 = Yes)')
ax.set_ylabel('Count')
st.pyplot(fig)
