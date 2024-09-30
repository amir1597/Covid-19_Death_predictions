import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix 
from sklearn.model_selection import train_test_split

# Title for the app
st.title("COVID-19 Death Prediction")

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
with st.expander("Dataset Preview", expanded=True):
    st.dataframe(covid_data.head())

# Handle missing and incorrect data (converting date columns)
with st.expander("Exploratory Data Analysis (EDA)", expanded=True):
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

    # Cleaning the data
    conditions = [
        'PNEUMONIA', 
        'DIABETES', 
        'COPD', 
        'ASTHMA', 
        'INMSUPR', 
        'HIPERTENSION', 
        'OTHER_DISEASE', 
        'CARDIOVASCULAR', 
        'OBESITY', 
        'RENAL_CHRONIC', 
        'TOBACCO'
    ]

    for condition in conditions:
        covid_data = covid_data[(covid_data[condition] == 1) | (covid_data[condition] == 2)]

    st.write("### After Cleaning the Data")
    st.dataframe(covid_data.head())

# Data Visualization Section
with st.expander("Data Visualization", expanded=True):
    # Age Distribution
    st.write("### Age Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(covid_data['AGE'].dropna(), bins=30, kde=True, color='blue', ax=ax)
    ax.set_title('Age Distribution')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Features to visualize
    features = {
        'DIABETES': 'Diabetes',
        'PNEUMONIA': 'Pneumonia',
        'COPD': 'COPD',
        'ASTHMA': 'Asthma',
        'INMSUPR': 'Immunosuppression',
        'HIPERTENSION': 'Hypertension',
        'CARDIOVASCULAR': 'Cardiovascular Disease',
        'OBESITY': 'Obesity',
        'RENAL_CHRONIC': 'Renal Chronic Disease',
        'TOBACCO': 'Tobacco Use',
        'PREGNANT' : 'PREGNANT',
        'INTUBED' : 'INTUBED',
        'ICU' : 'ICU'
    }

    for feature, title in features.items():
        st.write(f"### {title} vs Death Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=feature, hue='death', data=covid_data, palette='Set2', ax=ax)
        ax.set_title(f'{title} vs Death Outcome')
        ax.set_xlabel(f'{title} (1 = Yes, 2 = No)')
        ax.set_ylabel('Count')
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
    sns.countplot(x='SEX', hue='SEX', data=covid_data, palette='Set1', ax=ax, legend=False)
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
    st.write("### Conclusion")
    st.write("#### After all this Visualization we see that we dont need died_date for our model and we will drop the  icu and  INTUBED because they have to much missing values and for PREGNANT we will drop the missing values ")
    covid_data.drop(columns=["INTUBED","ICU","DATE_DIED"],
     inplace=True)
    
with st.sidebar():

    st.write("### Input Data for Prediction")
    
    # Define input fields for the model
    age = st.slider("Age", 0, 100, 30)
    pneumonia = st.selectbox("Pneumonia (1 = Yes, 2 = No)", (1, 2))
    diabetes = st.selectbox("Diabetes (1 = Yes, 2 = No)", (1, 2))
    copd = st.selectbox("COPD (1 = Yes, 2 = No)", (1, 2))
    asthma = st.selectbox("Asthma (1 = Yes, 2 = No)", (1, 2))
    inmsupr = st.selectbox("Immunosuppression (1 = Yes, 2 = No)", (1, 2))
    hypertension = st.selectbox("Hypertension (1 = Yes, 2 = No)", (1, 2))
    cardiovascular = st.selectbox("Cardiovascular Disease (1 = Yes, 2 = No)", (1, 2))
    obesity = st.selectbox("Obesity (1 = Yes, 2 = No)", (1, 2))
    renal_chronic = st.selectbox("Renal Chronic Disease (1 = Yes, 2 = No)", (1, 2))
    tobacco = st.selectbox("Tobacco Use (1 = Yes, 2 = No)", (1, 2))
    
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'AGE': [age],
        'PNEUMONIA': [pneumonia],
        'DIABETES': [diabetes],
        'COPD': [copd],
        'ASTHMA': [asthma],
        'INMSUPR': [inmsupr],
        'HIPERTENSION': [hypertension],
        'CARDIOVASCULAR': [cardiovascular],
        'OBESITY': [obesity],
        'RENAL_CHRONIC': [renal_chronic],
        'TOBACCO': [tobacco]
    })

    # Prepare features for training
    x = covid_data.drop(columns=['death', 'ID'])
    y = covid_data['death']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Button to make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.success("Predicted Outcome: Death (1)")
        else:
            st.success("Predicted Outcome: Survival (0)")