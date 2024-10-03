import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Title for the app
st.title("COVID-19 Death Prediction")

# Load dataset with caching for performance
@st.cache_data
def load_data():
    try:
        covid_data = pd.read_csv('Covid_19_data.csv')
        return covid_data
    except FileNotFoundError:
        st.error("Dataset not found. Please check the file path.")
        st.stop()

covid_data = load_data()

# Display dataset preview
with st.expander("Dataset Preview", expanded=True):
    st.dataframe(covid_data.head())

# EDA section
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

# Visualization section
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
        'PREGNANT': 'Pregnant',
        'INTUBED': 'Intubed',
        'ICU': 'ICU'
    }

    for feature, title in features.items():
        st.write(f"### {title} vs Death Outcome")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x=feature, hue='death', data=covid_data, palette='Set2', ax=ax)
        ax.set_title(f'{title} vs Death Outcome')
        ax.set_xlabel(f'{title} (1 = Yes, 2 = No)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    correlation_matrix = covid_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    covid_data.drop(columns=['Unnamed: 0',"INTUBED","ICU","DATE_DIED","SEX","PREGNANT","COPD","ASTHMA","INMSUPR","OTHER_DISEASE","CARDIOVASCULAR","OBESITY","TOBACCO"], inplace=True)

st.dataframe(covid_data.head())
# Cache the model training so it runs only once

x = covid_data.drop(columns=['death', 'ID',])
y = covid_data['death']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
model = LogisticRegression()
model.fit(x_train, y_train)

# Train the model and keep it cached
y_pred = model.predict(x_test)

auc_score =  accuracy_score(y_test ,y_pred)
auc_score
# Sidebar for user inputs using session state
st.sidebar.title("Input Data for Prediction")

# Define input fields for the model with st.session_state to persist values
def update_session_state(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

usmer = st.sidebar.selectbox("USMER (1 = Yes, 2 = No)", (1, 2), index=update_session_state('USMER', 0))
medical_unit = st.sidebar.selectbox("MEDICAL_UNIT (1 = Yes, 2 = No)", (1, 2), index=update_session_state('MEDICAL_UNIT', 0))
patient_type = st.sidebar.selectbox("PATIENT_TYPE (1 = Yes, 2 = No)", (1, 2), index=update_session_state('PATIENT_TYPE', 0))
pneumonia = st.sidebar.selectbox("Pneumonia (1 = Yes, 2 = No)", (1, 2), index=update_session_state('pneumonia', 0))
age = st.sidebar.slider("Age", 0, 100, update_session_state('age', 30))
diabetes = st.sidebar.selectbox("Diabetes (1 = Yes, 2 = No)", (1, 2), index=update_session_state('diabetes', 0))
hypertension = st.sidebar.selectbox("Hypertension (1 = Yes, 2 = No)", (1, 2), index=update_session_state('hypertension', 0))
renal_chronic = st.sidebar.selectbox("Renal Chronic Disease (1 = Yes, 2 = No)", (1, 2), index=update_session_state('renal_chronic', 0))
clasiffication_final = st.sidebar.slider("CLASIFFICATION_FINAL", 1, 7, update_session_state('CLASIFFICATION_FINAL', 3))


# Prepare input data for prediction
input_data = pd.DataFrame({
    'USMER': [usmer],
    'MEDICAL_UNIT': [medical_unit],
    'PATIENT_TYPE': [patient_type],
    'PNEUMONIA': [pneumonia],
    'AGE': [age],
    'DIABETES': [diabetes],
    'HIPERTENSION': [hypertension],
    'RENAL_CHRONIC': [renal_chronic],
    'CLASIFFICATION_FINAL': [clasiffication_final],
})

# Button to make prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.sidebar.success("Predicted Outcome: Death (1)")
    else:
        st.sidebar.success("Predicted Outcome: Survival (0)")
