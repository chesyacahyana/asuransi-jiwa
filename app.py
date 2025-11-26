
import streamlit as st
import pandas as pd
import pickle # Changed from joblib to pickle

st.markdown("""
    <h2 style='text-align: center; color: #4CAF50;'>Hasil Prediksi Biaya Asuransi</h2>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main {
        background-color: #fafafa;
    }
    .sidebar .sidebar-content {
        background-color: #eef2f3;
    }
</style>
""", unsafe_allow_html=True)



# Load the trained model
with open('GBR_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.title('Prediksi Biaya Asuransi Kesehatan') # Updated title
st.write('Aplikasi untuk memprediksi biaya asuransi kesehatan berdasarkan parameter yang diberikan.') # Updated description

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    age = st.sidebar.slider('Usia', 18, 64, 30)
    sex = st.sidebar.selectbox('Jenis Kelamin', ['male', 'female'])
    bmi = st.sidebar.slider('BMI (Body Mass Index)', 16.0, 53.0, 25.0)
    children = st.sidebar.slider('Jumlah Anak', 0, 5, 1)
    smoker = st.sidebar.selectbox('Perokok', ['yes', 'no'])
    region = st.sidebar.selectbox('Wilayah', ['southwest', 'southeast', 'northwest', 'northeast'])

    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Define the exact columns and their dtypes expected by the model during training
# This list ensures correct order and includes all dummy variables used during training
training_columns_and_dtypes = {
    'age': 'int64',
    'bmi': 'float64',
    'children': 'int64',
    'sex_male': 'bool',
    'smoker_yes': 'bool',
    'region_northwest': 'bool',
    'region_southeast': 'bool',
    'region_southwest': 'bool'
}

# Create an empty DataFrame with the correct columns and dtypes
final_input_df = pd.DataFrame(columns=training_columns_and_dtypes.keys())
for col, dtype in training_columns_and_dtypes.items():
    final_input_df[col] = final_input_df[col].astype(dtype)

# Add a single row of data, initially all zeros/False
final_input_df.loc[0] = 0
for col, dtype in training_columns_and_dtypes.items():
    if dtype == 'bool':
        final_input_df.loc[0, col] = False

# Populate numerical features
final_input_df.loc[0, 'age'] = df_input['age'][0]
final_input_df.loc[0, 'bmi'] = df_input['bmi'][0]
final_input_df.loc[0, 'children'] = df_input['children'][0]

# Populate one-hot encoded categorical features
if df_input['sex'][0] == 'male':
    final_input_df.loc[0, 'sex_male'] = True

if df_input['smoker'][0] == 'yes':
    final_input_df.loc[0, 'smoker_yes'] = True

if df_input['region'][0] == 'northwest':
    final_input_df.loc[0, 'region_northwest'] = True
elif df_input['region'][0] == 'southeast':
    final_input_df.loc[0, 'region_southeast'] = True
elif df_input['region'][0] == 'southwest':
    final_input_df.loc[0, 'region_southwest'] = True

# Make prediction
if st.sidebar.button('Prediksi Biaya Asuransi'): # Updated button text
    try:
        prediction = model.predict(final_input_df)
        st.subheader('Hasil Prediksi Biaya Asuransi:') # Updated output header
        st.write(f"Biaya Asuransi Diprediksi: $ {prediction[0]:,.2f} per tahun") # Updated output message and currency
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.exception(e)

