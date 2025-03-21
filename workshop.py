import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# Preprocess and split data
data.fillna(method='ffill', inplace=True)
train, test = train_test_split(data, test_size=0.2, random_state=42)
X_train = train.drop("median_house_value", axis=1)
y_train = train["median_house_value"]
X_test = test.drop("median_house_value", axis=1)
y_test = test["median_house_value"]

# Model training

# 假设encoder是之前训练时用来独热编码的编码器实例
# 这里的input_data应该是一个列表，包含除了独热编码特征之外的所有其他特征

def predict_price(input_data):
    # 假设input_data中的分类特征在最后一列，我们需要对其进行独热编码
    categorical_feature = np.array(input_data[-1]).reshape(-1, 1)
    encoded_feature = encoder.transform(categorical_feature).toarray()

    # 重新组合输入数据：数值特征加上独热编码后的分类特征
    numerical_features = input_data[:-1]
    final_features = np.concatenate([numerical_features, encoded_feature.flatten()])

    # 预测并返回结果
    prediction = model.predict([final_features])
    return prediction[0]

# Streamlit UI
st.title('California Housing Price Prediction')
st.write("This application predicts the median house values in California.")

# User input features
total_rooms = st.number_input('Total rooms', min_value=1, value=5)
total_bedrooms = st.number_input('Total bedrooms', min_value=1, value=2)
population = st.number_input('Population', min_value=1, value=3)
households = st.number_input('Households', min_value=1, value=1)
median_income = st.number_input('Median income', min_value=0.1, step=0.1, value=2.0)

input_data = [total_rooms, total_bedrooms, population, households, median_income]

# Predict button
if st.button('Predict Price'):
    result = predict_price(input_data)
    st.success(f'The predicted price of the house is ${result:.2f}')

# Run this app with: streamlit run app.py
