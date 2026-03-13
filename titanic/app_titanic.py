import streamlit as st
import pandas as pd
import joblib

modelo = joblib.load('modelo_titanic.pkl')

st.title("🚢 Previsão de Sobrevivência - Titanic")

pclass = st.selectbox("Classe", [1, 2, 3])
sex = st.radio("Sexo", ["Masculino", "Feminino"])
age = st.slider("Idade", 0, 100, 25)
sibsp = st.number_input("Irmãos/Cônjuges a bordo", 0, 10, 0)
fare = st.number_input("Preço da Passagem", 0.0, 500.0, 30.0)

if st.button("Descobrir se sobrevivi"):
    sex_bin = 1 if sex == "Feminino" else 0
    dados = pd.DataFrame([[pclass, sex_bin, age, sibsp, fare]], 
                         columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])
    
    resultado = modelo.predict(dados)[0]
    if resultado == 1:
        st.success("Você sobreviveria! 🎉")
    else:
        st.error("Infelizmente, você não sobreviveria. ❄️")