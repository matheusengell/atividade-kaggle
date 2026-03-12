import streamlit as st
import joblib
import pandas as pd

model = joblib.load('modelo_fifa.pkl') 

st.set_page_config(page_title="FIFA Predictor", page_icon="⚽")

st.title("⚽ FIFA Player Rating Predictor")
st.markdown("Ajuste os atributos para prever o **Overall** do jogador.")

st.sidebar.header("Atributos do Jogador")
pace = st.sidebar.slider("Pace (Velocidade)", 30, 99, 75)
shooting = st.sidebar.slider("Shooting (Chute)", 30, 99, 70)
passing = st.sidebar.slider("Passing (Passe)", 30, 99, 72)
dribbling = st.sidebar.slider("Dribbling (Drible)", 30, 99, 75)
physic = st.sidebar.slider("Physic (Físico)", 30, 99, 70)

if st.button("Calcular Rating"):
    input_df = pd.DataFrame([[pace, shooting, passing, dribbling, physic]], 
                            columns=['pace', 'shooting', 'passing', 'dribbling', 'physic'])
    
    prediction = model.predict(input_df)[0]
    
    st.markdown("---")
    st.subheader("Resultado Estimado:")
    st.metric(label="OVERALL RATING", value=int(prediction))
    
    if prediction >= 85:
        st.warning("🌟 Este é um jogador de Classe Mundial!")
    elif prediction >= 75:
        st.info("🏃 Jogador sólido para times principais.")
    else:
        st.success("📈 Jogador com potencial de desenvolvimento.")