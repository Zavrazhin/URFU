import streamlit as st
from transformers import pipeline

pipe = pipeline("text-classification", model="cointegrated/rubert-tiny2-cedr-emotion-detection")
st.title('Text-classification')
st.subheader("Данное Web-приложение опредялет эмоцианальную окраску введеного текста")
text = st.text_input('Введите текст', 'Я люблю программную инженерию')
result = st.button('Magic')

if result:
    st.markdown("Работаем....")
    st.markdown("...еще немного")
    st.write('**Результаты:**')
    st.write(pipe(text, top_k=None))
