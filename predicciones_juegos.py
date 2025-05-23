import streamlit as st
import pandas as pd
import pickle
import os

# Carga el modelo pre-entrenado
try:
    with open('modelo-reg-tree-knn-nn.pkl', 'rb') as file:
        model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("El archivo del modelo 'modelo-reg-tree-knn-nn.pkl' no se encontró. Asegúrate de que esté en la misma carpeta que este script.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al cargar el modelo: {e}")
    st.stop()

st.title("Predicción de Gasto en Videojuegos")

# Campos de entrada para el usuario
edad = st.number_input("Edad:", min_value=14, max_value=120, step=1)
genero = st.selectbox("Género:", ["Hombre", "Mujer", "Otro"])
tipo_juego = st.selectbox(
    "Tipo de Videojuego:",
    ["Mass Effect", "Sim City", "Dead Space", "Battlefield", "FIFA", "F1", "KOA: Reckoning", "Crysis"]
)
plataforma = st.selectbox(
    "Plataforma:",
    ["PC", "Xbox", "Play Station", "Otros"]
)
consumidor_habitual = st.radio("¿Eres consumidor habitual?", ["Si", "No"])

# Botón para realizar la predicción
if st.button("Realizar Predicción"):
    # Crea un DataFrame con los datos de entrada del usuario
    input_data = pd.DataFrame({
        'Edad': [edad],
        'videojuego': [tipo_juego],
        'Plataforma': [plataforma],
        'Sexo': [genero],
        'Consumidor_habitual': [consumidor_habitual]
    })

    # Muestra los datos ingresados en una tabla
    st.subheader("Datos Ingresados:")
    st.table(input_data)

    # **Preprocesamiento de datos para el modelo**
    data_preparada = input_data.copy()
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego'], prefix='videojuego')
    data_preparada = pd.get_dummies(data_preparada, columns=['Plataforma'], prefix='Plataforma')
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo'], prefix='Sexo')
    data_preparada = pd.get_dummies(data_preparada, columns=['Consumidor_habitual'], prefix='Consumidor_habitual')
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    data_preparada[['Edad']] = min_max_scaler.transform(data_preparada[['Edad']])

    try:
        # Realiza la predicción con los tres modelos
        prediccion_tree = model_Tree.predict(data_preparada.values)[0]
        prediccion_knn = model_Knn.predict(data_preparada.values)[0]
        prediccion_nn = model_NN.predict(data_preparada.values)[0]

        # Calcula el promedio de las predicciones
        prediccion_promedio = (prediccion_tree + prediccion_knn + prediccion_nn) / 3

        # Muestra el resultado de la predicción promediada
        st.subheader("Predicción de Gasto")
        st.write(f"Se estima que gastarás: ${prediccion_promedio:.2f} en este escenario.")
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")