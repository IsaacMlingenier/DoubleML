---
title: TB Double ML
emoji: 🚀
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: 'Causal Inference Simulated TB '
license: mit
---

Aplicación Streamlit: Double ML en tuberculosis pulmonar - versión 2
Instalación
pip install -r requirements_double_ml_tb_v2.txt
Ejecución
streamlit run app_double_ml_tb_v2.py
Datos
Cargar en la interfaz el archivo:

tb_pulmonar_sintetica_double_ml_100k.csv

Modelos incluidos
Double Machine Learning - IRM
Regresión logística ajustada con G-computation
IPTW
AIPW
Gráficos incluidos
Distribución del resultado final.
Resistencia farmacológica según resultado.
Éxito terapéutico según tratamiento oportuno.
Mortalidad según tratamiento oportuno.
Distribución de edad por tratamiento.
Demora diagnóstico-tratamiento por desenlace.
Edad y severidad inicial según éxito terapéutico.
Éxito terapéutico por VIH, diabetes, desnutrición y EPOC.
Tiempo hasta desenlace según mortalidad.
Comparación del ATE entre modelos.
Propensity score por grupo de tratamiento.
Distribución de pesos IPTW.
_____
