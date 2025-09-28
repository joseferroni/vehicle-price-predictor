# vehicle-price-predictor
Modelo para estimar precio de vehículos
# Proyecto: Regresión de Automóviles

Este repositorio acompaña al notebook `regresionautos_2`  y documenta el proceso completo de análisis de datos y modelado por regresión aplicado a un conjunto de datos de automóviles, fuente dato : https://www.kaggle.com/datasets/metawave/vehicle-price-prediction. El objetivo es construir un modelo predictivo capaz de estimar el precio de un vehículo a partir de sus características (año, kilometraje, marca, modelo, potencia, etc.) y ponerlo a disposición mediante una interfaz web con Streamlit.

## Contenido del Notebook

El notebook realiza las siguientes tareas:

- **Carga de datos**: Lectura del dataset con información de vehículos.
- **Exploración y limpieza**: Estadísticas, visualizaciones, tratamiento de valores nulos y outliers.
- **Ingeniería de características**: Transformaciones y codificaciones para mejorar el modelo.
- **División de datos**: Separación en conjuntos de entrenamiento y prueba.
- **Entrenamiento de modelos**: Aplicación de algoritmos de regresión.
- **Evaluación**: Métricas como RMSE, MAE, R² y visualización de resultados.


## Aplicación con Streamlit

Los últimos cuatro archivos generados por el notebook están diseñados para ser usados en el script `concesionaria.py`, que implementa una interfaz web con Streamlit. Este script:

- Carga el modelo entrenado y los preprocesadores.
- Recibe entradas del usuario (marca, año, km, etc.).
- Devuelve una predicción del precio del vehículo.

Los archivos generados, neceasrios para la interfaz son : brand_marca.csv , vars_cat.csv, X_train_cols.csv, modelo_entrenado.pkl

Para ejecutar la app:

```bash
streamlit run concesionaria.py
import requeriment.txt
