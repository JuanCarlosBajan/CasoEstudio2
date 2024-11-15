# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Leer el archivo CSV 'MarathonData.csv'
df = pd.read_csv('./Datos_Maraton/MarathonData.csv')

# 2. Preprocesamiento de Datos

# 2.1. Verificar si hay valores nulos
print("\n**Valores Nulos por Columna:**")
print(df.isnull().sum())

# 2.2. Manejar los valores faltantes en 'CrossTraining'
df['CrossTraining'].fillna('Sin CrossTraining', inplace=True)

# 2.3. Convertir 'CrossTraining' en variable binaria
df['CrossTraining_Bin'] = np.where(df['CrossTraining'] == 'Sin CrossTraining', 0, 1)

# 2.4. Seleccionar las columnas para el modelo
df_model = df[['km4week', 'sp4week', 'Wall21', 'MarathonTime', 'CrossTraining_Bin']]

# 2.5. Reemplazar valores problemáticos y convertir a numérico
# Reemplazar valores como ' -   ' por NaN
df_model.replace([' -   ', '-'], np.nan, inplace=True)

# Convertir las columnas a numéricas, manejando errores
df_model = df_model.apply(pd.to_numeric, errors='coerce')

# 2.6. Manejar los valores NaN resultantes
# Mostrar los valores nulos después de la conversión
print("\n**Valores Nulos Después de la Conversión a Numérico:**")
print(df_model.isnull().sum())

# Eliminar filas con valores NaN
df_model.dropna(inplace=True)

# 2.4. Seleccionar las columnas para el modelo
df_model = df[['km4week', 'sp4week', 'MarathonTime']]

# 4. Preparación de los Datos para el Modelo

# 4.1. Definir las variables independientes (X) y dependiente (y)
X = df_model.drop('MarathonTime', axis=1)
y = df_model['MarathonTime']

# 4.2. Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Construcción del Modelo de Regresión Lineal

# 5.1. Crear una instancia del modelo
modelo = LinearRegression()

# 5.2. Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# 5.3. Obtener los coeficientes del modelo
coeficientes = pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_})
print("\n**Coeficientes del Modelo:**")
print(coeficientes)

# 6. Evaluación del Modelo

# 6.1. Predecir con el conjunto de prueba
y_pred = modelo.predict(X_test)

# 6.2. Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 7. Guardar los Resultados en un Archivo CSV

# 7.1. Crear un DataFrame con los valores reales y las predicciones
resultados = pd.DataFrame({'Valor Real': y_test, 'Predicción': y_pred})

# 7.2. Guardar los resultados en un archivo CSV
resultados.to_csv('./Generacion_Modelo/resultados_prediccion.csv', index=False, encoding='utf-8-sig')
print("\nLos resultados de la predicción han sido guardados en 'resultados_prediccion.csv'.")

# 8. Guardar el Modelo Entrenado en un Archivo

import joblib

# Guardar el modelo en un archivo llamado 'modelo_entrenado.pkl'
joblib.dump(modelo, './Generacion_Modelo/modelo_entrenado.pkl')
print("\nEl modelo entrenado ha sido guardado en 'modelo_entrenado.pkl'.")

# Guardar el modelo en un archivo llamado 'modelo_entrenado.pkl'
joblib.dump(modelo, './contenedor/model.pkl')
print("\nEl modelo entrenado ha sido guardado en 'modelo_entrenado.pkl'.")
