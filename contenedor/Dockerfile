# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY ./contenedor/requirements.txt requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de la aplicación
COPY ./contenedor/app.py app.py

# Copia el modelo al contenedor
COPY ./contenedor/model.pkl model.pkl

# Expone el puerto en el que corre la API
EXPOSE 8080

# Define el comando para ejecutar la aplicación
CMD ["python", "app.py"]
