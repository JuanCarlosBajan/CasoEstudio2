from flask import Flask, request, jsonify
from google.cloud import firestore
from datetime import datetime
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('model.pkl')

# Inicializar Firestore usando la cuenta de servicio del entorno
db = firestore.Client()

# Referencias a las colecciones de Firestore
atletas_ref = db.collection('atletas')
actividades_ref = db.collection('actividades')

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Ruta no encontrada"}), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Solicitud incorrecta", "message": str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor", "details": str(error)}), 500

@app.route('/register_activity', methods=['POST'])
def register_activity():
    try:
        # Verificar la estructura de los datos de la solicitud
        data = request.get_json()
        required_fields = ["atleta_id", "tipo_actividad", "duracion_minutos", "distancia_km"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos en la solicitud", "required_fields": required_fields}), 400

        atleta_id = data['atleta_id']
        tipo_actividad = data['tipo_actividad']
        duracion_minutos = data['duracion_minutos']
        distancia_km = data['distancia_km']
        fecha = data.get('fecha', datetime.now().strftime('%Y-%m-%d'))

        # Verificar si el atleta existe
        atleta_doc = atletas_ref.document(atleta_id).get()
        if not atleta_doc.exists:
            # Crear el atleta si no existe
            atleta_data = {'id': atleta_id, 'esfuerzo': 0}
            atletas_ref.document(atleta_id).set(atleta_data)
            atleta_esfuerzo = 0
        else:
            # Obtener el score de esfuerzo actual del atleta
            atleta_esfuerzo = atleta_doc.to_dict().get('esfuerzo', 0)

        # Crear el registro de actividad
        activity_data = {
            'atleta_id': atleta_id,
            'tipo_actividad': tipo_actividad,
            'duracion_minutos': duracion_minutos,
            'distancia_km': distancia_km,
            'fecha': fecha
        }
        actividades_ref.document().set(activity_data)

        # Calcular el nuevo score de esfuerzo
        factor_distancia = 1.5
        factor_duracion = 0.5
        nuevo_score = distancia_km * factor_distancia + duracion_minutos * factor_duracion
        atleta_esfuerzo += nuevo_score

        # Actualizar el score de esfuerzo del atleta
        atletas_ref.document(atleta_id).update({'esfuerzo': atleta_esfuerzo})

        # Retornar el nuevo score de esfuerzo
        return jsonify({'nuevo_score': atleta_esfuerzo}), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Error procesando la solicitud", "details": str(e)}), 500

@app.route('/predict/<atleta_id>', methods=['GET'])
def predict_marathon(atleta_id):
    try:
        # Obtener todas las actividades del atleta desde Firestore
        actividades = actividades_ref.where('atleta_id', '==', atleta_id).stream()

        # Variables para calcular la media
        total_km = 0
        total_sp = 0
        count = 0

        # Calcular el promedio de `km4week` y `sp4week` a partir de las actividades
        for actividad in actividades:
            data = actividad.to_dict()
            total_km += data.get('distancia_km', 0)
            duracion_horas = data.get('duracion_minutos', 0) / 60
            if duracion_horas > 0:
                total_sp += data.get('distancia_km', 0) / duracion_horas
            count += 1

        # Si el atleta tiene actividades registradas, calcula la media; de lo contrario, usa 0
        km4week = total_km / count if count > 0 else 0
        sp4week = total_sp / count if count > 0 else 0

        # Crear un diccionario con los datos para el modelo
        input_data = pd.DataFrame([{'km4week': km4week, 'sp4week': sp4week}])

        # Realizar la predicción
        prediccion = modelo.predict(input_data)

        # Devolver el resultado de la predicción
        return jsonify({
            'MarathonTime': prediccion[0],
            'km4week': km4week,
            'sp4week': sp4week
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Error al realizar la predicción", "details": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de la solicitud y verificar estructura
        data = request.get_json(force=True)
        required_fields = ["km4week", "sp4week"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos en la solicitud", "required_fields": required_fields}), 400

        # Crear un DataFrame a partir de los datos
        input_data = pd.DataFrame([data])[['km4week', 'sp4week']]

        # Realizar la predicción
        prediccion = modelo.predict(input_data)

        # Devolver el resultado
        return jsonify({'MarathonTime': prediccion[0]})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Error al realizar la predicción", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
