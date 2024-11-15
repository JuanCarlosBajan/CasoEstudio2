from flask import Flask, request, jsonify
import os
from datetime import datetime
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('model.pkl')

# Establecer el valor predeterminado de USE_FIRESTORE en False
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "false").lower() == "true"

# Si Firestore está habilitado, inicializarlo
if USE_FIRESTORE:
    from google.cloud import firestore
    db = firestore.Client()
    atletas_ref = db.collection('atletas')
    actividades_ref = db.collection('actividades')
else:
    # Si Firestore no está disponible, usa datos simulados
    db, atletas_ref, actividades_ref = None, None, None

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
        data = request.get_json()
        required_fields = ["atleta_id", "tipo_actividad", "duracion_minutos", "distancia_km"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos en la solicitud", "required_fields": required_fields}), 400

        atleta_id = data['atleta_id']
        tipo_actividad = data['tipo_actividad']
        duracion_minutos = data['duracion_minutos']
        distancia_km = data['distancia_km']
        fecha = data.get('fecha', datetime.now().strftime('%Y-%m-%d'))

        if USE_FIRESTORE:
            # Interactuar con Firestore
            atleta_doc = atletas_ref.document(atleta_id).get()
            if not atleta_doc.exists:
                atleta_data = {'id': atleta_id, 'esfuerzo': 0}
                atletas_ref.document(atleta_id).set(atleta_data)
                atleta_esfuerzo = 0
            else:
                atleta_esfuerzo = atleta_doc.to_dict().get('esfuerzo', 0)

            activity_data = {
                'atleta_id': atleta_id,
                'tipo_actividad': tipo_actividad,
                'duracion_minutos': duracion_minutos,
                'distancia_km': distancia_km,
                'fecha': fecha
            }
            actividades_ref.document().set(activity_data)

            factor_distancia = 1.5
            factor_duracion = 0.5
            nuevo_score = distancia_km * factor_distancia + duracion_minutos * factor_duracion
            atleta_esfuerzo += nuevo_score

            atletas_ref.document(atleta_id).update({'esfuerzo': atleta_esfuerzo})
            return jsonify({'nuevo_score': atleta_esfuerzo}), 200

        else:
            # Respuesta simulada cuando Firestore no está disponible
            return jsonify({'message': 'Actividad registrada (simulada)', 'nuevo_score': 42}), 200

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Error procesando la solicitud", "details": str(e)}), 500

@app.route('/predict/<atleta_id>', methods=['GET'])
def predict_marathon(atleta_id):
    try:
        if USE_FIRESTORE:
            actividades = actividades_ref.where('atleta_id', '==', atleta_id).stream()
            total_km, total_sp, count = 0, 0, 0
            for actividad in actividades:
                data = actividad.to_dict()
                total_km += data.get('distancia_km', 0)
                duracion_horas = data.get('duracion_minutos', 0) / 60
                if duracion_horas > 0:
                    total_sp += data.get('distancia_km', 0) / duracion_horas
                count += 1
            km4week = total_km / count if count > 0 else 0
            sp4week = total_sp / count if count > 0 else 0
        else:
            # Valores simulados en caso de que Firestore no esté disponible
            km4week, sp4week = 10, 8

        input_data = pd.DataFrame([{'km4week': km4week, 'sp4week': sp4week}])
        prediccion = modelo.predict(input_data)

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
        data = request.get_json(force=True)
        required_fields = ["km4week", "sp4week"]
        if not data or not all(field in data for field in required_fields):
            return jsonify({"error": "Faltan campos en la solicitud", "required_fields": required_fields}), 400

        input_data = pd.DataFrame([data])[['km4week', 'sp4week']]
        prediccion = modelo.predict(input_data)

        return jsonify({'MarathonTime': prediccion[0]})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Error al realizar la predicción", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
