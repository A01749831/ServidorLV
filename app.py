from flask import Flask, render_template,jsonify
from pymongo import MongoClient

app = Flask(__name__)

# Conectar a MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Cambia la URI según tu configuración
db = client["productosLV"]
coleccion = db['productos']

@app.route('/', methods=['GET'])
def obtener_datos():
    datos = coleccion.find()
    resultado = []
    for dato in datos:
        # Convertir ObjectId a cadena
        dato['_id'] = str(dato['_id'])
        data = {
            'SKU':dato['SKU'],
            'Nombre':dato['Name'],
            'url':dato['Imagen 1']
        }
        resultado.append(data)
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True,host='10.48.102.254')
