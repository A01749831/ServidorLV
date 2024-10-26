from flask import Flask, render_template,jsonify,request
from pymongo import MongoClient
import os

app = Flask(__name__)
os.makedirs("Images", exist_ok=True)
app.config['Images'] = "Images"

# Conectar a MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Cambia la URI según tu configuración
db = client["Liverpool"]
coleccion1 = db['Productos']
coleccion2 = db['ProductosData']

@app.route('/', methods=['GET'])
def obtener_datos():
    datos = coleccion2.find()
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

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings_dict = list(coleccion1.find())

def preprocesar_imagen(img_path, size=(224, 224)):
    """Redimensiona la imagen a un tamaño uniforme."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        return img
    except Exception as e:
        # print(f"Error procesando {img_path}: {e}")
        return None

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def obtener_embedding(imagen):
    """Genera el embedding de una imagen."""
    inputs = processor(images=imagen, return_tensors="pt")
    outputs = model.get_image_features(**inputs)

    return outputs / outputs.norm(dim=-1, keepdim=True)

def calcular_similares(image):
    preprocesada = preprocesar_imagen(image)
    embedding = obtener_embedding(preprocesada)
    
    embedding = embedding.detach().numpy()
    similares = []
    skus_agregados = set()  # Conjunto para rastrear los SKUs ya agregados

    for item in embeddings_dict:
        sku = item.get('sku')
        index = 0
        
        # Si el SKU ya está en el conjunto, saltamos a la siguiente iteración
        if sku in skus_agregados:
            continue

        for emb in item['embedding']:
            index += 1
            tensor = torch.tensor(emb)
            similitud = cosine_similarity(embedding, tensor.detach().numpy())

            if similitud[0][0] > 0.78:
                x = coleccion2.find_one({'SKU': sku})
                similares.append({'SKU': sku, 'url': x[f'Imagen {index}'], 'Nombre': x['Name'], 'Pt': float(similitud[0][0])})
                skus_agregados.add(sku)  # Agrega el SKU al conjunto para evitar duplicados
                break  # Sale del loop interno una vez agregado el SKU

    return similares


@app.route("/search",methods=['POST'])
def comparar():
    if 'image' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Guardar la imagen
    file_path = os.path.join(app.config['Images'], file.filename)
    file.save(file_path)
    lista = calcular_similares(file_path)
    os.remove(file_path)
    return jsonify(lista)

if __name__ == '__main__':
    app.run(debug=True,host='10.48.73.189')
