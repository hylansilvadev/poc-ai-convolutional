from fastapi import FastAPI, File, UploadFile, status 
from keras._tf_keras.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI(
    title="CNN Model with FastAPI",
    version="0.0.1",
    docs_url="/",
    description="A Prove of concept, in test for Convolutional Neural Networks Model, with FastAPI"
)

# Carregar o modelo
model_path = 'src/resources/model.h5'
loaded_model = load_model(model_path)


# Função para processar a imagem e obter a previsão
def predict_image_class(image_data):
    # Pré-processamento da imagem
    img = Image.open(BytesIO(image_data))
    img = img.resize((224, 224))  # Tamanho esperado pelo modelo
    img = np.array(img) / 255.0  # Normalização

    # Realizar a previsão
    prediction = loaded_model.predict(np.expand_dims(img, axis=0))

    # Determinar a classe predita
    predicted_class = 'Cachorro' if prediction[0][1] > 0.5 else 'Gato'

    return {"result":predicted_class}


# Rota para enviar a imagem e obter a previsão
@app.post(
    "/predict/",
    tags=["CNN Validator"],
    description="Send image file with cat or dogs, and the Model classifier the image",
    status_code=status.HTTP_200_OK
    )
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    prediction = predict_image_class(contents)
    return prediction
