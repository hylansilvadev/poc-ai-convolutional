from fastapi.testclient import TestClient
from src.app.main import app


def test_validade_dog_image():
    url: str = '/predict/'

    client = TestClient(app)

    image_path: str = 'tests/resources/dog.9887.jpg'

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    request = client.post(url, files={'files': image_bytes})

    response = request.json()

    assert response['result'] == 'Cachorro'


def test_validade_cat_image():
    url: str = '/predict/'

    client = TestClient(app)

    image_path: str = 'tests/resources/cat.2504.jpg'

    with open(image_path, "rb") as file:
        image_bytes = file.read()

    request = client.post(url, files={'files': image_bytes})

    response = request.json()

    assert response.result == 'Gato'
