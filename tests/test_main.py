# fix: 完善后端测试逻辑
from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

def test_heal_endpoint():
    response = client.get("/heal")
    assert response.status_code == 200

def test_predict_normal():
    payload = {
        "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
        "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06, 
        "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
        "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_predict_negative_boundary():
    payload = {
        "alcohol": -5.0, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
        "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06, 
        "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
        "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_extreme_boundary():
    payload = {
        "alcohol": 9999.0, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
        "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06, 
        "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
        "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "异常极端数值" in response.json()["detail"]

def test_predict_performance():
    payload = {
        "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6,
        "magnesium": 127.0, "total_phenols": 2.8, "flavanoids": 3.06, 
        "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29, "color_intensity": 5.64,
        "hue": 1.04, "od280_od315_of_diluted_wines": 3.92, "proline": 1065.0
    }
    start_time = time.time()
    response = client.post("/predict", json=payload)
    end_time = time.time()
    assert response.status_code == 200
    assert (end_time - start_time) < 0.5  # 性能测试：接口响应在500ms以内

def test_predict_missing_field():
    payload = {"alcohol": 14.23}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
