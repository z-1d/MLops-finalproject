from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import WineNet

app = FastAPI()

model = WineNet()
model_path = 'ml/registry/model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model_loaded = True
else:
    model_loaded = False

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.get("/heal")
def heal():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: WineFeatures):
    if features.alcohol < 0 or features.magnesium < 0:
        raise HTTPException(status_code=400, detail="特征值不能为负数")
    if features.alcohol > 100:
        raise HTTPException(status_code=400, detail="异常极端数值：酒精浓度过高")
        
    if not model_loaded:
        return {"prediction": "Model not loaded, mock response"}
        
    input_tensor = torch.tensor([[
        features.alcohol, features.malic_acid, features.ash, features.alcalinity_of_ash,
        features.magnesium, features.total_phenols, features.flavanoids, features.nonflavanoid_phenols,
        features.proanthocyanins, features.color_intensity, features.hue, 
        features.od280_od315_of_diluted_wines, features.proline
    ]], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        
    return {"prediction": int(predicted.item())}
