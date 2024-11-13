# pip install -r requirements.txt
# uvicorn main:app --host=0.0.0.0 --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import Model

app = FastAPI()

model = Model()

class FeaturesPackage(BaseModel):
    cap_diameter : int
    cap_shape : int
    gill_attachment : int
    gill_color : int
    stem_height : float
    stem_width : int
    stem_color : int
    season : float


@app.post('/predict/')
async def predict(package : FeaturesPackage):

    try:
        prediction = model.predict(
            package.cap_diameter,
            package.cap_shape,
            package.gill_attachment,
            package.gill_color,
            package.stem_height,
            package.stem_width,
            package.stem_color,
            package.season
        )
        return {'prediction': prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
