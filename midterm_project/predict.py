from fastapi import FastAPI, HTTPException, Request
import pickle
import json
from fastapi.encoders import jsonable_encoder
import numpy as np

app = FastAPI()

# how to run the server:
# uvicorn predict:app --reload

# Load the pre-trained model, DictVectorizer and StandardScaler
with open('models_binary/catboost_classifier_model.pkl', 'rb') as f_model:
    model = pickle.load(f_model)
with open('models_binary/dict_vectorizer.pkl', 'rb') as f_bin:
    dict_vectorizer = pickle.load(f_bin)
with open('models_binary/standard_scaler.pkl', 'rb') as f_bin:
    standart_scaler = pickle.load(f_bin)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    print("data", data)

    # Ensure the required fields are present
    required_fields = {'runtime',
      'collection',  
      'is_english',
      'log_adjusted_budget',
      'winter', 
      'animation', 
      'drama', 
      'moderate performing', 
      'others', 
      'num_spoken_languages',
      'num_production_companies', 
      'num_production_countries', 
      'is_foreign',
      'director_popularity', 
      'writer_popularity', 
      'producer_popularity', 
      'average_crew_popularity',
      'number_crew_members', 
      'average_cast_popularity', 
      'number_cast_members'}

    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")

    print("dict_vectorizer")
    X = dict_vectorizer.transform(data)
    print("standart_scaler")
    X = standart_scaler.transform(X)

    # Make a prediction
    print("prediction")
    prediction = model.predict(X)

    prediction_list = prediction.tolist()
  
    return jsonable_encoder({"prediction": prediction_list})
    