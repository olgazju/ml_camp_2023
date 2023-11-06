from fastapi import FastAPI, HTTPException, Request
import pickle
import json

app = FastAPI()

# how to run the server:
# uvicorn server:app --reload

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
      'number_cast_members', 
      "numerical_ROI_category", 
      'numerical_rating_category', 
      'numerical_award_category'}

    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")

    X = dict_vectorizer.transform(data)
    X = standart_scaler.transform(X)

    # Make a prediction
    prediction = model.predict(X)

    # Return the prediction
    return {"prediction": prediction}
