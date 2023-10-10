from fastapi import FastAPI, HTTPException, Request
import pickle
import json

app = FastAPI()

# how to run the server:
# uvicorn server:app --reload

# Load the pre-trained model and DictVectorizer
with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)
with open('dv.bin', 'rb') as f_bin:
    dict_vectorizer = pickle.load(f_bin)

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    # Ensure the required fields are present
    required_fields = {"job", "duration", "poutcome"}
    if not all(field in data for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")

    X = dict_vectorizer.transform(data)

    # Make a prediction
    prediction = model.predict_proba(X)[0, 1]

    # Return the prediction
    return {"prediction": prediction}
