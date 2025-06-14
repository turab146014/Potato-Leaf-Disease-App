
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
API = "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"
from langchain_groq import ChatGroq

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("F:/Potato_leaf_diesase_app/Code/Potato Leaf Disease-app/models/potatoes.h5", compile=False)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

last_predicted_class = None


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_predicted_class

    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    last_predicted_class = predicted_class

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }



llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key= API,
)

@app.get("/disease-info")
async def get_disease_info():
    global last_predicted_class

    if last_predicted_class is None:
        return {"error": "No prediction available yet. Please upload an image first."}

    message = f"""Explain the following potato disease in detail: {last_predicted_class}.
Include the following:
- What is {last_predicted_class}?
- Symptoms of {last_predicted_class}
- Ways to prevent {last_predicted_class}
- Effective treatment options for {last_predicted_class}"""

    response = llm.invoke(message)
    return {"disease": last_predicted_class, "details": response.content}



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)



