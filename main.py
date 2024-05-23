import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import inference_single_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceResponse(BaseModel):
    predicted_class_name: str
    predicted_probability: float
    plane_detail: str

@app.post("/upload", response_model=InferenceResponse)
async def upload_file(file: UploadFile = File(...)):
    file_location = f"./uploads/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    inference_result = inference_single_image(file_location)
    print (inference_result)
    return inference_result
