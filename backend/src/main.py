from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from io import BytesIO

from torchvision.transforms import InterpolationMode
from torchvision import transforms

from prompt import initiation_prompt
from train import load
from invoke import invoke
import gemini

import redis
import json
import uuid

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_LENGTH = 4000
SESSION_TTL_SECONDS = 1800
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

server = redis.Redis(host="localhost", port=6379)
app = FastAPI()

load()

class MessageRequest(BaseModel):
    session_id: str
    message: str

@app.post("/sessions/initiate")
async def initiate(file: UploadFile = File(...)):

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type; upload a PNG,JPEG or JPG image")

    bytes_data = await file.read()

    if len(bytes_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeded maximum allowed size of 8 megabytes")

    try:
        verification = Image.open(BytesIO(bytes_data))
        verification.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File failed to verify as an image")

    # ---------------------------------------------------------------------------------------------- #
    # -- Missing: Verify Image Is An Image Of An MRI Scan And That The Image Is Clear And Visible -- #
    # ---------------------------------------------------------------------------------------------- #

    image = Image.open(BytesIO(bytes_data)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)

    classification = invoke(image)

    unique_id = str(uuid.uuid4())
    session_key = f"Conversation:{unique_id}"

    prompt = initiation_prompt(classification)
    user = {
        "role": "user",
        "parts": [
            {"text": prompt}
        ]
    }

    try:
        response = gemini.invoke([user])
    except gemini.GeminiInvocationError as error:
        raise HTTPException(status_code=502, detail=str(error))

    assistant = {
        "role": "model",
        "parts": [
            {"text": response}
        ]
    }

    pipe = server.pipeline()

    pipe.rpush(session_key, json.dumps(user))
    pipe.rpush(session_key, json.dumps(assistant))
    pipe.expire(session_key, SESSION_TTL_SECONDS)

    pipe.execute()

    return {"id": unique_id, "response": response}

@app.post("/sessions/message")
async def message(payload: MessageRequest):

    try:
        session_key = f"Conversation:{uuid.UUID(payload.session_id)}"
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid Session Id")

    if not server.exists(session_key):
        raise HTTPException(status_code=400, detail="Session Not Found")

    if len(payload.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=413, detail="Message Exceeded Maximum Allowed Length")

    user = {
        "role": "user",
        "parts": [
            {"text": payload.message}
        ]
    }


    server.rpush(session_key, json.dumps(user))

    history = server.lrange(session_key, 0, -1)
    history = [json.loads(turn) for turn in history]

    try:
        response = gemini.invoke(history)
    except gemini.GeminiInvocationError as error:
        raise HTTPException(status_code=502, detail=str(error))

    assistant = {
        "role": "model",
        "parts": [
            {"text": response}
        ]
    }

    pipe = server.pipeline()

    pipe.rpush(session_key, json.dumps(assistant))
    pipe.expire(session_key, SESSION_TTL_SECONDS)

    pipe.execute()

    return {"response": response}

@app.exception_handler(Exception)
async def unhandled_exception_handler():
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

if __name__ == "__main__":
    ...