import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from invoke import invoke
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from train import load
import redis
import uuid
import gemini
from prompt import initiation_prompt
from pydantic import BaseModel

load()

server = redis.Redis(host="localhost", port=6379, decode_responses=True)
app = FastAPI()

class MessageRequest(BaseModel):
    session_id: str
    message: str

@app.post("/sessions/initiate")
async def initiate(file: UploadFile = File(...)):

    bytes_data = await file.read()

    try:
        verification = Image.open(BytesIO(bytes_data))
        verification.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="File Failed To Verify As An Image")

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

    server.rpush(session_key, json.dumps(user))

    response = gemini.invoke([user])
    assistant = {
        "role": "model",
        "parts": [
            {"text": response}
        ]
    }

    server.rpush(session_key, json.dumps(assistant))

    server.expire(session_key, 1800)

    return {"id": unique_id, "response": response}

@app.post("/sessions/message")
async def message(payload: MessageRequest):

    session_key = f"Conversation:{payload.session_id}"

    if server.exists(session_key):

        user = {
            "role": "user",
            "parts": [
                {"text": payload.message}
            ]
        }

        server.rpush(session_key, json.dumps(user))

        history = server.lrange(session_key, 0, -1)
        history = [json.loads(turn) for turn in history]

        response = gemini.invoke(history)

        assistant = {
            "role": "model",
            "parts": [
                {"text": response}
            ]
        }

        server.rpush(session_key, json.dumps(assistant))

        server.expire(session_key, 1800)

        return {"session_active": True,"response": response}

    else:
        return {"session_active": False, "response": "Session Expired"}