from typing import List

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
from prompt import initialization_prompt

app = FastAPI()

server = redis.Redis(host="localhost", port=6379, decode_responses=True)

@app.post("/process")
async def initiate(file: UploadFile = File(...)):

    bytes_data = await file.read()

    try:

        verification = Image.open(BytesIO(bytes_data))
        verification.verify()

        # ---------------------------------------------------------------------------------------------- #
        # -- Missing: Verify Image Is An Image Of An MRI Scan And That The Image Is Clear And Visible -- #
        # ---------------------------------------------------------------------------------------------- #

        image = Image.open(BytesIO(bytes_data)).convert("RGB")

        load()

        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)

        classification = invoke(image)

        unique_id = str(uuid.uuid4())
        key = f"Conversation:{unique_id}"

        prompt = initialization_prompt(classification)
        response = gemini.invoke(prompt)

        server.rpush(key,prompt, response)
        server.expire(key, 1800)

        return {"id": unique_id, "response": response}

    except Exception:
                raise HTTPException(status_code=400, detail="File Failed To Verify As An Image")

@app.post("/chat")
async def chat(session_id: str, user: str):

    key = f"Conversation:{session_id}"

    server.rpush(key, f"User: {user}")

    history = server.lrange(key, 0, -1)
    response = gemini.invoke(history) #type: ignore

    server.rpush(key, f"AI: {response}")
    server.expire(key, 1800)

    return {"response": response}