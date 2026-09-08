from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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

# ----------------------- Configuration ----------------------- #
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_LENGTH = 4000

RATE_LIMIT = 10

SESSION_TTL_SECONDS = 1800
RATE_LIMIT_TTL_SECONDS = 600

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
CLASSES = ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"]
# ------------------------------------------------------------- #

server = redis.Redis(host="localhost", port=6379)
app = FastAPI()

load()

class MessageRequest(BaseModel):
    """
    Pydantic model representing an incoming chat message payload.

    Attributes:
        session_id (str): The unique identifier for the chat session.
        message (str): The message text sent by the user.
    """

    session_id: str
    message: str

@app.post("/sessions/initiate")
async def initiate(request: Request, file: UploadFile = File(...)):
    """
    Initiates a new chat session by processing an uploaded MRI image.

    Performs rate limiting, validates the file format and size, runs classification,
    generates an initial context-aware response using Gemini, and stores the session history in Redis.

    Args:
        request (Request): The incoming FastAPI request object used to determine client IP.
        file (UploadFile): The uploaded MRI image file (PNG, JPEG, or JPG).

    Returns:
        dict: A dictionary containing the unique session identifier (`id`), initial model `response`,
            classification `probabilities`, and `session_ttl`.

    Raises:
        HTTPException:
            - 400 if client IP cannot be identified, file type is unsupported, or image verification fails.
            - 413 if the uploaded file exceeds the maximum allowed size.
            - 429 if the rate limit is exceeded.
            - 502 if an error occurs during Gemini API invocation.
    """

    # -------------------------------------------------------------------- #
    # -- Missing: Check File For Any Viruses Using An Antivirus Scanner -- #
    # -------------------------------------------------------------------- #

    client = request.client if request else None
    host = client.host if client else None

    if not client or not host:
        raise HTTPException(status_code=400, detail="Unable to identify client ip address")

    rate_limit_key = f"Rate:{host}"

    current_rate = server.incr(rate_limit_key)
    if current_rate == 1:
        server.expire(rate_limit_key, RATE_LIMIT_TTL_SECONDS)

    if current_rate > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate Limit Exceeded", headers={"Retry-After": str(max(server.ttl(rate_limit_key), 1))})
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
        response = await gemini.invoke([user])
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

    return {"id": unique_id, "response": response, "probabilities": classification, "session_ttl": SESSION_TTL_SECONDS}

@app.post("/sessions/message")
async def message(request: Request, payload: MessageRequest):
    """
    Appends a user message to an active chat session and gets a response from Gemini.

    Performs rate limiting, validates session existence and message length, updates chat history,
    and queries the Gemini model.

    Args:
        request (Request): The incoming FastAPI request object used for rate limiting.
        payload (MessageRequest): The request body containing the session ID and user message.

    Returns:
        dict: A dictionary containing the model's text `response` and the updated `session_ttl`.

    Raises:
        HTTPException:
            - 400 if client IP is missing, session ID is invalid, or session is not found.
            - 413 if the message exceeds the maximum allowed length.
            - 429 if the rate limit is exceeded.
            - 502 if an error occurs during Gemini API invocation.
    """

    client = request.client if request else None
    host = client.host if client else None

    if not client or not host:
        raise HTTPException(status_code=400, detail="Unable to identify client ip address")

    rate_limit_key = f"Rate:{host}"

    current_rate = server.incr(rate_limit_key)
    if current_rate == 1:
        server.expire(rate_limit_key, RATE_LIMIT_TTL_SECONDS)
    if current_rate > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate Limit Exceeded", headers={"Retry-After": str(max(server.ttl(rate_limit_key), 1))})

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
        response = await gemini.invoke(history)
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

    return {"response": response, "session_ttl": server.ttl(session_key)}

@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """
    Global exception handler for unhandled server errors.

    Logs the request path and exception details, then returns a standardized 500 internal server error JSON response.

    Args:
        request: The incoming FastAPI request where the exception occurred.
        exc (Exception): The unhandled exception instance.

    Returns:
        JSONResponse: A 500 status response with an internal server error detail.
    """

    print(f"Request:{request.url.path}\nUnhandled Exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.get("/health")
async def health():
    """
    Performs a health check on the application.

    Returns:
        dict: A status dictionary indicating the server is operational, the active model name, and supported classes.
    """
    return {"status": "ok", "model": "resnet50", "classes": CLASSES}

app.mount("/", StaticFiles(directory=r"../static", html=True), name="static")

if __name__ == "__main__":
    ...