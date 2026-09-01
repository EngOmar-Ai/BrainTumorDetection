from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO
from invoke import invoke
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from train import load

app = FastAPI()

@app.post("/process")
async def create_user(file: UploadFile = File(...)):

    bytes_data = await file.read()

    try:

        image = Image.open(BytesIO(bytes_data)).convert("RGB")
        image.verify()

        load()

        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0)

        classification = invoke(image)

        return classification

    except Exception:
                raise HTTPException(status_code=400, detail="File Failed To Verify As An Image")

