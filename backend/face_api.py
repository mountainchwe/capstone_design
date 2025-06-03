from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition
import chromadb

app = FastAPI()

# ✅ CORS Middleware (still needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use "*" temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Manual OPTIONS fallback
@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str):
    return JSONResponse(
        status_code=204,
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )

# ────────────────────────────────
# ✅ ChromaDB
# ────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="face_vector")

class RegisterFaceRequest(BaseModel):
    student_id: str
    name: str
    image_base64: str

@app.post("/register-face")
async def register_face(data: RegisterFaceRequest):
    try:
        image_data = base64.b64decode(data.image_base64.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        if len(face_encodings) == 1:
            vector = face_encodings[0].tolist()
            collection.add(
                embeddings=[vector],
                ids=[data.student_id],
                metadatas=[{"name": data.name}]
            )
            return JSONResponse(
                content={"status": "success", "message": f"Registered {data.name}"},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        elif len(face_encodings) > 1:
            return JSONResponse(
                content={"status": "error", "message": "Multiple faces detected."},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        else:
            return JSONResponse(
                content={"status": "error", "message": "No face detected."},
                headers={"Access-Control-Allow-Origin": "*"}
            )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
