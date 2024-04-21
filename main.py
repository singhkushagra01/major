from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Sample data for endpoint 1
data = '''{
  "FedAvg": {
    "Round 1": {
      "accuracy": 0.87,
      "precision": 0.85,
      "recall": 0.88
    },
    "Round 2": {
      "accuracy": 0.89,
      "precision": 0.82,
      "recall": 0.90
    },
    "Round 3": {
      "accuracy": 0.91,
      "precision": 0.88,
      "recall": 0.84
    }
  },
  "FedMedian": {
    "Round 1": {
      "accuracy": 0.86,
      "precision": 0.90,
      "recall": 0.82
    },
    "Round 2": {
      "accuracy": 0.88,
      "precision": 0.87,
      "recall": 0.85
    },
    "Round 3": {
      "accuracy": 0.83,
      "precision": 0.91,
      "recall": 0.89
    }
  }
}'''

@app.get("/items")
async def get_items(model_name: str):
    """Get data for a specific model"""
    try:
        parsed_data = json.loads(data)
        return parsed_data[model_name]
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/upload_image")
async def upload_image(image: UploadFile = File(...)):
    """Securely receive and store an image (implementation for storage omitted)"""
    content = await image.read()

    # Implement secure storage for the image content here (e.g., database)

    return JSONResponse({"message": "Image uploaded successfully"})

@app.get("/text")
async def get_text():
    """Sends a predefined text message"""
    return {"message": "This is a sample text message"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
