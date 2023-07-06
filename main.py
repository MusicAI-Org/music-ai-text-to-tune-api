from fastapi import FastAPI
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
from pydantic import BaseModel
import torch

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/generate_audio")
def generate_audio(request: QueryRequest):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = musicgen.MusicGen.get_pretrained('small', device=device)
    model.set_generation_params(duration=8)
    res = model.generate(request.query,
        progress=True)
    return display_audio(res, 32000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
