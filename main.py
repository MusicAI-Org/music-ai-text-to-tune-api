from fastapi import FastAPI
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch

app = FastAPI()

@app.get("/generate_audio")
def generate_audio():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = musicgen.MusicGen.get_pretrained('medium', device=device)
    model.set_generation_params(duration=8)
    res = model.generate([
        'crazy EDM, heavy bang',
        'classic reggae track with an electronic guitar solo',
        'lofi slow bpm electro chill with organic samples',
        'rock with saturated guitars, a heavy bass line and crazy drum break and fills.',
        'earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves',
    ],
        progress=True)
    return display_audio(res, 32000)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
