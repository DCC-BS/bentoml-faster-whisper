import uvicorn

from bentoml_faster_whisper.service import FasterWhisper

if __name__ == "__main__":
    app = FasterWhisper.to_asgi()
    uvicorn.run(app, host="0.0.0.0", port=8004)
