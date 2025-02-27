import uvicorn

from service import FasterWhisper

if __name__ == "__main__":
    app = FasterWhisper.to_asgi()  # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8004)
