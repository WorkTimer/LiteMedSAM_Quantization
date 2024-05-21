from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api import api_app

app = FastAPI(docs_url=None, redoc_url=None)

@app.get("/")
async def root():
    return FileResponse('static/index.html')

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


app.mount("/api", api_app)

