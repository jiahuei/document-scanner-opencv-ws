# -*- coding: utf-8 -*-
r"""
Created on 17 Mar 2021 14:52:33
@author: jiahuei

Server based on FastAPI, with WebSocket endpoints

python main_fastapi_ws.py
gunicorn main_fastapi_ws:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000
hypercorn main_fastapi_ws:app -w 1 -k uvloop --bind 0.0.0.0:5000

"""
import os
import logging
import uvicorn
from time import perf_counter
from typing import Optional, Union, List
from pydantic import BaseModel
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from scanner.detect_doc import A4Detector, visualise_detection
from scanner.utils import image as image_utils, misc as misc_utils

logger = logging.getLogger(__name__)
log_level = str(os.getenv("WEBSOCKET_LOG_LEVEL", "info"))
misc_utils.configure_logging(log_level.upper(), logger_obj=logger)

doc_detector = A4Detector(
    use_image_features=bool(os.getenv("DET_SIFT_FEATURE", False)),
    doc_extract_width=int(os.getenv("DOC_EXTRACT_WIDTH", 1414)),
    doc_extract_height=int(os.getenv("DOC_EXTRACT_HEIGHT", 2000)),
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
web_app = Starlette(
    routes=[
        # Route("/", upload_page, methods=["GET", "POST"]),
        Mount("/static", app=StaticFiles(directory="static"), name="static"),
    ],
)
app.mount("/web_app", web_app)


class DocImage(BaseModel):
    name: Optional[str] = None
    image: str
    doc_type: int = 0


@app.websocket("/detect_doc/{mode}")
async def detect_document(websocket: WebSocket, mode: str):
    if mode not in ("real", "debug_upload"):
        raise ValueError(
            f"Expected parameter `mode` to be one of (`real`, `debug_upload`), saw `{mode}`"
        )
    await websocket.accept()
    try:
        while True:
            item = await websocket.receive_json(mode="text")
            tic = perf_counter()
            item = DocImage(**item)
            image = image_utils.base64str_to_ndarray(item.image)
            res = visualise_detection(
                detector=doc_detector,
                image=image
            )
            elapsed_time = perf_counter() - tic
            logger.info(f"Time taken for card check: {elapsed_time:.6f} sec")
            await websocket.send_json(res, mode="text")
    except WebSocketDisconnect:
        logger.info(f'"WebSocket {websocket.url.path}" [disconnected]')


@app.get("/", response_class=HTMLResponse)
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload_ws.html", {"request": request})


if __name__ == "__main__":
    host = str(os.getenv("WEBSOCKET_HOST", "0.0.0.0"))
    port = int(os.getenv("WEBSOCKET_PORT", 5000))
    uvicorn.run("main_fastapi_ws:app", host=host, port=port, log_level=log_level)
