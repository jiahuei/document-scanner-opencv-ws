# -*- coding: utf-8 -*-
r"""
Created on 17 Mar 2021 17:02:29
@author: jiahuei
"""
import logging
from typing import List, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Will only work with a single process.
    See https://github.com/encode/broadcaster for Redis support.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        # await websocket.close(code=1000)
        self.active_connections.remove(websocket)
        logger.info(f'{self.__class__.__name__}: "WebSocket {websocket.url.path}" [disconnected]')

    async def send(self, data: Any, websocket: WebSocket, mode: str = "text"):
        if mode == "text":
            await websocket.send_text(data)
        elif mode == "bytes":
            await websocket.send_bytes(data)
        elif mode == "json":
            await websocket.send_json(data, mode="binary")
        else:
            raise ValueError(f"Invalid `mode`: {mode}")

    async def broadcast(self, data: Any, mode: str = "text"):
        for connection in self.active_connections:
            await self.send(data, connection, mode)
