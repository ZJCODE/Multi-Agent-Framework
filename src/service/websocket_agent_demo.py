# -*- coding: utf-8 -*-
"""
@Time: 2024/12/25 17:00
@Author: ZJun
@File: agent.py
@Description: This file build a simple WebSocket agent demo by using WebSocket.
"""

from fastapi import FastAPI, WebSocket
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

client = AsyncOpenAI()

class Input(BaseModel):
    content: str

@app.websocket("/ws_agent_demo")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            input = Input(**data)
            model = "gpt-4o-mini"
            system_message = [{"role": "system", "content": "You are a helpful assistant. Your name is 'WebSocket Agent'"}]
            messages = system_message + [{"role": "user", "content": input.content}]

            response = await client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            res = response.choices[0].message.content

            await websocket.send_json({"response": res})
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5358)



# ------------------------------------------------------------------------------------------------

# python src/service/websocket_agent_demo.py


# ------------------------------------------------------------------------------------------------

# import asyncio
# import websockets
# import json

# async def send_message():
#     uri = "ws://localhost:5358/ws_agent_demo"
#     async with websockets.connect(uri) as websocket:
#         message = {"content": "你是谁？"}
#         await websocket.send(json.dumps(message))
        
#         response = await websocket.recv()
#         print(f"Response from server: {response}")

# asyncio.run(send_message())

# ------------------------------------------------------------------------------------------------

# import websockets.sync.client
# import json

# def send_message():
#     uri = "ws://localhost:5358/ws_agent_demo"
#     ws = websockets.sync.client.connect(uri)

#     message = {"content": "你是谁？"}
#     ws.send(json.dumps(message))
    
#     response = ws.recv()
#     print(f"Response from server: {response}")

# send_message()

# ------------------------------------------------------------------------------------------------

