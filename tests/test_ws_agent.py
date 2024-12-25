import asyncio
import websockets
import json

async def send_message():
    uri = "ws://localhost:5358/ws_agent_demo"
    async with websockets.connect(uri) as websocket:
        message = {"content": "你是谁？"}
        await websocket.send(json.dumps(message))
        
        response = await websocket.recv()
        print(f"Response from server: {response}")

asyncio.run(send_message())
