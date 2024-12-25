import websockets.sync.client
import json

def send_message():
    uri = "ws://localhost:5358/ws_agent_demo"
    ws = websockets.sync.client.connect(uri)

    message = {"content": "你是谁？"}
    ws.send(json.dumps(message))
    
    response = ws.recv()
    print(f"Response from server: {response}")

send_message()