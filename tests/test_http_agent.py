import requests

def send_request():
    url = "http://localhost:1415/http_agent_demo"
    data = {"content": "你是谁？"}
    
    response = requests.post(url, json=data)
    print(response.json())

send_request()