# -*- coding: utf-8 -*-
"""
@Time: 2024/12/25 17:00
@Author: ZJun
@File: agent.py
@Description: This file build a simple HTTP agent demo by using FastAPI.
"""


from fastapi import FastAPI
from openai import AsyncOpenAI
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

client = AsyncOpenAI()

class Input(BaseModel):
    content: str

@app.post("/http_agent_demo")
async def do(input: Input):
    model = "gpt-4o-mini"
    system_message = [{"role": "system", "content": "You are a helpful assistant. Your name is 'HTTP Agent'"}]
    messages = system_message + [{"role": "user", "content": input.content}]

    response = await client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    res = response.choices[0].message.content

    return {"response": res}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1415)


# ------------------------------------------------------------------------------------------------

# python src/service/http_agent_demo.py


# ------------------------------------------------------------------------------------------------

# import requests

# def send_request():
#     url = "http://localhost:1415/http_agent_demo"
#     data = {"content": "你是谁？"}
    
#     response = requests.post(url, json=data)
#     print(response.json())

# send_request()

# ------------------------------------------------------------------------------------------------

# curl -X POST "http://localhost:1415/http_agent_demo" \
# -H "Content-Type: application/json" \
# -d '{"content": "你是谁？"}'


# ------------------------------------------------------------------------------------------------