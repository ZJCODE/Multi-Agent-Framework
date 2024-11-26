# Multi Agent Framework

## 简介

Multi-Agent-Framework 是一个基于OpenAI API的多Agent对话框枧架，可以实现多个Agent之间的对话协作，实现更加复杂的对话场景。

## 基础准备

```bash
pip install -r requirements.txt
cp .env_example .env
```

填写个人的KEY和URL(可选)

```
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx
```


## 使用案例

具体可在 [Multi Agent Framework Demo](notebook/demo.ipynb) 中体验

```python

from openai import OpenAI

from agents import BaseAgent,Agent,MultiAgent

client = OpenAI()

# 初始化几个Agent
general_agent = Agent(base_agent = BaseAgent(client),
           name = "general agent",
           instructions= "Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
           handoff_description = "Call this agent if you don't know how to answer the user's question or do not has access to the necessary information.")

science_agent = Agent(base_agent = BaseAgent(client),
           name = "science agent",
           instructions = "Agent who is knowledgeable about science topics and can answer questions about them.",
           handoff_description = "Call this agent if a user is asking about a science topic like physics, chemistry, biology, etc.")

music_agent = Agent(base_agent = BaseAgent(client),
           name = "music agent",
           instructions = "Agent who is knowledgeable about music topics and can answer questions about them.",
           handoff_description = "Call this agent if a user is asking about a music topic like music theory, music history, music genres, etc.")

daily_agent = Agent(base_agent = BaseAgent(client),
           name = "daily agent",
           instructions = "Agent who is knowledgeable about daily topics and can answer questions about them.",
           handoff_description = "Call this agent if a user is asking about a daily topic like weather, news, etc.")




# 创建Multi-Agent环境 & 增加handoff的关系
ma = MultiAgent(start_agent=general_agent)
ma.add_handoff_relations(from_agent=general_agent,to_agents=[science_agent,music_agent,daily_agent])
ma.add_handoff_relations(from_agent=science_agent,to_agents=[general_agent])
ma.add_handoff_relations(from_agent=music_agent,to_agents=[general_agent])
ma.add_handoff_relations(from_agent=daily_agent,to_agents=[general_agent])
# ma.add_handoff_relations(from_agent=science_agent,to_agents=[general_agent,music_agent,daily_agent])
# ma.add_handoff_relations(from_agent=music_agent,to_agents=[general_agent,science_agent,daily_agent])
# ma.add_handoff_relations(from_agent=daily_agent,to_agents=[general_agent,science_agent,music_agent])



# Multi-Agent handoff 的能力展示
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue"}],agent=general_agent)



# Multi-Agent 对话能力展示 
# 会根据用户的消息内容自动选择agent
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])
# 设置agent为music_agent可以强制对话由music_agent处理
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])
ma.chat(messages=[{"role": "user", "content": "who are you"}],agent=music_agent)
ma.chat(messages=[{"role": "user", "content": "why the sky is blue and recommend me some music"}],agent=general_agent)


# Multi-Agent 工具使用展示
def get_weather(city:str)->str:
    """ 
    Get the weather for a specified city.
    """
    return f"The weather in {city} is sunny."


def get_today_news()->str:
    """ 
    Get today's news.
    """
    return "ZJun created an Agent Framework."


daily_agent.add_tools([get_weather,get_news])

ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou?"}])
ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou and what happened today?"}])


```

## 更多案例

- [至上而下结构](notebook/structure_top_down.ipynb)
- [自主构建新的Agent并交接](notebook/auto_create_agent.ipynb)

## 实验尝试

- [Stand-alone Multi-Agent Framework](experiment/standalone_multi_framework.ipynb)
- [Stand-alone Multi-Agent Framework V2](experiment/standalone_multi_framework_v2.ipynb)
- [Stand-alone Multi-Agent Framework V3 Async](experiment/standalone_multi_framework_v3.ipynb)