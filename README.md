# Multi Agent Framework

中文版本 : [README-ZH](README-ZH.md)

## Introduction

Multi Agent Framework is a multi-agent dialogue framework based on the OpenAI API, which can realize the dialogue cooperation between multiple agents and realize more complex dialogue scenarios.

## Pre-requirements

```bash
pip install -r requirements.txt
cp .env_example .env
```

write your own KEY and URL(optional)

```
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx
```


## Usage


You can experience it in [Multi Agent Framework Demo](notebook/demo.ipynb)

```python

from openai import OpenAI

from agents import BaseAgent,Agent,MultiAgent

client = OpenAI()

# Initialize several agents
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


# Build a Multi-Agent environment & add handoff relationships
ma = MultiAgent(start_agent=general_agent)
ma.add_handoff_relations(from_agent=general_agent,to_agents=[science_agent,music_agent,daily_agent])
ma.add_handoff_relations(from_agent=science_agent,to_agents=[general_agent])
ma.add_handoff_relations(from_agent=music_agent,to_agents=[general_agent])
ma.add_handoff_relations(from_agent=daily_agent,to_agents=[general_agent])
# ma.add_handoff_relations(from_agent=science_agent,to_agents=[general_agent,music_agent,daily_agent])
# ma.add_handoff_relations(from_agent=music_agent,to_agents=[general_agent,science_agent,daily_agent])
# ma.add_handoff_relations(from_agent=daily_agent,to_agents=[general_agent,science_agent,music_agent])



# Multi-Agent handoff example
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue"}],agent=general_agent)



# Multi-Agent dialogue example
# agent will be automatically selected according to the content of the user's message
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])
# set agent to music_agent can force the conversation to be handled by music_agent
ma.chat(messages=[{"role": "user", "content": "who are you"}],agent=music_agent) 
ma.chat(messages=[{"role": "user", "content": "why the sky is blue and recommend me some music"}],agent=general_agent)


# Multi-Agent tools use example
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

## Examples

- [Multi Agent Framework Demo 1](notebook/structure_top_down.ipynb)