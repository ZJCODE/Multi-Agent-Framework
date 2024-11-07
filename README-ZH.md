# Multi Agent Framework

### 简介

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

具体可在 [Multi Agent Framework](MultiAgent.ipynb) 中体验

```python

from openai import OpenAI

from agents import BaseAgent,Agent,MultiAgent

client = OpenAI()

# 初始化几个Agent
general_agent = Agent(base_agent = BaseAgent(client),
           name = "general agent",
           instructions= "Agent used for daily questions named Ada.")

science_agent = Agent(base_agent = BaseAgent(client),
           name = "science agent",
           instructions = "Agent used for science questions named Albert.")

music_agent = Agent(base_agent = BaseAgent(client),
           name = "music agent",
           instructions = "Agent used for music questions named Mozart.")

# 创建Multi-Agent环境 & 增加handoff的关系
ma = MultiAgent(start_agent=general_agent)
ma.add_handoff_relations(from_agent= general_agent,to_agents=[science_agent,music_agent])
ma.add_handoff_relations(from_agent= science_agent,to_agents=[general_agent])
ma.add_handoff_relations(from_agent= music_agent,to_agents=[general_agent])
# ma.add_handoff_relations(from_agent=science_agent,to_agents=[general_agent,music_agent])
# ma.add_handoff_relations(from_agent=music_agent,to_agents=[general_agent,science_agent])


# Multi-Agent handoff 的能力展示
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue"}],agent=a1)

# Multi-Agent 对话能力展示 
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])

# Multi-Agent 工具使用展示

def get_weather(city:str)->str:
    """ 
    Get the weather for a specified city.
    """
    return f"The weather in {city} is sunny."

general_agent.add_tools([get_weather])

ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou?"}])



```


### Chat Example


运行
```python
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])
```
输出
```bash
[{'role': 'tool',
  'handoff': 'general_agent -> science_agent',
  'agent_name': 'science_agent',
  'agent': <__main__.Agent at 0x116f643d0>},
 {'role': 'assistant',
  'content': "The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth's atmosphere, it collides with molecules and small particles in the air. Sunlight is made up of many colors, each with different wavelengths. Blue light has a shorter wavelength and is scattered more than other colors when it strikes these particles.\n\nDuring the day, when the sun is high in the sky, more of the blue light is scattered in all directions, making the sky appear blue to our eyes. At sunrise and sunset, the sun is lower on the horizon, and the light has to pass through a greater thickness of the atmosphere. As a result, the blue and green wavelengths are scattered out of our line of sight, and the longer wavelengths like orange and red become more visible, leading to the beautiful colors we see at those times.",
  'agent_name': 'science_agent'}]
```
运行
```python
ma.chat(messages=[{"role": "user", "content": "who are you"}])
```
输出
```bash
[{'role': 'assistant',
  'content': 'I am Albert, your virtual assistant for science questions. How can I assist you today?',
  'agent_name': 'science_agent'}]
```

运行
```python
ma.chat(messages=[{"role": "user", "content": "What are the different music styles?"}])
```
输出
```bash
[{'role': 'tool',
  'handoff': 'science_agent -> general_agent',
  'agent_name': 'general_agent',
  'agent': <__main__.Agent at 0x116f64410>},
 {'role': 'tool',
  'handoff': 'general_agent -> music_agent',
  'agent_name': 'music_agent',
  'agent': <__main__.Agent at 0x116f641d0>},
 {'role': 'assistant',
  'content': 'Music styles encompass a vast array of genres and subgenres. Here are some of the main categories:\n\n1. **Classical**: Includes orchestral, chamber music, opera, and choral works, with notable periods such as Baroque, Classical, Romantic, and Contemporary.\n\n2. **Jazz**: Characterized by improvisation and swing, with styles like bebop, smooth jazz, and free jazz.\n\n3. **Rock**: Evolved from rock and roll, encompassing various subgenres like classic rock, punk rock, indie rock, and alternative rock.\n\n4. **Pop**: Popular music aimed at a wide audience, often characterized by catchy melodies and hooks.\n\n5. **Hip-Hop**: Focuses on rhythm and beats, incorporating rapping, DJing, and sampling; includes subgenres like trap and boom bap.\n\n6. **R&B (Rhythm and Blues)**: Combines elements of soul, funk, and pop; often emphasizes vocal performances.\n\n7. **Country**: Originated in the Southern United States; includes subgenres like country pop, bluegrass, and Americana.\n\n8. **Electronic**: Encompasses a range of styles produced using electronic instruments; includes techno, house, trance, and dubstep.\n\n9. **Reggae**: Originating in Jamaica, characterized by offbeat rhythms and socially conscious lyrics.\n\n10. **Blues**: Rooted in African American history, featuring expressive guitar work and a melancholic themes.\n\n11. **Folk**: Traditional music that reflects cultural stories and values, with regional variations.\n\n12. **Metal**: Features heavy guitar riffs and strong rhythms; includes subgenres like heavy metal, thrash metal, and black metal.\n\nThese styles often blend into one another, leading to numerous hybrid genres. Music is constantly evolving, and new styles continue to emerge as artists experiment and innovate.',
  'agent_name': 'music_agent'}]
```

运行
```python
ma.chat(messages=[{"role": "user", "content": "who are you"}],agent=science_agent)
```
输出
```bash
[{'role': 'assistant',
  'content': 'I am Albert, your assistant for science-related questions. How can I help you today?',
  'agent_name': 'science_agent'}]
```

运行
```python
ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou?"}])
```
输出
```bash
[{'role': 'handoff',
  'handoff': 'science_agent -> general_agent',
  'agent_name': 'general_agent',
  'agent': <__main__.Agent at 0x11528e1d0>},
 {'role': 'tool',
  'content': 'The weather in Hangzhou is sunny.'}]
```


#### Handoff Example

运行
```python
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue"}],agent=a1)
```
输出
```python
{'science_agent': <__main__.Agent at 0x116f643d0>}
```

运行
```python
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue and recommend me some music"}],agent=a1)
```
输出
```python
{'science_agent': <__main__.Agent at 0x116f643d0>,
 'music_agent': <__main__.Agent at 0x116f641d0>}
```
