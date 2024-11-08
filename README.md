# Multi Agent Framework

中文版本 : [README-ZH](README-ZH.md)

### Introduction

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


You can experience it in [Multi Agent Framework Demo](notebooks/demo.ipynb)

```python

from openai import OpenAI

from agents import BaseAgent,Agent,MultiAgent

client = OpenAI()

# Initialize several agents
general_agent = Agent(base_agent = BaseAgent(client),
           name = "general agent",
           instructions= "Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
           handoff_description = "Call this agent if a user is asking about a topic that is not handled by the activated agent.")

science_agent = Agent(base_agent = BaseAgent(client),
           name = "science agent",
           instructions = "Agent used for science questions named Albert.",
           handoff_description = "Call this agent if a user is asking about a science topic.")

music_agent = Agent(base_agent = BaseAgent(client),
           name = "music agent",
           instructions = "Agent used for music questions named Mozart.",
           handoff_description = "Call this agent if a user is asking about a music topic.")

daily_agent = Agent(base_agent = BaseAgent(client),
           name = "daily agent",
           instructions = "Agent used for daily questions named Ada.",
           handoff_description = "Call this agent if a user is asking about a daily topic.")


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
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])
ma.chat(messages=[{"role": "user", "content": "who are you"}],agent=music_agent)
ma.chat(messages=[{"role": "user", "content": "why the sky is blue and recommend me some music"}],agent=general_agent)


# Multi-Agent tools use example
def get_weather(city:str)->str:
    """ 
    Get the weather for a specified city.
    """
    return f"The weather in {city} is sunny."


def get_news()->str:
    """ 
    Get the latest news.
    """
    return "The latest news is that the sun is shining."


daily_agent.add_tools([get_weather,get_news])

ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou?"}])
ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou and what happened today?"}])


```


### Chat Example


Run
```python
ma.chat(messages=[{"role": "user", "content": "why the sky is blue"}])
```
Output
```bash
[{'role': 'handoff',
  'handoff': 'general_agent -> science_agent',
  'agent_name': 'science_agent',
  'agent': <__main__.Agent at 0x1143e20d0>,
  'message': 'why the sky is blue'},
 {'role': 'assistant',
  'content': "The sky appears blue due to a phenomenon called Rayleigh scattering. Here's how it works:\n\n1. **Sunlight Composition**: Sunlight, or white light, is made up of multiple colors, each with different wavelengths. Blue light has a shorter wavelength, while red light has a longer wavelength.\n\n2. **Atmospheric Interaction**: When sunlight enters the Earth's atmosphere, it interacts with air molecules and small particles. Because blue light is shorter in wavelength, it is scattered in all directions more than other colors when it strikes these molecules.\n\n3. **Observer's Perspective**: As we look up at the sky, we see this scattered blue light coming from all directions, which makes the sky appear blue.\n\n4. **Variations**: During sunrise and sunset, the sky can appear red or orange because the sunlight has to travel through more of the Earth's atmosphere. This causes more scattering of the shorter wavelengths, allowing the longer wavelengths like red and orange to become more prominent.\n\nIn summary, the blue color of the sky is primarily due to the scattering of sunlight by the gases and particles in the atmosphere.",
  'agent_name': 'science_agent'}]
```
Run
```python
ma.chat(messages=[{"role": "user", "content": "who are you"}])
```
Output
```bash
[{'role': 'assistant',
  'content': 'I am Albert, your assistant for science-related questions. How can I help you today?',
  'agent_name': 'science_agent'}]
```

Run
```python
ma.chat(messages=[{"role": "user", "content": "recommend me some music"}])
```
Output
```bash
[{'role': 'handoff',
  'handoff': 'science_agent -> general_agent',
  'agent_name': 'general_agent',
  'agent': <__main__.Agent at 0x114343c10>,
  'message': 'recommend me some music'},
 {'role': 'handoff',
  'handoff': 'general_agent -> music_agent',
  'agent_name': 'music_agent',
  'agent': <__main__.Agent at 0x11437d350>,
  'message': 'recommend me some music'},
 {'role': 'assistant',
  'content': 'Sure! Here are some recommendations across various genres:\n\n**Classical:**\n- **Ludwig van Beethoven** – Symphony No. 9 in D minor, Op. 125 (“Choral”)\n- **Johann Sebastian Bach** – Brandenburg Concerto No. 3\n- **Wolfgang Amadeus Mozart** – Symphony No. 40 in G minor, K. 550\n\n**Jazz:**\n- **Miles Davis** – Kind of Blue\n- **John Coltrane** – A Love Supreme\n- **Ella Fitzgerald & Louis Armstrong** – Ella and Louis\n\n**Rock:**\n- **The Beatles** – Abbey Road\n- **Led Zeppelin** – IV\n- **Fleetwood Mac** – Rumours\n\n**Pop:**\n- **Dua Lipa** – Future Nostalgia\n- **Taylor Swift** – 1989\n- **Billie Eilish** – When We All Fall Asleep, Where Do We Go?\n\n**Indie/Alternative:**\n- **Tame Impala** – Currents\n- **Arctic Monkeys** – AM\n- **Phoebe Bridgers** – Punisher\n\n**Hip-Hop:**\n- **Kendrick Lamar** – To Pimp a Butterfly\n- **J. Cole** – 2014 Forest Hills Drive\n- **OutKast** – Speakerboxxx/The Love Below\n\n**Electronic:**\n- **Daft Punk** – Discovery\n- **ODESZA** – A Moment Apart\n- **Flume** – Flume\n\nIf there’s a specific genre or mood you’re in the mood for, let me know and I can tailor the recommendations further!',
  'agent_name': 'music_agent'}]
```

Run
```python
ma.chat(messages=[{"role": "user", "content": "who are you"}],agent=science_agent)
```
Output
```bash
[{'role': 'assistant',
  'content': 'I am Albert, your assistant for science-related questions. How can I help you today?',
  'agent_name': 'science_agent'}]
```

Run
```python
ma.chat(messages=[{"role": "user", "content": "why the sky is blue and recommend me some music"}],agent=general_agent)
```
Output
```bash
[{'role': 'handoff',
  'handoff': 'general_agent -> science_agent',
  'agent_name': 'science_agent',
  'agent': <__main__.Agent at 0x114341390>,
  'message': 'Why is the sky blue?'},
 {'role': 'handoff',
  'handoff': 'general_agent -> music_agent',
  'agent_name': 'music_agent',
  'agent': <__main__.Agent at 0x11437d350>,
  'message': 'Can you recommend me some music?'},
 {'role': 'assistant',
  'content': "The sky appears blue primarily due to a phenomenon called Rayleigh scattering. Sunlight, or white light, is made up of many colors, each with different wavelengths. When sunlight enters Earth's atmosphere, it interacts with gas molecules and small particles.\n\nShorter wavelengths of light (blue and violet) are scattered more than longer wavelengths (red and yellow). Although violet light is scattered even more than blue, our eyes are more sensitive to blue light and some of the violet is absorbed by the ozone layer. As a result, we perceive the sky as blue during the day.\n\nDuring sunrise and sunset, the light from the sun has to pass through a thicker layer of the atmosphere. The shorter blue wavelengths are scattered out of our line of sight, while the longer wavelengths (reds and oranges) dominate, giving the sky those colors during those times.",
  'agent_name': 'science_agent'},
 {'role': 'assistant',
  'content': 'Of course! What kind of music are you in the mood for? Here are a few suggestions from different genres:\n\n1. **Classical**: Try Beethoven\'s Symphony No. 5 for its dramatic intensity, or Debussy\'s "Clair de Lune" for something more serene.\n\n2. **Jazz**: Listen to Miles Davis\'s "So What" for a classic modal jazz experience, or Ella Fitzgerald and Louis Armstrong\'s duets for some timeless vocal jazz.\n\n3. **Rock**: Check out Led Zeppelin\'s "Stairway to Heaven" for a classic rock anthem, or The Beatles\' "Come Together" for something iconic.\n\n4. **Pop**: Explore Dua Lipa\'s "Don’t Start Now" for a modern dance-pop hit, or Taylor Swift\'s "Blank Space" for catchy storytelling.\n\n5. **Indie**: Give Bon Iver\'s "Holocene" a play for a beautiful, introspective sound, or Tame Impala\'s "The Less I Know the Better" for a groovy vibe.\n\n6. **World Music**: Try listening to Tinariwen, a Tuareg band, for some incredible desert blues, or Carlos Vives for a taste of Colombian vallenato.\n\nLet me know if you’re looking for something specific!',
  'agent_name': 'music_agent'}]
```



### Tools Example

Run
```python

def get_weather(city:str)->str:
    """ 
    Get the weather for a specified city.
    """
    return f"The weather in {city} is sunny."


def get_news()->str:
    """ 
    Get the latest news.
    """
    return "The latest news is that the sun is shining."


daily_agent.add_tools([get_weather,get_news])


ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou?"}])
```
Output
```bash
[{'role': 'handoff',
  'handoff': 'general_agent -> daily_agent',
  'agent_name': 'daily_agent',
  'agent': <__main__.Agent at 0x1143e0850>,
  'message': 'what is the weather in Hangzhou?'},
 {'role': 'tool', 'content': 'The weather in Hangzhou is sunny.'}]
```

Run
```python
ma.chat(messages=[{"role": "user", "content": "what is the weather in Hangzhou and what happened today?"}])
```
Output
```bash
[{'role': 'tool', 'content': 'The weather in Hangzhou is sunny.'},
 {'role': 'tool', 'content': 'The latest news is that the sun is shining.'}]
```



#### Handoff Example

Run
```python
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue"}],agent=general_agent)
```
Output
```python
{'science_agent': <__main__.Agent at 0x116f643d0>}
```

Run
```python
ma.handoff(messages=[{"role": "user", "content": "why the sky is blue and recommend me some music"}],agent=general_agent)
```
Output
```python
{'science_agent': <__main__.Agent at 0x116f643d0>,
 'music_agent': <__main__.Agent at 0x116f641d0>}
```
