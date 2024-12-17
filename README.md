# Multi Agent Framework

## Introduction

The Multi-Agent Framework can communicate, execute tasks, and manage low-level control.

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

## Tutorial

- [How to build Group of Agents](examples/001%20group.ipynb)
- [Chat with Group of Agents](examples/002%20chat.ipynb)
- [Task for Group of Agents](examples/002%20task.ipynb)
- [Low Level API for Group Discussion with Human in the Loop](examples/999%20low-level.ipynb)


## Usage

### Step Zero

```python
from protocol import Env,Message
from group import Group
from agent import Agent
from openai import OpenAI
import os
```

### Step One

Creat Agent like this 

```python
artist = Agent(name="agent2", 
        role="Artist", 
        description="Transfer to me if you need help with art.",
        model_client=OpenAI(),
        verbose=True)
```

or like this (third-party agent like Dify)

```python
mathematician = Agent(name="agent1", 
    role="Mathematician", 
    description="Transfer to me if you need help with math.", 
    dify_access_token=os.environ.get("AGENT1_ACCESS_TOKEN"),
    verbose=True)
```


### Step Two

Create Env like this

```python

env = Env(
    description="This is a test environment",
    members=[mathematician, artist]
)
```

or like this

```python
env = Env(
    description="This is a test environment",
    members=[mathematician, artist],
    relationships={"agent1": ["agent2"]}
)
```

or with set output language (default is English)

```python
env = Env(
    description="This is a test environment",
    members=[mathematician, artist],
    language="中文"
)
```


### Step Three

Build Group like this

```python
g = Group(env=env,model_client=model_client,verbose=True)
```


### Step Four

Some examples of how to use the group

```python
g.user_input("can you help me with math?")
next_agent = g.handoff(next_speaker_select_mode="auto2",include_current=True,model="gpt-4o-mini")
```

```python
g.user_input("How about music for reading?")
response = g.call_agent(next_speaker_select_mode="auto2",include_current=True,model="gpt-4o-mini")
```

```python
response= g.chat("Can explain the concept of complex numbers?")
```

```python
response = g.task("I want to build a simplistic and user-friendly bicycle help write a design brief.",model="gpt-4o-mini",strategy="auto")
```


