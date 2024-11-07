
from openai import OpenAI
from typing import Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import List, Optional, Dict
import logging


class BaseAgent:
    """
    An example of a base agent class, including an OpenAI client.
    """
    def __init__(self,client:OpenAI):
        self.client = client
        # can add more attributes and methods here
        

class Agent:
    """
    A class representing an agent that can handle chat messages and handoff to other agents.
    """
    def __init__(
            self,
            agent:BaseAgent,
            name:str,
            instructions: Optional[str] = None):
        self.DEFAULT_MODEL = "gpt-4o-mini"
        self.name = name.replace(" ","_")
        self.agent = agent
        self.instructions = instructions
        self.handoff_agent_schemas: List[Dict] = []
        self.handoff_agents: Dict[str, "Agent"] = {}

    def __str__(self):
        """
        Returns a string representation of the Agent instance.
        """
        return f"Agent(name={self.name}, instructions={self.instructions},handoff_agents={list(self.handoff_agents.keys())})"


    def add_handoffs(self, agents: List["Agent"]) -> None:
        """ 
        Adds handoff agents to the current agent.
        """
        self.handoff_agent_schemas.clear()
        for agent in agents:
            agent_schema = {
                'type': 'function',
                'function': {
                    'name': agent.name,
                    'description': agent.instructions or '',
                    'parameters': {'type': 'object', 'properties': {}, 'required': []}
                }
            }
            self.handoff_agent_schemas.append(agent_schema)
            self.handoff_agents[agent.name] = agent

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat(self,messages:List[Dict],model: Optional[str] = None,disable_tools:bool = False):
        """ 
        Handles chat messages and returns responses, with optional handoff to other agents.
        """

        if self.instructions:
            messages = [{"role": "system", "content": self.instructions}] + messages
        
        model = model or self.DEFAULT_MODEL

        try:

            response = self.agent.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=None if disable_tools else self.handoff_agent_schemas or None,
                tool_choice=None,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                result = [{"role": "assistant", "content": message.content, "agent_name": self.name}]
                return result
            
            result = []
            for tool_call in message.tool_calls:
                call_agent = self.handoff_agents[tool_call.function.name]
                call_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "handoff": f"{self.name} -> {call_agent.name}",
                    "agent_name": call_agent.name,
                    "agent": call_agent,
                }
                result.append(call_message)      
            return result
        except Exception as e:
            logging.error("Unable to generate ChatCompletion response")
            logging.error(f"Exception: {e}")
            return None


class MultiAgent:
    def __init__(self,start_agent:Optional[Agent] = None):
        """
        Initialize the MultiAgent with an optional starting agent.
        """
        self.current_agent = start_agent

    def add_handoff_relations(self,from_agent:Agent,to_agents:List[Agent])->None:
        """
        Add handoff relations from one agent to multiple agents.
        """
        from_agent.add_handoffs(to_agents)

    def chat(self,messages:List[Dict],model: Optional[str] = None,agent:Optional[Agent] = None,max_handoof_depth:int = 2)->List[Dict]:
        """
        Chat with the current agent or a specified agent using the provided messages and model.
        """
        if agent:
            self.current_agent = agent

        if not self.current_agent:
            logging.error("No agent is set for conversation. You can set it in init or pass as argument.")
            return None
        
        try:
            # currently only one handoff is supported
            res = [self.current_agent.chat(messages, model)[0]]
            if res and res[0].get("content"):
                return res
            while res and res[-1].get("tool_call_id") and max_handoof_depth > 0:
                max_handoof_depth -= 1
                self.current_agent = res[-1]["agent"]
                if max_handoof_depth == 0:
                    res.extend(self.current_agent.chat(messages, model, disable_tools=True))
                else:
                    res.extend(self.current_agent.chat(messages, model))
            return res
        except Exception as e:
            logging.error("Error during chat with agent.")
            logging.error(f"Exception: {e}")
            return None
        
    def handoff(self,messages:List[Dict],model: Optional[str] = None,agent:Agent = None)-> Agent:
        """
        Perform a handoff to the specified agent using the provided messages and model.
        """
        current_agent = agent

        if not current_agent:
            logging.error("No agent is set for conversation. You must specify an agent.")
            return None
        
        try:
            res = current_agent.chat(messages, model)
            handoff_agents = {}
            if res:
                for r in res:
                    if r.get("tool_call_id"):
                        handoff_agents[r["agent_name"]] = r["agent"]
            if handoff_agents:
                return handoff_agents
            return None
        except Exception as e:
            logging.error("Error during handoff with agent.")
            logging.error(f"Exception: {e}")
            return None