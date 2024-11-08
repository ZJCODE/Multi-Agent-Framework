
from openai import OpenAI
from typing import Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import List, Optional, Dict
import logging
from utils import function_to_schema
import json
import copy


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
            base_agent:BaseAgent,
            name:str,
            instructions: Optional[str] = None,
            handoff_description: Optional[str] = None,
            ):
        self.DEFAULT_MODEL = "gpt-4o-mini"
        self.name = name.replace(" ","_")
        self.agent = base_agent
        self.instructions = instructions
        self.handoff_description = handoff_description
        self.handoff_agent_schemas: List[Dict] = []
        self.tools_schema: List[Dict] = []
        self.handoff_agents: Dict[str, "Agent"] = {}
        self.tools_map: Dict[str, "function"] = {}

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
                            "type": "function",
                            "function": {
                                "name": agent.name,
                                "description": agent.instructions or "" + " \n" + agent.handoff_description or "",
                                'parameters': {
                                    "type": "object",
                                    "properties": {
                                        "message": {
                                            "type": "string",
                                            "description": "The message to send to the agent."
                                            }
                                    },
                                    "required": ["message"]
                                }
                            }
                            }
            self.handoff_agent_schemas.append(agent_schema)
            self.handoff_agents[agent.name] = agent

    def clear_handoffs(self) -> None:
        """ 
        Clears handoff agents from the current agent.
        """
        self.handoff_agent_schemas.clear()
        self.handoff_agents = {}

    def add_tools(self, tools: list) -> None:
        """ 
        Adds tools to the current agent.
        """
        self.tools_schema.clear()
        for tool in tools:
            tool_schema = function_to_schema(tool)
            self.tools_schema.append(tool_schema)
            self.tools_map[tool.__name__] = tool

    def clear_tools(self) -> None:
        """
        Clears tools from the current agent.
        """
        self.tools_schema.clear()

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat(self,
             messages:List[Dict],
             model: Optional[str] = None,
             disable_tools:bool = False,
             disable_handoffs:bool = False):
        """ 
        Handles chat messages and returns responses, with optional handoff to other agents.
        """

        if self.instructions:
            messages = [{"role": "system", "content": self.instructions}] + messages
        
        model = model or self.DEFAULT_MODEL

        try:

            
            tools = None if disable_tools else self.tools_schema or None
            handoffs = None if disable_handoffs else self.handoff_agent_schemas or None

            response = self.agent.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=tools + handoffs if tools and handoffs else tools or handoffs,
                            tool_choice=None,
                        )
            message = response.choices[0].message
            if not message.tool_calls:
                result = [{"role": "assistant", "content": message.content, "sender": self.name}]
                return result
            
            result = []
            handoff_agents_num = sum([1 for tool_call in message.tool_calls if tool_call.function.name in self.handoff_agents])
            for tool_call in message.tool_calls:
                if tool_call.function.name in self.handoff_agents:
                    handoff_agent = self.handoff_agents[tool_call.function.name]
                    if handoff_agents_num > 1:
                        message = json.loads(tool_call.function.arguments).get("message")
                        arguments = tool_call.function.arguments
                    else:
                        message = messages[-1].get("content")
                        arguments = json.dumps({"message": message})
                    tool_call_message = {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": tool_call.id, 
                             "function": {"arguments": arguments, "name": tool_call.function.name},
                             "type": "function"
                             }
                        ],
                        "type": "handoff",
                        "sender": self.name,
                    }
                    handoff_message = {
                        "role": "tool",
                        "handoff": f"{self.name} -> {handoff_agent.name}",
                        "content":f"Handing off to {handoff_agent.name}",
                        "tool_call_id": tool_call.id,
                        "agent": handoff_agent,
                        "message": message,
                        "type": "handoff",
                        "sender": handoff_agent.name,
                    }
                    result.append(tool_call_message)
                    result.append(handoff_message)
                elif tool_call.function.name in self.tools_map:
                    tool = self.tools_map[tool_call.function.name]
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = tool(**tool_args)

                    tool_call_message = {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": tool_call.id, 
                             "function": {"arguments": tool_call.function.arguments, "name": tool_call.function.name},
                             "type": "function"
                             }
                        ],
                        "type": "tool",
                        "sender": self.name,
                    }
                    tool_message = {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                        "type": "tool",
                        "sender": self.name,
                    }
                    result.append(tool_call_message)
                    result.append(tool_message)
                    # result.append(self.chat( [tool_call_message] + [tool_message], model, disable_tools=True, disable_handoffs=True)[0])
                    logging.info(f"Tool call: {tool_call.function.name} with args: {tool_args} returned: {tool_result}")
                else:
                    logging.error(f"Unknown tool call: {tool_call.function.name}")

            tool_message_for_summary = [r for r in result if ('tool_calls' in r or 'tool_call_id' in r) and r['type'] == 'tool']
            if len(tool_message_for_summary) > 0:
                temp_messages = copy.deepcopy(messages) + tool_message_for_summary
                temp_result = self.chat(temp_messages, model, disable_tools=True, disable_handoffs=True)
                result.extend(temp_result)

            return result
        except Exception as e:
            logging.error("Unable to generate ChatCompletion response")
            logging.error(f"Exception: {e}")
            return None

    def do_task():
        pass

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

    def chat(self,
             messages:List[Dict],
             model: Optional[str] = None,
             agent:Optional[Agent] = None,
             max_handoof_depth:int = 2,
             disable_tools:bool = False,
             disable_handoffs:bool = False,
             show_details:bool = False
             )->List[Dict]:
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
            res = self.current_agent.chat(messages, model, disable_tools=disable_tools, disable_handoffs=disable_handoffs)

            if not res:
                return res

            if len(res) == 1 and res[0].get("content"):
                return res
            elif len(res) == 2: # just a handoff
                # Handle handoff
                while res[-1].get("handoff") and max_handoof_depth > 0:
                    max_handoof_depth -= 1
                    self.current_agent = res[-1]["agent"]
                    disable_handoffs = max_handoof_depth == 0
                    res.extend(self.current_agent.chat(messages, model, disable_handoffs=disable_handoffs))
            else: # more than one handoff
                for r in res:
                    if r.get("handoff"):
                        temp_agent = r["agent"]
                        temp_messages = copy.deepcopy(messages)
                        temp_messages[-1]["content"] = r["message"]
                        res.extend(temp_agent.chat(temp_messages, model, disable_handoffs=True))
            
            for r in res:
                if 'agent' in r:
                    r.pop('agent')

            if show_details:
                return res
            else:
                return [r for r in res if 'tool_call_id' not in r and 'tool_calls' not in r]


            
        except Exception as e:
            logging.error("Error during chat with agent.")
            logging.error(f"Exception: {e}")
            return None
        
    def handoff(self,
                messages:List[Dict],
                model: Optional[str] = None,
                agent:Agent = None)-> Agent:
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
                    if r.get("handoff"):
                        handoff_agents[r["sender"]] = r["agent"]
            if handoff_agents:
                return handoff_agents
            return None
        except Exception as e:
            logging.error("Error during handoff with agent.")
            logging.error(f"Exception: {e}")
            return None