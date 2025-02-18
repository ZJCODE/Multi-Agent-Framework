# -*- coding: utf-8 -*-
"""
@Time: 2024/12/25 14:00
@Author: ZJun
@File: agent.py
@Description: This file contains the Agent class which is a subclass of the Member class. The Agent class is used to represent an agent in the system.
"""

from typing import List,Union,Dict
from openai import OpenAI,AsyncOpenAI
import requests
import json
import websockets.sync.client

from src.utilities.logger import Logger
from src.utilities.utils import function_to_schema
from src.protocol import Member,Message
from src.memory import Memory
from src.planner import Planner


class Agent(Member):
    def __init__(
            self, 
            name: str, 
            role: str, 
            description: str = None,
            persona: str = None, # context and personality of the agent more detailed than description
            model_client: Union[OpenAI, AsyncOpenAI] = None,
            temperature: float = None, # Temperature for openai model
            tools: List["function"] = None, # List of Python Functions for openai model
            dify_access_token: str = None,
            iagent_url: str = None,
            websocket_url: str = None,
            verbose: bool = False
            ):
        """
        Initializes the Agent class.

        Args:
            name (str): The name of the agent. (handoff reference)
            role (str): The role of the agent. (handoff reference)
            description (str, optional): The simple description of the agent. Defaults to None. (handoff reference)
            persona (str, optional): The persona of the agent can add more personality,backsotry and skills to the agent. Defaults to None.
            model_client (Union[OpenAI, AsyncOpenAI], optional): The model client for the agent. Defaults to None.
            temperature (float, optional): The temperature for the agent. Defaults to 0.5.
            tools (List["function"], optional): The tools for the agent. Defaults to None.
            dify_access_token (str, optional): The Dify access token for the agent. Defaults to None.
            iagent_url (str, optional): The iAgent URL for the agent. Defaults to None.
            websocket_url (str, optional): The websocket URL for the agent. Defaults to None.
            verbose (bool, optional): The verbosity of the agent. Defaults to False.
        
        """
        super().__init__(name, role, description)
        self._logger = Logger(verbose=verbose)
        self.persona = persona
        self.model_client = model_client
        self.temperature = temperature
        self.tools = tools
        self.dify_access_token = dify_access_token
        self.iagent_url = iagent_url
        self.websocket_url = websocket_url
        self.verbose = verbose
        self._connect_to_websocket()
        # Tools Related Attributes
        self.tools_schema: List[Dict] = []
        self.tools_map: Dict[str, "function"] = {}
        self._process_tools()
        self.memory = None
        self.planner = None
        
    def __str__(self):
        return f"{self.name} is a {self.role}."
    
    def do(self, 
           message: str,
           sender: str = None,
           model:str="gpt-4o-mini",
           use_tools:bool=True,use_memory:bool=True,use_planner:bool=True,
           keep_memory:bool=True) -> List[Message]:
        if self.dify_access_token:
            self._logger.log(level="info", message=f"Calling Dify agent [{self.name}]",color="bold_green")
            response = self._call_dify_http_agent(self.dify_access_token, message)
        elif self.iagent_url:
            self._logger.log(level="info", message=f"Calling iAgent agent [{self.name}]",color="bold_green")
            response = self._call_iagent_http_agent(message,sender)
        elif self.websocket_url:
            self._logger.log(level="info", message=f"Calling Websocket agent [{self.name}]",color="bold_green")
            response = self._call_websocket_agent(message)
        elif isinstance(self.model_client,OpenAI):
            self._logger.log(level="info", message=f"Calling OpenAI agent [{self.name}]",color="bold_green")
            response = self._call_openai_agent(message,model,use_tools,use_memory,use_planner,keep_memory)
        else:
            self._logger.log(level="error", message=f"No model client or Dify access token provided for agent {self.name}.",color="red")
            raise ValueError("No model client or Dify access token provided, please provide one for agent {self.name}.")
        return response

    def init_memory(self,working_memory_threshold:int=10,semantic_memory_db_path ="temp",model:str="gpt-4o-mini",language:str=None) -> None:
        """
        Initializes the memory for the agent. Currently only supported for OpenAI model client agent.
        """
        if not self.model_client:
            self.memory = None
            self._logger.log(level="error", message=f"Currently Memory is only supported for OpenAI model client.",color="bold_red")
            return
        self.memory = Memory(working_memory_threshold,self.model_client, model, verbose=self.verbose,db_path=semantic_memory_db_path,language=language)
        self._logger.log(level="info", message=f"Memory initialized for agent {self.name}.",color="bold_green")

    def init_planner(self,model:str="gpt-4o-mini",language:str=None) -> None:
        if not self.model_client:
            self.planner = None
            self._logger.log(level="error", message=f"Currently Planner is only supported for OpenAI model client.",color="bold_red")
            return
        self.planner = Planner(self.model_client, model, verbose=self.verbose,language=language)
        self._logger.log(level="info", message=f"Planner initialized for agent {self.name}.",color="bold_green")

    def add_memory(self,memory:str) -> None:
        if not self.memory:
            return
        self.memory.manual_add_long_term_memory(memory)

    def add_working_memory(self,memory:str) -> None:
        if not self.memory:
            return
        self.memory.add_working_memory(memory)

    def _call_openai_agent(self,query:str,
                           model:str="gpt-4o-mini",
                           use_tools:bool=True,
                           use_memory:bool=False,
                           use_planner:bool=False,
                           keep_memory:bool=False
                           ) -> List[Message]:
        """
        This function calls the agent function to get the response.

        Args:
            query (str): The query to send to the agent.

        Returns:
            Message: The response from the agent.
        """

        instructions =(
            f"## Your Name is :\n {self.name}\n\n"
            f"## Your Role is :\n {self.role}\n\n"
            f"## Description:\n {self.description}\n\n"
            f"## Your Persona is :\n {self.persona}\n\n" if self.persona else ""
        )

        system_message = [{"role": "system", "content": instructions}]

        # self._logger.log(level="info", message=f"instructions:\n{instructions}",color="bold_green")

        original_query = query

        if use_memory and self.memory and (memorys_str := self.memory.get_memorys_str(query = original_query,enhanced_filter=True)):
            query =  f"### Your Recent Memory:\n```{memorys_str}```\n\n" + query

        if use_planner and self.planner and (plan_str := self.planner.get_day_plan_str()):
            query = f"### Your Today's Plan:\n```{plan_str}```\n\n" + query

        messages = system_message + [{"role": "user", "content": query}]

        tools = self.tools_schema if self.tools_schema and use_tools else None
        response = self.model_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=None,
                        temperature=self.temperature,
                    )
        
        response_message = response.choices[0].message

        # If there are no tool calls, return the message [Most Common Case]
        if not response_message.tool_calls:
            if keep_memory and self.memory:
                self.memory.add_working_memory(json.dumps({
                "query": original_query,
                "response": response_message.content
            }))
            res = [Message(sender=self.name, action="talk", result=response_message.content)]
            return res
        
        res = []

        for tool_call in response_message.tool_calls:
            tool = self.tools_map[tool_call.function.name]
            tool_args = json.loads(tool_call.function.arguments)
            self._logger.log(level="info", message=f"Tool Call [{tool_call.function.name}] with arguments: {tool_args} by {self.name}",color="bold_green")
            tool_result = tool(**tool_args)
            self._logger.log(level="info", message=f"Tool Call [{tool_call.function.name}] Result Received",color="bold_green")
            tool_call_result = (
                f"By using the tool '{tool_call.function.name}' with the arguments {tool_args}, "
                f"the result is '{tool_result}'."
            )
            messages.append({"role": "assistant", "content": tool_call_result})

        self._logger.log(level="info", message=f"All Tool Calls Completed, Process All Tool Call Results",color="bold_green")
        messages.append({"role": "user", "content": "Based on the results from the tools, respond to my previous question."})
        response = self.model_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=None,
                tool_choice=None,
                temperature=0.0,
            )
        
        response_message = response.choices[0].message
        if keep_memory and self.memory:
            self.memory.add_working_memory(json.dumps({
                "query": original_query,
                "response": response_message.content
            }))
        res.append(Message(sender=self.name, action="talk", result=response_message.content))

        return res

    def _call_dify_http_agent(self, token: str, query: str) -> List[Message]:
        """
        Calls the Dify.ai API to get a response.

        Args:
            token (str): The agent's access token for Dify.ai API.
            query (str): The query to send to the agent.

        Returns:
            List[Message]: List containing a Message object with the agent's response.

        Raises:
            ValueError: If token is empty or invalid
            requests.exceptions.RequestException: If HTTP request fails
        """
        if not token or not isinstance(token, str):
            raise ValueError("Invalid Dify access token")

        url = 'https://api.dify.ai/v1/chat-messages'
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        data = {
            "inputs": {},
            "query": query.strip(),  # Clean the query
            "response_mode": "blocking",
            "conversation_id": "",
            "user": self.name,
            "files": []
        }

        try:
            self._logger.log(
                level="info", 
                message=f"Sending request to Dify API", 
                color="bold_blue"
            )
            
            response = requests.post(
                url=url, 
                headers=headers, 
                json=data,
                timeout=30  # Add timeout
            )
            
            response.raise_for_status()  # Raise exception for bad status codes
            response_data = response.json()

            if 'answer' not in response_data:
                self._logger.log(
                    level="error",
                    message="Unexpected response format from Dify API",
                    color="bold_red"
                )
                return []

            self._logger.log(
                level="info",
                message="Successfully received response from Dify API",
                color="bold_green"
            )

            return [Message(
                sender=self.name,
                action="talk",
                result=response_data['answer']
            )]

        except requests.exceptions.Timeout:
            self._logger.log(
                level="error",
                message="Request timed out while connecting to Dify API",
                color="bold_red"
            )
            return []

        except requests.exceptions.RequestException as e:
            self._logger.log(
                level="error",
                message=f"Failed to communicate with Dify API: {str(e)}",
                color="bold_red"
            )
            return []

        except json.JSONDecodeError:
            self._logger.log(
                level="error",
                message="Failed to parse Dify API response as JSON",
                color="bold_red"
            )
            return []

    def _call_iagent_http_agent(self, query: str, sender: str = None) -> List[Message]:
        """
        Calls the iAgent HTTP endpoint to get a response.

        Args:
            query (str): The query to send to the agent.
            sender (str, optional): The sender of the message. Defaults to None.

        Returns:
            List[Message]: List of Message objects containing the agent's responses.
        
        Raises:
            ValueError: If iagent_url is not set
            requests.exceptions.RequestException: If HTTP request fails
        """
        if not self.iagent_url:
            raise ValueError("iAgent URL is not configured")

        if not sender:
            sender = "user"  # Default sender name

        headers = {
            'User-Agent': 'Multi-Agent-System/1.0',
            'Content-Type': 'application/json',
            'Accept': '*/*'
        }

        payload = {
            'sender_nickname': sender,
            'content': query.strip()  # Remove leading/trailing whitespace
        }

        try:
            self._logger.log(
                level="info", 
                message=f"Sending request to iAgent: {self.iagent_url}", 
                color="bold_blue"
            )
            
            response = requests.post(
                url=self.iagent_url, 
                headers=headers, 
                json=payload,
                timeout=30  # Add timeout
            )
            
            response.raise_for_status()
            response_data = response.json()

            if not isinstance(response_data, list):
                self._logger.log(
                    level="warning",
                    message="Unexpected response format - expected list",
                    color="yellow"
                )
                return []

            messages = []
            for item in response_data:
                # Validate response item structure
                if not isinstance(item, dict):
                    continue

                content_type = item.get('content_type')
                content = item.get('content')
                nickname = item.get('sender_nickname', self.name)

                if content_type == '1' and content:  # TEXT type
                    messages.append(Message(
                        sender=nickname,
                        action="talk",
                        result=content
                    ))
                else:
                    self._logger.log(
                        level="debug",
                        message=f"Skipping unsupported content type: {content_type}",
                        color="yellow"
                    )

            return messages

        except requests.exceptions.Timeout:
            self._logger.log(
                level="error",
                message="Request timed out while connecting to iAgent",
                color="bold_red"
            )
            return []

        except requests.exceptions.RequestException as e:
            self._logger.log(
                level="error",
                message=f"Failed to communicate with iAgent: {str(e)}",
                color="bold_red"
            )
            return []

        except json.JSONDecodeError:
            self._logger.log(
                level="error",
                message="Failed to parse iAgent response as JSON",
                color="bold_red"
            )
            return []

    def _call_websocket_agent(self, query: str) -> List[Message]:
        """
        This function calls the agent function to get the response.

        Args:
            query (str): The query to send to the agent.

        Returns:
            Message: The response from the agent.
        """
        if not self.ws:
            self._logger.log(level="error", message="Websocket connection is not established", color="bold_red")
            return []

        message = {"content": query}

        try:
            self.ws.send(json.dumps(message))
            response = self.ws.recv()
            res = [Message(sender=self.name, action="talk", result=response)]
            return res

        except Exception as e:
            self._logger.log(level="error", message=f"Error during websocket communication: {e}", color="bold_red")
            return []

    def _connect_to_websocket(self):
        """
        Connects to the websocket server.
        """
        if self.websocket_url:
            try:
                self.ws = websockets.sync.client.connect(self.websocket_url)
                self._logger.log(level="info", message="Websocket connection established", color="bold_green")
            except (websockets.exceptions.InvalidURI, websockets.exceptions.InvalidHandshake, OSError) as e:
                self._logger.log(level="error", message=f"Failed to connect to websocket: {e}", color="bold_red")
                self.ws = None
        else:
            self.ws = None

    def _process_tools(self) -> None:
        """
        Processes the tools added to the agent.
        """
        if not self.tools:
            return
        for tool in self.tools:
            self._add_tool(tool)

    def _add_tool(self, tool: "function") -> None:
        """ 
        Adds a tool to the current agent.
        """
        tool_schema = function_to_schema(tool)
        self.tools_schema.append(tool_schema)
        self.tools_map[tool.__name__] = tool
