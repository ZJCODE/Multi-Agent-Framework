from typing import List,Union,Dict
from openai import OpenAI,AsyncOpenAI
import requests
import json
from .utilities.logger import Logger
from .utilities.utils import function_to_schema
from .protocol import Member,Message

class Agent(Member):
    def __init__(
            self, 
            name: str, 
            role: str, 
            description: str = None,
            persona: str = None, # context and personality of the agent more detailed than description
            model_client: Union[OpenAI, AsyncOpenAI] = None,
            tools: List["function"] = None, # List of Python Functions
            dify_access_token: str = None,
            verbose: bool = False
            ):
        """
        Initializes the Agent class.

        Args:
            name (str): The name of the agent.
            role (str): The role of the agent.
            description (str, optional): The simple description of the agent. Defaults to None.
            persona (str, optional): The persona of the agent can add more personality,backsotry and skills to the agent. Defaults to None.
            model_client (Union[OpenAI, AsyncOpenAI], optional): The model client for the agent. Defaults to None.
            tools (List["function"], optional): The tools for the agent. Defaults to None.
            dify_access_token (str, optional): The Dify access token for the agent. Defaults to None.
            verbose (bool, optional): The verbosity of the agent. Defaults to False.
        
        """
        super().__init__(name, role, description)
        self._logger = Logger(verbose=verbose)
        self.persona = persona
        self.model_client = model_client
        self.tools = tools
        self.dify_access_token = dify_access_token
        self.verbose = verbose
        # Tools Related Attributes
        self.tools_schema: List[Dict] = []
        self.tools_map: Dict[str, "function"] = {}
        self._process_tools()
        
    def __str__(self):
        return f"{self.name} is a {self.role}."
    
    def do(self, message: str,model:str="gpt-4o-mini",use_tools:bool=True) -> List[Message]:
        if self.dify_access_token:
            self._logger.log(level="info", message=f"Calling Dify agent [{self.name}]",color="bold_green")
            response = self._call_dify_http_agent(self.dify_access_token, message)
        elif isinstance(self.model_client,OpenAI):
            self._logger.log(level="info", message=f"Calling OpenAI agent [{self.name}]",color="bold_green")
            response = self._call_openai_agent(message,model,use_tools)
        else:
            self._logger.log(level="error", message=f"No model client or Dify access token provided for agent {self.name}.",color="red")
            raise ValueError("No model client or Dify access token provided, please provide one for agent {self.name}.")
        return response

    def _call_openai_agent(self,query:str,model:str="gpt-4o-mini",use_tools:bool=True) -> List[Message]:
        """
        This function calls the agent function to get the response.

        Args:
            query (str): The query to send to the agent.

        Returns:
            Message: The response from the agent.
        """

        instructions =(
            f"## Role:\n {self.role}\n\n"
            f"## Description:\n {self.description}\n\n"
        )
        if self.persona:
            instructions += f"## Persona:\n {self.persona}\n\n"
        system_message = [{"role": "system", "content": instructions}]
        messages = system_message + [{"role": "user", "content": query}]

        tools = self.tools_schema if self.tools_schema and use_tools else None
        response = self.model_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=None,
                    )
        
        response_message = response.choices[0].message

        # If there are no tool calls, return the message [Most Common Case]
        if not response_message.tool_calls:
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
            )
        
        response_message = response.choices[0].message
        res.append(Message(sender=self.name, action="talk", result=response_message.content))

        return res

    def _call_dify_http_agent(self,token:str,query:str) -> List[Message]:
        """
        This function calls the agent function to get the response.

        Args:
            token (str): The agent's access token.
            query (str): The query to send to the agent.

        Returns:
            Message: The response from the agent.
        """
        url = 'https://api.dify.ai/v1/chat-messages'
        headers = {
            'Authorization': 'Bearer {}'.format(token),
            'Content-Type': 'application/json'
        }

        data = {
            "inputs": {},
            "query": query,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": self.name,
            "files": []
        }

        response = requests.post(url, headers=headers, json=data)

        res = [Message(sender=self.name, action="talk", result=response.json()['answer'])]

        return res  
    

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
