import graphviz
from openai import OpenAI
import random
import yaml
import uuid
import itertools
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
from typing import Dict, Optional, Literal, Tuple,List,Union
from utilities.logger import Logger

from protocol import Member, Env, Message, GroupMessageProtocol

class Group:
    def __init__(
        self, 
        env: Env,
        model_client: OpenAI,
        group_id: Optional[str] = None,
        verbose: bool = False
    ):
        self._logger = Logger(verbose=verbose)
        self.fully_connected = False # will be updated in _rectify_relationships
        self.group_id:str = group_id if group_id else str(uuid.uuid4()) # unique group
        self.env: Env = env
        self.model_client: OpenAI = model_client # currently only supports OpenAI synthetic API
        self.current_agent: Optional[str] = self.env.members[0].name # default current agent is the first agent in the members list
        self.members_map: Dict[str, Member] = {m.name: m for m in self.env.members}
        self.member_iterator = itertools.cycle(self.env.members)
        self._rectify_relationships()
        self._set_env_public()
        self._update_response_format_maps()
        self.group_messages: GroupMessageProtocol = GroupMessageProtocol(group_id=self.group_id,env=self.env_public)
        
    def set_current_agent(self, agent_name:str):
        if agent_name not in self.members_map:
            raise ValueError(f"Member with name {agent_name} does not exist")
        self.current_agent = agent_name
        self._logger.log("info",f"manually set the current agent to {agent_name}")


    def add_member(self, member: Member,relation:Optional[Tuple[str,str]] = None):
        if member.name in self.members_map:
            raise ValueError(f"Member with name {member.name} already exists")
        self.env.members.append(member)
        self.members_map[member.name] = member
        self.member_iterator = itertools.cycle(self.env.members)
        self._rectify_relationships()
        self._add_relationship(member,relation)
        self._set_env_public()
        self._update_response_format_maps()
        self.group_messages.env = self.env_public
        self._logger.log("info",f"Succesfully add member {member.name}")


    def delete_member(self, member_name:str):
        # if current agent is the one to be deleted, handoff to the next agent by order
        if member_name not in self.members_map:
            raise ValueError(f"Member with name {member_name} does not exist")
        self.env.members = [m for m in self.env.members if m.name != member_name]
        self.members_map.pop(member_name)
        self.member_iterator = itertools.cycle(self.env.members)
        self._rectify_relationships()
        self._remove_relationships(member_name)
        self._set_env_public()
        self._update_response_format_maps()
        self.group_messages.env = self.env_public

        if self.current_agent == member_name:
            self.current_agent = random.choice([m.name for m in self.env.members]) if self.env.members else None
            self._logger.log("info",f"current agent {member_name} is deleted, randomly select {self.current_agent} as the new current agent")
        self._logger.log("info",f"Successfully delete member {member_name}")

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def handoff(
            self,
            handoff_max_turns:int=3,
            next_speaker_select_mode:Literal["order","auto","auto2","random"]="auto",
            model:str="gpt-4o-mini",
            include_current:bool = True
    )->str:
        
        visited_agent = set([self.current_agent])
        next_agent = self.handoff_one_turn(next_speaker_select_mode, model, include_current)

        if self.current_agent != next_agent:
            self._logger.log("info",f"handoff from {self.current_agent} to {next_agent} by using {next_speaker_select_mode} mode")
        else:
            self._logger.log("info",f"no handoff needed, stay with {self.current_agent} judge by {next_speaker_select_mode} mode")
            
        if self.fully_connected or next_speaker_select_mode in ["order","random"] or handoff_max_turns == 1:
            self.current_agent = next_agent
            return self.current_agent
        # recursive handoff until the next agent is same as the current agent (for auto and auto2 with handoff_max_turns > 1)
        visited_agent.add(next_agent)
        next_next_agent =  self.handoff_one_turn(next_speaker_select_mode,model,True)
        while next_next_agent != next_agent and handoff_max_turns > 1:
            self._logger.log("info",f"handoff from {next_agent} to {next_next_agent} by using {next_speaker_select_mode} mode")
            if next_next_agent in visited_agent:
                break 
            next_agent = next_next_agent
            self.current_agent = next_agent
            next_next_agent = self.handoff_one_turn(next_speaker_select_mode,model,True)
            handoff_max_turns -= 1

        self.current_agent = next_agent

        return self.current_agent

    def handoff_one_turn(
            self,
            next_speaker_select_mode: Literal["order", "auto", "auto2", "random"] = "auto",
            model: str = "gpt-4o-mini",
            include_current: bool = True
    ) -> str:
        if next_speaker_select_mode == "order":
            return next(self.member_iterator).name
        elif next_speaker_select_mode == "random":
            return random.choice([m.name for m in self.env.members])
        elif next_speaker_select_mode in ("auto", "auto2"):
            if not self.env.relationships[self.current_agent]:
                return self.current_agent
            return self._select_next_agent_auto(model, include_current, 
                                                use_tool = next_speaker_select_mode == "auto")
        else:
            raise ValueError("next_speaker_select_mode should be one of 'order', 'auto', 'auto2', 'random'")

    def update_group_messages(self, message:Union[Message,List[Message]]):
        if isinstance(message,Message):
            self.group_messages.context.append(message)
        elif isinstance(message,list):
            self.group_messages.context.extend(message)
        else:
            raise ValueError("message should be either Message or List[Message]")

    def reset_group_messages(self):
        # prsisit the whole group_messages context to the storage(local file or database) then reset the context
        self.group_messages.context = []

    def call_agent(
            self,
            next_speaker_select_mode:Literal["order","auto","auto2","random"]="auto",
            include_current:bool = True,
            model:str="gpt-4o-mini",
            message_cut_off:int=3,
            agent:str = None # can mauanlly set the agent to call
    ) -> List[Message]:
        if agent:
            self.set_current_agent(agent)
        else:
            self.handoff(next_speaker_select_mode=next_speaker_select_mode,model=model,include_current=include_current)
        message_send = self._build_send_message(self.group_messages,cut_off=message_cut_off,send_to=self.current_agent)
        response = self.members_map[self.current_agent].do(message_send,model)
        self.update_group_messages(response)
        self._logger.log("info",f"Call agent {self.current_agent}",color="bold_green")
        for r in response:
            self._logger.log("info",f"Agent {self.current_agent} response: {r.result}",color="bold_purple")
        return response

    def task(
            self,
            task:str,
            strategy:Literal["sequential","hierarchical"] = "sequential",
            model:str="gpt-4o-mini",
            entry_agent: Optional[str] = None,
        ):
        self.reset_group_messages()
        if strategy == "sequential":
            return self._task_sequential(task,model,entry_agent)
        elif strategy == "hierarchical":
            return self._task_hierarchical(task,model)
        else:
            raise ValueError("strategy should be one of 'sequential' or 'hierarchical'")
        
    def draw_relations(self):
        """ 
        Returns:
            bytes: A PNG image of the graph representing the relations between the agents.
        """
        dot = graphviz.Digraph(format='png')
        for member in self.env.members:
            color = 'orange' if member.name == self.current_agent else 'black'
            label = f"{member.name}\n{member.role}"
            dot.node(member.name, label, color=color)
        for m1, m2 in self.env.relationships.items():
            for m in m2:
                dot.edge(m1, m)
        return dot.pipe()
    
    def _rectify_relationships(self):
        """
        Rectify the relationships between the agents.
        """
        if self.env.relationships is None or self.fully_connected:
            self._logger.log("info","All agents are fully connected")
            self.fully_connected = True
            self.env.relationships = {m.name: [n.name for n in self.env.members if n.name != m.name] for m in self.env.members}
        elif isinstance(self.env.relationships, list):
            self._logger.log("info","Self-defined relationships,covnert relationships from list to dictionary")
            relationships = {m.name: [] for m in self.env.members}
            for m1, m2 in self.env.relationships:
                relationships[m1].append(m2)
                relationships[m2].append(m1)
        else:
            self._logger.log("info","Self-defined relationships")
            for m in self.env.members:
                if m.name not in self.env.relationships:
                    self.env.relationships[m.name] = []

    def _add_relationship(self,member:Member,relation:Optional[Tuple[str,str]] = None):
        """
        Add a relationship for the new member.

        Args:
            member (Member): The member to add the relationship for.
            relation (Optional[Tuple[str, str]]): The relationship tuple. Defaults to None.
        """
        if not self.fully_connected and relation is not None:
            for r in relation:
                if r[0] not in self.env.relationships:
                    raise ValueError(f"Member with name {r[0]} does not exist")
                if member.name not in r:
                    continue
                self.env.relationships[r[0]].append(r[1])

    def _remove_relationships(self, member_name: str):
        """
        Remove relationships for the deleted member.

        Args:
            member_name (str): The name of the member to remove relationships for.
        """
        if not self.fully_connected:
            self.env.relationships.pop(member_name, None)
            for k, v in self.env.relationships.items():
                if member_name in v:
                    v.remove(member_name)


    def _update_response_format_maps(self):
        """
        Update the next choice response format maps.
        """
        self.next_choice_response_format_map: Dict[str, BaseModel] = self._build_next_choice_response_format_map(False)
        self.next_choice_response_format_map_include_current: Dict[str, BaseModel] = self._build_next_choice_response_format_map(True)


    def _set_env_public(self):
        self.env_public = Env(
            description=self.env.description,
            members=[Member(name=m.name, role=m.role, description=m.description) for m in self.env.members],
            relationships=self.env.relationships
        )

    def _build_current_agent_handoff_tools(self, include_current_agent:bool = False):
        handoff_tools = [self._build_agent_handoff_tool_function(self.members_map[self.current_agent])] if include_current_agent else []
        handoff_tools.extend(self._build_agent_handoff_tool_function(self.members_map[agent]) for agent in self.env.relationships[self.current_agent])
        return handoff_tools

    def _build_next_choice_response_format_map(self,include_current:bool = False):
        """
        
        """
        response_format_map = {}
        if include_current:
            for k,v in self.env.relationships.items():
                agent_name_list = ",".join([f'"{i}"' for i in v+[k]])
                class_str = (
                    f"class SelectFor{k}IncludeCurrent(BaseModel):\n"
                    f"    agent_name:Literal[{agent_name_list}]\n"
                )
                exec(class_str)
                response_format_map.update({k:eval(f"SelectFor{k}IncludeCurrent")})
        else:
            for k,v in self.env.relationships.items():
                if len(v) == 0:
                    continue
                agent_name_list = ",".join([f'"{i}"' for i in v])
                class_str = (
                    f"class SelectFor{k}ExcludeCurrent(BaseModel):\n"
                    f"    agent_name:Literal[{agent_name_list}]\n"
                )
                exec(class_str)
                response_format_map.update({k:eval(f"SelectFor{k}ExcludeCurrent")})
        return response_format_map

    def _select_next_agent_auto(self, model: str, include_current: bool, use_tool: bool) -> str:

        messages = [{"role": "system", "content": "Decide who should be the next person to talk. Transfer the conversation to the next person."}]
        handoff_message = self._build_handoff_message(self.group_messages, cut_off=1, use_tool=use_tool) # notice the cut_off setting
        messages.extend([{"role": "user", "content": handoff_message}])

        # if use_tool is True, the agent will be selected based on the tool call [auto]
        if use_tool:
            handoff_tools = self._build_current_agent_handoff_tools(include_current)
            response = self.model_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                tools=handoff_tools,
                tool_choice="required"
            )
            return response.choices[0].message.tool_calls[0].function.name
        # if use_tool is False, the agent will be selected based on the response format [auto2]
        else:
            response_format = self.next_choice_response_format_map_include_current[self.current_agent] if include_current else self.next_choice_response_format_map[self.current_agent]
            completion = self.model_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=response_format,
                max_tokens=10
            )
            return completion.choices[0].message.parsed.agent_name

    def _task_sequential(self,task:str,model:str="gpt-4o-mini",entry_agent: str = None):
        if entry_agent is None:
            raise ValueError("Entry agent is not defined, sequential task need to define the entry agent")
        step = 0
        self.update_group_messages(Message(sender="user",action="task",result=task))
        self._logger.log("info",f"Start task: {task}")
        self.current_agent = entry_agent
        while step < len(self.env.members):
            self._logger.log("info",f"===> Step {step + 1}")
            self.call_agent(next_speaker_select_mode="order",model=model,include_current=False,message_cut_off=None)
            step += 1
        self._logger.log("info","Task finished")
        return self.group_messages

    @staticmethod
    def _build_agent_handoff_tool_function(agent: Member):
        """
        Builds the schema for the given agent. 
        """
        return {
            "type": "function",
            "function": {
                "name": agent.name,
                "description": f"{agent.description} ({agent.role})",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }

    @staticmethod
    def _build_handoff_message(gmp:GroupMessageProtocol,cut_off:int=1,use_tool:bool=False):
        """
        This function builds a prompt for llm to decide who should be the next person been handoff to.

        Args:
            gmp (GroupMessageProtocol): The group message protocol Instance.
            cut_off (int): The number of previous messages to consider.

        Returns:
            str: The prompt for the agent to decide who should be the next person been handoff to.
        """

        messages = "\n\n".join([f"```{m.sender}\n {m.result}\n```" for m in gmp.context[-cut_off:]])

        if use_tool:
            prompt = (
                f"### Background Information\n"
                f"{gmp.env.description}\n\n"
                f"### Messages\n"
                f"{messages}\n\n"
            )
        else:
            members_description = "\n\n".join([f"```{m.name}\n({m.role}):{m.description}\n```" for m in gmp.env.members])
            prompt = (
                    f"### Background Information\n"
                    f"{gmp.env.description}\n\n"
                    f"### Members\n"
                    f"{members_description}\n\n"
                    f"### Messages\n"
                    f"{messages}\n\n"
                    f"### Task\n"
                    f"Consider the Background Information and the previous messages. "
                    f"Decide who should be the next person to send a message. Choose from the members."
                )
            
        return prompt     

    @staticmethod
    def _build_send_message(gmp:GroupMessageProtocol,cut_off:int=None,send_to:str=None) -> str:
        """ 
        This function builds a prompt for the agent to send a message in the group message protocol.

        Args:
            gmp (GroupMessageProtocol): The group message protocol Instance.
            cut_off (int): The number of previous messages to consider.
            send_to (str): The agent to send the message

        Returns:
            str: The prompt for the agent to send a message.
        """
        
        members_description = "\n".join([f"- {m.name} ({m.role})" for m in gmp.env.members])

        if cut_off is None:
            previous_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in gmp.context if m.sender == send_to])
            others_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in gmp.context if m.sender != send_to])
        else:
            previous_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in gmp.context[-cut_off:] if m.sender == send_to])
            others_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in gmp.context[-cut_off:] if m.sender != send_to])

        prompt = (
            f"### Background Information\n"
            f"{gmp.env.description}\n\n"
            f"### Members\n"
            f"{members_description}\n\n"
            f"### Your Previous Message\n"
            f"{previous_messages}\n\n"
            f"### Other people's Messages\n"
            f"{others_messages}\n\n"
            f"### Task\n"
            f"Consider the Background Information and the previous messages. Now, it's your turn."
        )

        return prompt
