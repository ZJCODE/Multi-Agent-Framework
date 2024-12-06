import graphviz
from openai import OpenAI
import random
import yaml
import uuid
import itertools
from pydantic import BaseModel
import requests
from typing import Dict, Optional, Literal, Union, Tuple
from enum import Enum

from protocol import Member, Env, Message, GroupMessageProtocol

class SpeakerSelectMode(Enum):
    ORDER = "order"
    RANDOM = "random"
    AUTO = "auto"
    AUTO2 = "auto2"

class Group:
    def __init__(
        self, 
        env: Union[Env, str],
        model_client: OpenAI,
        group_id: str = None,
        entry_agent: Optional[str] = None
    ):
        self.fully_connected = False # will be updated in _rectify_relationships
        self.group_id:str = group_id if group_id else str(uuid.uuid4())
        self.env: Env = self._read_env_from_file(env) if isinstance(env, str) else env
        self.model_client: OpenAI = model_client # currently only supports OpenAI synthetic API
        self.current_agent: Optional[str] = entry_agent if entry_agent else random.choice([m.name for m in self.env.members])
        self.members_map: Dict[str, Member] = {m.name: m for m in self.env.members}
        self.member_iterator = itertools.cycle(self.env.members)
        self._rectify_relationships()
        self._set_env_public()
        self.next_choice_base_model_map: Dict[str, BaseModel] = self._build_next_choice_base_model_map(False)
        self.next_choice_base_model_map_include_current: Dict[str, BaseModel] = self._build_next_choice_base_model_map(True)
        self.group_messages: GroupMessageProtocol = GroupMessageProtocol(group_id=self.group_id,env=self.env_public)

    def add_member(self, member: Member,relation:Optional[Tuple[str,str]] = None):
        if member.name in self.members_map:
            raise ValueError(f"Member with name {member.name} already exists")
        self.env.members.append(member)
        self.members_map[member.name] = member
        self.member_iterator = itertools.cycle(self.env.members)
        self._rectify_relationships()
        if not self.fully_connected and relation is not None:
            for r in relation:
                if r[0] not in self.env.relationships:
                    raise ValueError(f"Member with name {r[0]} does not exist")
                if member.name not in r:
                    continue
                self.env.relationships[r[0]].append(r[1])
        self._set_env_public()
        self.next_choice_base_model_map = self._build_next_choice_base_model_map(False)
        self.next_choice_base_model_map_include_current = self._build_next_choice_base_model_map(True)
        self.group_messages.env = self.env_public

    def delete_member(self, member_name:str):
        # if current agent is the one to be deleted, handoff to the next agent by order
        if member_name not in self.members_map:
            raise ValueError(f"Member with name {member_name} does not exist")
        self.env.members = [m for m in self.env.members if m.name != member_name]
        self.members_map.pop(member_name)
        self.member_iterator = itertools.cycle(self.env.members)
        self._rectify_relationships()
        if not self.fully_connected:
            self.env.relationships.pop(member_name)
            for k,v in self.env.relationships.items():
                if member_name in v:
                    v.remove(member_name)
        self._set_env_public()
        self.next_choice_base_model_map = self._build_next_choice_base_model_map(False)
        self.next_choice_base_model_map_include_current = self._build_next_choice_base_model_map(True)
        self.group_messages.env = self.env_public

        if self.current_agent == member_name:
            self.current_agent = random.choice([m.name for m in self.env.members]) if self.env.members else None

    def handoff(
            self,
            handoff_max_turns:int=3,
            next_speaker_select_mode:Literal["order","auto","auto2","random"]="auto",
            model:str="gpt-4o-mini",
            include_current:bool = True,
            vobose:bool = False
    ):
        visited_agent = set([self.current_agent])
        next_agent = self.handoff_one_turn(next_speaker_select_mode, model, include_current)
        if vobose:
            print(f"handoff from {self.current_agent} to {next_agent}")
        if self.fully_connected or next_speaker_select_mode in ["order","random"] or handoff_max_turns == 1:
            self.current_agent = next_agent
            self.group_messages.next_agent = next_agent
            return self.current_agent
        # recursive handoff until the next agent is same as the current agent (for auto and auto2 with handoff_max_turns > 1)
        visited_agent.add(next_agent)
        next_next_agent =  self.handoff_one_turn(next_speaker_select_mode,model,True)
        while next_next_agent != next_agent and handoff_max_turns > 1:
            if vobose:
                print(f"handoff from {self.current_agent} to {next_next_agent}")
            if next_next_agent in visited_agent:
                break 
            next_agent = next_next_agent
            self.current_agent = next_agent
            next_next_agent = self.handoff_one_turn(next_speaker_select_mode,model,True)
            handoff_max_turns -= 1
        
        self.group_messages.next_agent = next_agent
        self.current_agent = next_agent

        return self.current_agent

    def handoff_one_turn(
            self,
            next_speaker_select_mode: Literal["order", "auto", "auto2", "random"] = "auto",
            model: str = "gpt-4o-mini",
            include_current: bool = True
    ) -> str:
        if next_speaker_select_mode == SpeakerSelectMode.ORDER.value:
            return next(self.member_iterator).name
        elif next_speaker_select_mode == SpeakerSelectMode.RANDOM.value:
            return random.choice([m.name for m in self.env.members])
        elif next_speaker_select_mode in (SpeakerSelectMode.AUTO.value, SpeakerSelectMode.AUTO2.value):
            if not self.env.relationships[self.current_agent]:
                return self.current_agent
            return self._select_next_agent_auto(model, include_current, 
                                                use_tool = next_speaker_select_mode == SpeakerSelectMode.AUTO.value)
        else:
            raise ValueError("next_speaker_select_mode should be one of 'order', 'auto', 'auto2', 'random'")

    def _select_next_agent_auto(self, model: str, include_current: bool, use_tool: bool) -> str:

        messages = [{"role": "system", "content": "Decide who should be the next person to talk. Transfer the conversation to the next person."}]
        handoff_message = self._build_handoff_message(self.group_messages, cut_off=1, use_tool=use_tool) # notice the cut_off setting
        messages.extend([{"role": "user", "content": handoff_message}])

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
        else:
            response_format = self.next_choice_base_model_map_include_current[self.current_agent] if include_current else self.next_choice_base_model_map[self.current_agent]
            completion = self.model_client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                temperature=0.0,
                response_format=response_format,
                max_tokens=10
            )
            return completion.choices[0].message.parsed.agent_name
    
    def update_group_messages(self, message:Message):
        self.group_messages.context.append(message)

    def call_agent(
            self,
            next_speaker_select_mode:Literal["order","auto","auto2","random"]="auto",
            include_current:bool = True,
            model:str="gpt-4o-mini",
            agent:str = None # can mauanlly set the agent to call
    ) -> Message:
        if agent is not None and agent in self.members_map:
            self.group_messages.next_agent = agent
            self.current_agent = agent
        else:
            self.handoff(next_speaker_select_mode=next_speaker_select_mode,model=model,include_current=include_current)
        message_send = self._build_send_message(self.group_messages,cut_off=3)
        response = self._call_agent_func(self.current_agent,self.members_map[self.current_agent].access_token,message_send,self.group_id)
        self.update_group_messages(response)
        return response

    def draw_relations(self):
        """ 
        Returns:
            bytes: A PNG image of the graph representing the relations between the agents.
        """
        dot = graphviz.Digraph(format='png')
        for member in self.env.members:
            dot.node(member.name, f"{member.name}\n{member.role}")
        for m1, m2 in self.env.relationships.items():
            for m in m2:
                dot.edge(m1, m)
        return dot.pipe()
    

    def _read_env_from_file(self, env_file:str):
        with open(env_file, 'r') as file:
            env_data = yaml.safe_load(file)
            env = Env(
                description=env_data['description'],
                members=[Member(**member) for member in env_data['members']],
                relationships=env_data.get('relationships', None)
            )
        return env

    def _rectify_relationships(self):
        """
        Rectify the relationships between the agents.
        """
        if self.env.relationships is None or self.fully_connected:
            self.fully_connected = True
            self.env.relationships = {m.name: [n.name for n in self.env.members if n.name != m.name] for m in self.env.members}
        elif isinstance(self.env.relationships, list):
            relationships = {m.name: [] for m in self.env.members}
            for m1, m2 in self.env.relationships:
                relationships[m1].append(m2)
                relationships[m2].append(m1)
        else:
            for m in self.env.members:
                if m.name not in self.env.relationships:
                    self.env.relationships[m.name] = []

    def _set_env_public(self):
        self.env_public = Env(
            description=self.env.description,
            members=[Member(name=m.name, role=m.role, description=m.description) for m in self.env.members],
            relationships=self.env.relationships
        )

    def _build_current_agent_handoff_tools(self, include_current_agent:bool = False):
        handoff_tools = [self._build_agent_call_function(self.members_map[self.current_agent])] if include_current_agent else []
        handoff_tools.extend(self._build_agent_call_function(self.members_map[agent]) for agent in self.env.relationships[self.current_agent])
        return handoff_tools


    def _build_next_choice_base_model_map(self,include_current:bool = False):
        base_model_map = {}
        if include_current:
            for k,v in self.env.relationships.items():
                class_str = """
class SelectFor{}IncludeCurrent(BaseModel):
    agent_name:Literal[{}]
""".format(k,",".join([f'"{i}"' for i in v+[k]]))
                exec(class_str)
                base_model_map.update({k:eval(f"SelectFor{k}IncludeCurrent")})
        else:
            for k,v in self.env.relationships.items():
                if len(v) == 0:
                    continue
                class_str = """
class SelectFor{}ExcludeCurrent(BaseModel):
    agent_name:Literal[{}]
    """.format(k,",".join([f'"{i}"' for i in v]))
                exec(class_str)
                base_model_map.update({k:eval(f"SelectFor{k}ExcludeCurrent")})
        return base_model_map
    
    @staticmethod
    def _build_agent_call_function(agent: Member):
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

        if use_tool:
            prompt = """### Background Information

{}

### Messages

{}""".format(
                gmp.env.description,
                    "\n\n".join([f"```{m.sender}\n {m.result}\n```" for m in gmp.context[-cut_off:]])
                )
        else:
            prompt = """### Background Information

{}

### Members

{}

### Messages

{}

### Task

Consider the Background Information and the previous messages. Decide who should be the next person to send a message,choose from members.""".format(
                gmp.env.description,
                "\n\n".join([f"```{m.name}\n({m.role}):{m.description}\n```" for m in gmp.env.members]),
                "\n\n".join([f"```{m.sender}\n{m.result}\n```" for m in gmp.context[-cut_off:]])
                )
        return prompt
    
    @staticmethod
    def _call_agent_func(agent:str,token:str,query:str,group_id:str = None):

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
            "user": group_id,
            "files": []
        }

        response = requests.post(url, headers=headers, json=data)

        message = Message(sender=agent, action="talk", result=response.json()['answer'])

        return message        

    @staticmethod
    def _build_send_message(gmp:GroupMessageProtocol,cut_off:int=3):
        """ 
        This function builds a prompt for the agent to send a message in the group message protocol.

        Args:
            gmp (GroupMessageProtocol): The group message protocol Instance.
            cut_off (int): The number of previous messages to consider.

        Returns:
            str: The prompt for the agent to send a message.
        """
        assert gmp.next_agent is not None, "next_agent should be set"
        assert cut_off > 0, "cut_off should be greater than 0"

        prompt = """### Background Information

{}

### Members

{}

### Your Previous Message

{}

### Other people's Messages

{}

### Task

Consider the Background Information and the previous messages. Now, it's your turn.""".format(
            gmp.env.description,
            "\n".join([f"- {m.name} ({m.role})" for m in gmp.env.members]),
            "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in gmp.context[-cut_off:] if m.sender == gmp.next_agent]),
            "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in gmp.context[-cut_off:] if m.sender != gmp.next_agent]),
        )
        return prompt
