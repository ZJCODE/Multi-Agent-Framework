from typing import Literal
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI,AsyncOpenAI
from dotenv import load_dotenv
import uuid
import asyncio

load_dotenv()

class Agent:
    """
    simple demo
    """
    def __init__(self,name: str,description: str,
                 base_url: str = None,api_key: str = None,
                 model: str = "gpt-4o-mini"):
        self.name = name
        self.description = description
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: list|str,stream: bool = False):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.description}] + messages,
                stream=stream
            )
        
        if not stream:
            return [{"role": "assistant", "content": response.choices[0].message.content,"sender": self.name}]
        else:
            return response

    async def chat_async(self, messages: list|str,stream: bool = False):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.description}] + messages,
                stream=stream
            )
        
        if not stream:
            return [{"role": "assistant", "content": response.choices[0].message.content,"sender": self.name}]
        else:
            return response
        
@dataclass
class AgentSchema:
    """
    This class defines the schema of the agent used for the handoff process.
    """
    name: str
    transfer_to_me_description: str
    agent: Agent
    relations: Optional[List[str]] = None # agent names that this agent can transfer to
    as_entry: bool = False  # Default to False
    as_exit: bool = False   # Default to False


class Group:
    def __init__(self, 
                 participants: list[AgentSchema], 
                 model_client: OpenAI = None,
                 base_url: str = None,api_key: str = None,
                 model: str = "gpt-4o-mini"):
        """ 
        Initializes the group with the given participants and model client.

        Args:
            participants (list[AgentSchema]): A list of AgentSchema objects representing the participants in the group.
            model_client (OpenAI, optional): The OpenAI client used for the handoff process. Defaults to None.
            base_url (str, optional): The base URL of the OpenAI API. Defaults to None.
            api_key (str, optional): The API key used for the OpenAI client. Defaults to None.
            model (str, optional): The model used for the handoff process. Defaults to "gpt-4o-mini".

        Raises:
            ValueError: If the group structure is not valid.
        """
        self.participants = participants
        self.model_client = AsyncOpenAI(api_key=api_key, base_url=base_url) if model_client is None else model_client
        self.model = model
        self.entry_agent = next((p for p in participants if p.as_entry), random.choice(participants))
        self.current_agent = {"DEFAULT":self.entry_agent}
        self.exit_agent = next((p for p in participants if p.as_exit), None)
        self.agent_map = {p.name: p for p in participants}
        self.handoff_tools = {"DEFAULT":[]}
        self.participants_order_map = self._build_participant_order_map()
        self.group_structure = self._decide_group_structure()

    def reset(self,
              thread_id:Optional[str] = None):
        """
        Resets the state of the framework by setting the current agent to the first participant
        that has the 'as_entry' attribute set to True.
        """
        self.entry_agent = next((p for p in self.participants if p.as_entry), random.choice(self.participants))
        if thread_id:
            self.current_agent[thread_id] = self.entry_agent
            self.handoff_tools[thread_id] = []
        else:
            self.current_agent = {"DEFAULT":self.entry_agent}
            self.handoff_tools = {"DEFAULT":[]}

    async def handoff_one_turn(self, 
                         messages: list|str,
                         model:str="gpt-4o-mini",
                         next_speaker_select_mode:Literal["order","auto","random"]="auto",
                         include_current:bool = True,
                         thread_id: str = "DEFAULT",
                         verbose=False
                         ):
        """ 
        Performs a single turn of the handoff process.

        Args:
            messages (list|str): The messages to be used for the handoff process.
            model (str, optional): The model used for the handoff process. Defaults to "gpt-4o-mini".
            next_speaker_select_mode (Literal["order","auto","random"], optional): The mode used to select the next speaker. Defaults to "auto".
            include_current (bool, optional): Whether to include the current agent in the handoff tools. Defaults to True.
            thread_id (str, optional): The thread ID used to keep track of the current agent. Defaults to "DEFAULT".
            verbose (bool, optional): Whether to print the handoff process. Defaults to False.
        """

        if next_speaker_select_mode == "order":
            if self.group_structure != "CONNECTED":
                raise ValueError("next_speaker_select_mode 'order' is only supported when group_structure is 'CONNECTED'")
            next_agent = self.participants_order_map[self.current_agent.get(thread_id,self.entry_agent).name]
            if verbose:
                current_agent = self.current_agent.get(thread_id,self.entry_agent).name
                print(f"\n-> handoff from {current_agent} to {next_agent} (order mode)")
            self.current_agent[thread_id] = self.agent_map[next_agent]
            return next_agent
        
        elif next_speaker_select_mode == "random":
            next_agent = random.choice(self.agent_names)
            if verbose:
                current_agent = self.current_agent.get(thread_id,self.entry_agent).name
                print(f"\n-> handoff from {current_agent} to {next_agent} (random mode)")
            self.current_agent[thread_id] = self.agent_map[next_agent]
            return next_agent
 
        elif next_speaker_select_mode == "auto":
            
            self._build_current_handoff_tools(include_current=include_current,thread_id=thread_id)
    
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            messages = [{"role": "system", "content":"deciding which agent to transfer to"}] + messages

            response = await self.model_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=self.handoff_tools.get(thread_id,[]),
                        tool_choice="required"
                    )
            next_agent = response.choices[0].message.tool_calls[0].function.name

            if next_agent in self.agent_names:
                if self.current_agent.get(thread_id,self.entry_agent).name != next_agent and verbose:
                    print("\n-> handoff from {} to {} (auto mode)".format(self.current_agent.get(thread_id,self.entry_agent).name, next_agent))
                self.current_agent[thread_id] = self.agent_map[next_agent]
            else:
                raise ValueError(f"Handoff to unknown agent: {next_agent}")

            return next_agent
    
        else:
            raise ValueError(f"Unknown next_speaker_select_mode: {next_speaker_select_mode} , Currently only 'order', 'random' and 'auto' are supported")

    async def handoff(self, 
                messages: list|str,
                model:str="gpt-4o-mini",
                handoff_max_turns:int=10,
                next_speaker_select_mode:Literal["order","auto","random"]="auto",
                include_current = True,
                thread_id: str = "DEFAULT",
                verbose=False
                ):

        """ 
        Performs the handoff process.

        Args:
            messages (list|str): The messages to be used for the handoff process.
            model (str, optional): The model used for the handoff process. Defaults to "gpt-4o-mini".
            handoff_max_turns (int, optional): The maximum number of turns to perform the handoff process. used in auto mode. Defaults to 10.
            next_speaker_select_mode (Literal["order","auto","random"], optional): The mode used to select the next speaker. Defaults to "auto".
            include_current (bool, optional): Whether to include the current agent in the handoff tools. Defaults to True.
            thread_id (str, optional): The thread ID used to keep track of the current agent. Defaults to "DEFAULT".
            verbose (bool, optional): Whether to print the handoff process. Defaults to False.
        """

        next_agent = await self.handoff_one_turn(messages,model,next_speaker_select_mode,include_current,thread_id,verbose)
        if next_speaker_select_mode != "auto" or handoff_max_turns == 1:
            return next_agent
        next_next_agent = await self.handoff_one_turn(messages,model,"auto",include_current,thread_id,verbose)
        while next_next_agent != next_agent and handoff_max_turns > 1:
            next_agent = next_next_agent
            next_next_agent = await self.handoff_one_turn(messages,model,"auto",include_current,thread_id,verbose)
            handoff_max_turns -= 1
        return next_agent

            
    @property
    def agent_names(self):
        """
        Returns:
            list: A list of names of all participants.
        """
        return [p.name for p in self.participants]
    
    @property
    def relations(self):
        """
        Returns:
            list: A list of tuples representing the relations between participants.
                Each tuple contains the name of a participant and the name of a related participant.
        """
        relations = [("START", self.entry_agent.name)]
        if self.exit_agent:
            relations.append((self.exit_agent.name, "END"))
        if all(not hasattr(p, 'relations') or p.relations is None for p in self.participants):
            for i in range(len(self.participants)):
                for j in range(len(self.participants)):
                    if i != j:
                        relations.append((self.participants[i].name, self.participants[j].name))
        else:
            for p in self.participants:
                if hasattr(p, 'relations') and isinstance(p.relations, list):
                    relations.extend((p.name, r) for r in p.relations)
        return relations
    
    @property
    def relation_agents(self):
        """
        Returns:
            dict: A dictionary where the keys are participant names and the values are lists of AgentSchema objects
                representing the related agents.
        """
        if self.group_structure == "CUSTOM" or self.group_structure == "SEQUENCE":
            return {p.name: [self.agent_map[r] for r in p.relations] if p.relations else [] for p in self.participants}
        elif self.group_structure == "CONNECTED":
            return {p.name: [a for a in self.participants if a.name != p.name] for p in self.participants}
    

    def _build_current_handoff_tools(self, include_current=True,thread_id: str = "DEFAULT"):
        """ 
        Builds the handoff tools based on the current agent and its related agents.

        Args:
            include_current (bool, optional): Whether to include the current agent in the handoff tools. Defaults to True.

        Returns:
            list: A list of handoff tools.
        """
        self.handoff_tools[thread_id] = [self._build_agent_schema(self.current_agent.get(thread_id,self.entry_agent))] if include_current else []
        self.handoff_tools.get(thread_id).extend(self._build_agent_schema(r) for r in self.relation_agents[self.current_agent.get(thread_id,self.entry_agent).name])

    @staticmethod
    def _build_agent_schema(agent: AgentSchema):
        """
        Builds the schema for the given agent. 
        """
        return {
            "type": "function",
            "function": {
                "name": agent.name,
                "description": agent.transfer_to_me_description,
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    
    def _build_participant_order_map(self):
        """
        Builds a map of participants in order.

        Returns:
            dict: A dictionary mapping participant names to the next participant in order.
        """
        return {p.name: self.participants[(i+1) % len(self.participants)].name for i, p in enumerate(self.participants)}
    

    def _decide_group_structure(self):
        """ 
        Decides the group structure based on the participants and their relations.

        Returns:
            str: The group structure.

        Raises:
            ValueError: If the group structure is not valid.
        """
        # if all participants do not have relations, then group_structure is CONNECTED
        if all(not hasattr(p, 'relations') or p.relations is None for p in self.participants):
            return "CONNECTED"
        # if any participant has relations, and entry and exit agents are defined, and no circular relations, then group_structure is SEQUENCE
        if any(hasattr(p, 'relations') and p.relations for p in self.participants) and self.entry_agent and self.exit_agent:
            if self._is_valid_sequence(self.relations):
                return "SEQUENCE"
            else:
                print("[Warning] You may want to construct a SEQUENCE but now this is an invalid SEQUENCE, A SEUQENCE must have both entry and exit agents and no circular relations, Setting group_structure to CUSTOM temporarily")
        return "CUSTOM"
    
 
    @staticmethod
    def _is_valid_sequence(relations):
        """ 
        Checks if the given relations form a valid sequence.

        Args:
            relations (list): A list of tuples representing the relations between participants.

        Returns:
            bool: True if the relations form a valid sequence, False otherwise.
        """
        # Build the graph
        graph = defaultdict(list)
        reverse_graph = defaultdict(list)
        nodes = set()
        for u, v in relations:
            graph[u].append(v)
            reverse_graph[v].append(u)
            nodes.update([u, v])

        # Check for cycles using DFS
        def has_cycle(v, visited, rec_stack):
            visited.add(v)
            rec_stack.add(v)
            for neighbor in graph[v]:
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(v)
            return False

        visited = set()
        rec_stack = set()
        for node in list(graph.keys()):  # Iterate over a list of the dictionary keys
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    print("[SEQUENCE INVALID] Cycle detected")
                    return False

        # Check if all nodes are reachable from 'START'
        def bfs_reachable_from_start(start):
            queue = deque([start])
            visited = set()
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    queue.extend(graph[node])
            return visited

        # Check if all nodes can reach 'END'
        def bfs_reachable_to_end(end):
            queue = deque([end])
            visited = set()
            while queue:
                node = queue.popleft()
                if node not in visited:
                    visited.add(node)
                    queue.extend(reverse_graph[node])
            return visited

        reachable_from_start = bfs_reachable_from_start('START')
        reachable_to_end = bfs_reachable_to_end('END')

        if nodes != reachable_from_start:
            missing_nodes = nodes - reachable_from_start
            print(f"[SEQUENCE INVALID] Nodes not reachable from START: {missing_nodes}")
            return False

        if nodes != reachable_to_end:
            missing_nodes = nodes - reachable_to_end
            print(f"[SEQUENCE INVALID] Nodes not reachable to END: {missing_nodes}")
            return False

        return True
    
    @staticmethod
    def _generate_thread_id():
        """ 
        Generates a random thread ID by using uuid
        """
        return str(uuid.uuid4())