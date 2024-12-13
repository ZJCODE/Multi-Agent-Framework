import graphviz
from openai import OpenAI
import random
import uuid
import itertools
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Optional, Literal, Tuple,List,Union
from utilities.logger import Logger

from protocol import Member, Env, Message, GroupMessageProtocol
from agent import Agent

class Group:
    def __init__(
        self, 
        env: Env,
        model_client: OpenAI,
        group_id: Optional[str] = None,
        verbose: bool = False,
        manager:Union[Agent,bool] = None # if manager is True, the group will have a default manager or you can pass a Agent instance
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
        self._create_manager(manager)

    def set_current_agent(self, agent_name: str):
        """
        Set the current agent by name if the agent exists in the members map.

        Args:
            agent_name (str): The name of the agent to set as current.

        Raises:
            ValueError: If the agent name does not exist in the members map.
        """
        if agent_name not in self.members_map:
            self._logger.log("error", f"Attempted to set non-existent member {agent_name} as current agent")
            raise ValueError(f"Member with name {agent_name} does not exist")

        self.current_agent = agent_name
        self._logger.log("info", f"Manually set the current agent to {agent_name}")


    def add_member(self, member: Member,relation:Optional[Tuple[str,str]] = None):
        """
        Add a new member to the group.

        Args:
            member (Member): The member to add to the group.
            relation (Optional[Tuple[str, str]]): The relationship tuple. Defaults to None.
        """
        if member.name in self.members_map:
            self._logger.log("warning",f"Member with name {member.name} already exists",color="red")
            return
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
        """
        Delete a member from the group.

        Args:
            member_name (str): The name of the member to delete.
        """
        if member_name not in self.members_map:
            self._logger.log("warning",f"Member with name {member_name} does not exist",color="red")
            return
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
            next_speaker_select_mode:Literal["order","auto","auto2","random"]="auto2",
            model:str="gpt-4o-mini",
            include_current:bool = True
    )->str:
        """
        Handoff the conversation to the next agent.

        Args:
            handoff_max_turns (int): The maximum number of turns to handoff. Defaults to 3.
            next_speaker_select_mode (Literal["order","auto","auto2","random"]): The mode to select the next speaker. Defaults to "auto2".
            model (str): The model to use for the handoff. Defaults to "gpt-4o-mini".
            include_current (bool): Whether to include the current agent in the handoff. Defaults to True.
        """
        if self.fully_connected or next_speaker_select_mode in ["order","random"]:
            handoff_max_turns = 1

        visited_agent = set([self.current_agent])
        next_agent = self.handoff_one_turn(next_speaker_select_mode, model, include_current)

        while next_agent != self.current_agent and handoff_max_turns > 0:
            if next_agent in visited_agent:
                break 
            self._logger.log("info",f"handoff from {self.current_agent} to {next_agent} by using {next_speaker_select_mode} mode")
            self.current_agent = next_agent
            visited_agent.add(next_agent)
            next_agent = self.handoff_one_turn(next_speaker_select_mode,model,True)
            handoff_max_turns -= 1

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
        """
        Reset the group messages.
        """
        self.group_messages.context = []

    def user_input(self, message:str,action:str="talk",alias = None):
        """
        Record the user input.
        """
        self.update_group_messages(Message(sender="user",action=action,result=message))
        if alias:
            self._logger.log("info",f"[{alias}] input ({action}): {message}",color="bold_blue")
        else:
            self._logger.log("info",f"User input ({action}): {message}",color="bold_blue")

    def call_agent(
            self,
            next_speaker_select_mode:Literal["order","auto","auto2","random"]="auto2",
            include_current:bool = True,
            model:str="gpt-4o-mini",
            message_cut_off:int=3,
            agent:str = None # can mauanlly set the agent to call
    ) -> List[Message]:
        """
        Call the agent to respond to the group messages.

        Args:
            next_speaker_select_mode (Literal["order","auto","auto2","random"]): The mode to select the next speaker. Defaults to "auto2".
            include_current (bool): Whether to include the current agent in the handoff. Defaults to True.
            model (str): The model to use for the handoff. Defaults to "gpt-4o-mini".
            message_cut_off (int): The number of previous messages to consider. Defaults to 3.
            agent (str): Specify the agent to call. Defaults to None meaning the agent will be selected based on the next_speaker_select_mode.
        """
        if agent:
            self.set_current_agent(agent)
        else:
            self.handoff(next_speaker_select_mode=next_speaker_select_mode,model=model,include_current=include_current)
        message_send = self._build_send_message(cut_off=message_cut_off,send_to=self.current_agent)
        response = self.members_map[self.current_agent].do(message_send,model)
        self.update_group_messages(response)
        for r in response:
            self._logger.log("info",f"Agent {self.current_agent} response:\n\n{r.result}",color="bold_purple")
        return response

    def call_manager(self,model:str="gpt-4o-mini",message_cut_off:int=3) -> List[Message]:
        """
        Call the manager to respond to the group messages.
        """
        if not self.manager:
            self._logger.log("warning","No manager in the group , you can set the manager in Group initialization",color="red")
            return
        message_send = self._build_send_message(cut_off=message_cut_off,send_to=self.manager.name)
        response = self.manager.do(message_send,model)
        self.update_group_messages(response)
        self._logger.log("info",f"Call manager {self.manager.name}",color="bold_green")
        for r in response:
            self._logger.log("info",f"Manager {self.manager.name} response: {r.result}",color="bold_purple")

        return response


    def talk(
            self, 
            message:str,
            model:str="gpt-4o-mini",
            message_cut_off:int=3,
            agent:str = None # can mauanlly set the agent to call
        )-> List[Message]:
        """
        Talk to the agent.

        Args:
            message (str): The message to send to the agent.
            model (str): The model to use for the handoff. Defaults to "gpt-4o-mini".
            message_cut_off (int): The number of previous messages to consider. Defaults to 3.
            agent (str): Specify the agent to call. Defaults to None meaning the agent will be selected based on the next_speaker_select_mode.
        """
        self.user_input(message)
        response = self.call_agent("auto2",include_current=True,model=model,message_cut_off=message_cut_off,agent=agent)
        return response

    def task(
            self,
            task:str,
            strategy:Literal["sequential","hierarchical","auto"] = "auto",
            model:str="gpt-4o-mini"
        ) -> List[Message]:
        """
        Execute a task with the given strategy.

        Args:
            task (str): The task to execute.
            strategy (Literal["sequential","hierarchical","auto"], optional): The strategy to use for the task. Defaults to "auto".
            model (str, optional): The model to use for the task. Defaults to "gpt-4o-mini".

        Returns:
            List[Message]: The response

        More details about the strategy:
            - sequential: The task will be executed sequentially by each agent in the group.
            - hierarchical: To be implemented.
            - auto: The task will be executed automatically by the agents in the group based on the planning.
        """
        self.reset_group_messages()
        if strategy == "sequential":
            return self._task_sequential(task,model)
        elif strategy == "hierarchical":
            return self._task_hierarchical(task,model)
        elif strategy == "auto":
            return self._task_auto(task,model)
        else:
            raise ValueError("strategy should be one of 'sequential' or 'hierarchical' or 'auto'")
        
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
        handoff_message = self._build_handoff_message(cut_off=1, use_tool=use_tool) # notice the cut_off setting
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

    def _task_sequential(self,task:str,model:str="gpt-4o-mini"):
        self.user_input(task,action="task")
        step = 0
        self._logger.log("info",f"Start task: {task}")
        for member in self.env.members:
            step += 1
            self._logger.log("info",f"===> Step {step} for {member.name}")
            response = self.call_agent(agent=member.name,model=model,include_current=False,message_cut_off=None)
        self._logger.log("info","Task finished")
        return response

    def _task_auto(self,task:str,model:str="gpt-4o-mini"):
        tasks = self._planning(task,model)

        tasks_str = "\n".join([f"{t.json()}" for t in tasks])
        self._logger.log("info",f"Initial plan is \n\n{tasks_str}",color="bold_blue")

        tasks = self._revise_plan(task,tasks,model=model)
        tasks_str = "\n".join([f"{t.json()}" for t in tasks])
        self._logger.log("info",f"Revised plan is \n\n{tasks_str}",color="bold_blue")

        step = 0
        self._logger.log("info",f"Start task: {task}")
        for t in tasks:
            step += 1
            self._logger.log("info",f"===> Step {step} for {t.agent_name} \n\ndo task: {t.task} \n\nreceive information from: {t.receive_information_from}")
            self.set_current_agent(t.agent_name)
            message_send = self._build_auto_task_message(t,cut_off=2,model=model)
            response = self.members_map[t.agent_name].do(message_send,model)
            self.update_group_messages(response)
            for r in response:
                self._logger.log("info",f"Agent {self.current_agent} response:\n\n{r.result}",color="bold_purple")
        self._logger.log("info","Task finished")
        return response
    
    def _build_auto_task_message(self,task,cut_off:int=None,model:str="gpt-4o-mini"):
        if cut_off < 1:
            cut_off = None
        agent_name = task.agent_name
        task_deatil = task.task
        receive_information_from = task.receive_information_from

        members_description = "\n".join([f"- {m.name} ({m.role})" for m in self.env.members])

        previous_messages = [message for message in self.group_messages.context if message.sender == agent_name]
        if cut_off is not None:
            previous_messages = previous_messages[-cut_off:]

        receive_informations = []
        receive_sender_couter = {}
        for message in self.group_messages.context:
            if message.sender in receive_information_from and (receive_sender_couter.get(message.sender,0) < cut_off or cut_off is None):
                if message.sender not in receive_sender_couter:
                    receive_sender_couter[message.sender] = 0
                receive_sender_couter[message.sender] += 1
                receive_informations.append(message)

        previous_messages_str = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in previous_messages])
        receive_informations_str = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in receive_informations])

        promote = (
            f"### Background Information\n"
            f"{self.env.description}\n\n"
            f"### Members\n"
            f"{members_description}\n\n"
            f"### Your Previous Message\n"
            f"{previous_messages_str}\n\n"
            f"### Received Information\n"
            f"{receive_informations_str}\n\n"
            f"### Task\n"
            f"```\n{task_deatil}\n```\n\n"
            f"Please respond to the task."
        )

        self._logger.log("info",f"Auto task message for {agent_name}:\n\n{promote}",color="bold_blue")

        return promote

    def _planning(self,task:str,model:str="gpt-4o-mini"):
        """
        Plan the task and assign sub-tasks to the members.

        Args:
            task (str): The task to plan.
            model (str): The model to use for planning.
        """
        self._logger.log("info","Start planning the task")
        member_list = ",".join([f'"{m.name}"' for m in self.env.members])

        class_str = (
            f"class Task(BaseModel):\n"
            f"    agent_name:Literal[{member_list}]\n"
            f"    task:str\n"
            f"    receive_information_from:List[Literal[{member_list}]]\n"
            f""
            f"class Tasks(BaseModel):\n"
            f"    tasks:List[Task]\n"
        )
        
        exec(class_str, globals())

        response_format = eval("Tasks")

        members_description = "\n".join([f"- {m.name} ({m.role})" + (f" [tools available: {', '.join([x.__name__ for x in m.tools])}]" if m.tools else "") for m in self.env.members])

        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Task for Planning\n"
            f"```\n{task}\n```\n\n"
            f"### Strategy\n"
            f"Step One: Choose a group of members to delegate the task.\n"
            f"Step Two: Create a list of sub-tasks derived from the main task and allocate them to the chosen members in best order.\n"
        )

        messages = [{"role": "system", "content": "You are a experienced planner."}]
        messages.extend([{"role": "user", "content": prompt}])
        
        completion = self.model_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format=response_format,
        )
        self._logger.log("info","Planning finished")
        return completion.choices[0].message.parsed.tasks

    def _revise_plan(self,task:str,init_paln:list,model:str="gpt-4o-mini"):
        
        self._logger.log("info","Start revising the plan")

        member_list = ",".join([f'"{m.name}"' for m in self.env.members])

        members_description = "\n".join([f"- {m.name} ({m.role})" + (f" [tools available: {', '.join([x.__name__ for x in m.tools])}]" if m.tools else "") for m in self.env.members])

        self._logger.log("info","Get feedback from the members")

        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Task for Planning\n"
            f"```\n{task}\n```\n\n"
            f"### Initial Plan\n"
            f"```\n{init_paln}\n```\n\n"
            f"Please provide your feedback on the initial plan in a concise and clear sentence."
        )

        feedbacks = []
        for member in self.env.members:
            response = self.members_map[member.name].do(prompt,model)
            for r in response:
                feedbacks.append(r)
        
        feedbacks_str = "\n".join([f"{f.sender}: {f.result}" for f in feedbacks])

        self._logger.log("info",f"Feedbacks from the members:\n\n{feedbacks_str}",color="bold_blue")

        class_str = (
            f"class Task(BaseModel):\n"
            f"    agent_name:Literal[{member_list}]\n"
            f"    task:str\n"
            f"    receive_information_from:List[Literal[{member_list}]]\n"
            f""
            f"class Tasks(BaseModel):\n"
            f"    tasks:List[Task]\n"
        )
        
        exec(class_str, globals())

        response_format = eval("Tasks")

        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Task for Planning\n"
            f"```\n{task}\n```\n\n"
            f"### Initial Plan\n"
            f"```\n{init_paln}\n```\n\n"
            f"### Feedbacks\n"
            f"{feedbacks_str}\n\n"
            f"Please revise the plan based on the feedbacks."
        )

        messages = [{"role": "system", "content": "You are a experienced planner."}]
        messages.extend([{"role": "user", "content": prompt}])
        
        completion = self.model_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format=response_format,
        )
        self._logger.log("info","Revising the plan finished")
        return completion.choices[0].message.parsed.tasks

    def _create_manager(self,manager:Union[Agent,bool]):
        # planner and moderator
        if manager == True:
            self.manager = Agent(name=f"GroupManager-{self.group_id}",
                                 role="Manager",
                                 description="The manager of the group",
                                 backstory="The manager is responsible for planning and moderating the group conversation",
                                 model_client=self.model_client,
                                 verbose=self._logger.verbose)
            self._logger.log("info","Create a default manager for the group")
        elif isinstance(manager,Agent):
            self.manager = manager
        else:
            self.manager = None

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

    
    def _build_handoff_message(self,cut_off:int=1,use_tool:bool=False):
        """
        This function builds a prompt for llm to decide who should be the next person been handoff to.

        Args:
            cut_off (int): The number of previous messages to consider.

        Returns:
            str: The prompt for the agent to decide who should be the next person been handoff to.
        """

        messages = "\n\n".join([f"```{m.sender}\n {m.result}\n```" for m in self.context[-cut_off:]])

        if use_tool:
            prompt = (
                f"### Background Information\n"
                f"{self.env.description}\n\n"
                f"### Messages\n"
                f"{messages}\n\n"
            )
        else:
            members_description = "\n\n".join([f"```{m.name}\n({m.role}):{m.description}\n```" for m in self.env.members])
            prompt = (
                    f"### Background Information\n"
                    f"{self.env.description}\n\n"
                    f"### Members\n"
                    f"{members_description}\n\n"
                    f"### Messages\n"
                    f"{messages}\n\n"
                    f"### Task\n"
                    f"Consider the Background Information and the previous messages. "
                    f"Decide who should be the next person to send a message. Choose from the members."
                )
            
        return prompt     

    def _build_send_message(self,cut_off:int=None,send_to:str=None) -> str:
        """ 
        This function builds a prompt for the agent to send a message in the group message protocol.

        Args:
            cut_off (int): The number of previous messages to consider.
            send_to (str): The agent to send the message

        Returns:
            str: The prompt for the agent to send a message.
        """
        
        members_description = "\n".join([f"- {m.name} ({m.role})" for m in self.env.members])


        if cut_off is None:
            previous_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in self.context if m.sender == send_to])
            others_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in self.context if m.sender != send_to])
        else:
            previous_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in self.context[-cut_off:] if m.sender == send_to])
            others_messages = "\n\n".join([f"```{m.sender}:{m.action}\n{m.result}\n```" for m in self.context[-cut_off:] if m.sender != send_to])

        prompt = (
            f"### Background Information\n"
            f"{self.env.description}\n\n"
            f"### Members\n"
            f"{members_description}\n\n"
            f"### Your Previous Message\n"
            f"{previous_messages}\n\n"
            f"### Other people's Messages\n"
            f"{others_messages}\n\n"
            f"### Task\n"
            f"Consider the Background Information and the previous messages. Now, it's your turn."
        )

        if self.context[-1].sender == "user":
            if self.context[-1].action == "task":
                current_user_task = self.context[-1].result
                prompt += f"\n\n### Current Task\n{current_user_task}\n\n"
            elif self.context[-1].action == "talk":
                current_user_message = self.context[-1].result
                prompt += f"\n\n### Current User's Input\n{current_user_message}\n\n"

        return prompt
