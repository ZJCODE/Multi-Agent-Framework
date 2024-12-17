
from utilities.logger import Logger
from openai import OpenAI
from protocol import Env
from pydantic import BaseModel
from typing import List,Literal

class GroupPlanner:
    def __init__(self, env: Env,model_client: OpenAI,verbose: bool = False):
        self.env = env
        self.model_client = model_client
        self.plan = []
        self._logger = Logger(verbose=verbose)

    def set_task(self,task:str):
        """
        Set the task for the group.

        Args:
            task (str): The task for the group.
        """
        self.task = task

    def planning(self,model:str="gpt-4o-mini"):
        """
        Plan the task and assign sub-tasks to the members.

        Args:
            model (str): The model to use for planning.
        """
        self._logger.log("info","Start planning the task")

        member_list = ",".join([f'"{m.name}"' for m in self.env.members]) # for pydantic Literal

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

        planner_prompt = (
        "As an experienced planner with strong analytical and organizational skills, your role is to analyze tasks and delegate sub-tasks to group members." 
        "Ensure efficient completion by considering task order, member capabilities, and resource allocation." 
        "Communicate clearly and adapt to changing circumstances." 
        "Each task should include the agent's name, the task description, and a list of agents from whom they need to receive information (this list can be empty)."
        )

        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Task for Planning\n"
            f"```\n{self.task}\n```\n\n"
            f"### Strategy\n"
            f"First, evaluate team members' skills and availability to form a balanced group, ensuring a mix of competencies and expertise."
            f"Then, break the main task into prioritized sub-tasks and assign them based on expertise"
        )

        if self.env.language is not None:
            prompt += f"\n\n### Response in Language: {self.env.language}\n"

        messages = [{"role": "system", "content": planner_prompt}]

        messages.extend([{"role": "user", "content": prompt}])
        
        completion = self.model_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format=response_format,
        )
        
        self.plan = completion.choices[0].message.parsed.tasks
        self._logger.log("info","Planning finished")

        tasks_str = "\n\n".join([f"Step {i+1}: {t.agent_name}\n{t.task}\nreceive information from: {t.receive_information_from}\n" for i,t in enumerate(self.plan)])
        
        self._logger.log("info",f"Task: {self.task}\n\nPlan:\n{tasks_str}",color="bold_blue")


    def revise_plan(self,model:str="gpt-4o-mini"):

        if self.plan is None:
            raise ValueError("No plan to revise, please plan the task first by calling the planning method.")

        self._logger.log("info","Start revising the plan")

        members_description = "\n".join([f"- {m.name} ({m.role})" + (f" [tools available: {', '.join([x.__name__ for x in m.tools])}]" if m.tools else "") for m in self.env.members])

        self._logger.log("info","Get feedback from the members")

        feedback_prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Task for Planning\n"
            f"```\n{self.task}\n```\n\n"
            f"### Initial Plan\n"
            f"```\n{self.plan}\n```\n\n"
            f"Please review the initial plan and offer constructive feedback, "
            f"highlighting any improvements or adjustments that could enhance the project's success, "
            f"response in a concise and clear sentence."
        )

        if self.env.language is not None:
            feedback_prompt += f"\n\n### Response in Language: {self.env.language}\n"

        feedbacks = []
        for member in self.env.members:
            response = member.do(feedback_prompt,model)
            for r in response:
                feedback_str = f"Feedback from {member.name}: {r.result}"
                self._logger.log("info",feedback_str,color="bold_blue")
                feedbacks.append(r)
        
        feedbacks_str = "\n".join([f"{f.sender}: {f.result}" for f in feedbacks])

        member_list = ",".join([f'"{m.name}"' for m in self.env.members]) # for pydantic Literal

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

        planner_prompt = (
        "As an experienced planner with strong analytical and organizational skills, your role is to analyze tasks and delegate sub-tasks to group members." 
        "Ensure efficient completion by considering task order, member capabilities, and resource allocation." 
        "Communicate clearly and adapt to changing circumstances." 
        "Each task should include the agent's name, the task description, and a list of agents from whom they need to receive information (this list can be empty)."
        )

        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Task for Planning\n"
            f"```\n{self.task}\n```\n\n"
            f"### Initial Plan\n"
            f"```\n{self.plan}\n```\n\n"
            f"### Feedbacks\n"
            f"{feedbacks_str}\n\n"
            f"Please revise the plan by addressing the feedback provided. Ensure that all concerns are considered,"
            f"and make necessary adjustments to improve the plan's effectiveness and feasibility. "
        )

        if self.env.language is not None:
            prompt += f"\n\n### Response in Language: {self.env.language}\n"

        messages = [{"role": "system", "content": planner_prompt}]
        messages.extend([{"role": "user", "content": prompt}])
        
        completion = self.model_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format=response_format,
        )
        self.plan = completion.choices[0].message.parsed.tasks
        self._logger.log("info","Revising the plan finished, replacing the initial plan with the revised plan")

        tasks_str = "\n\n".join([f"Step {i+1}: {t.agent_name}\n{t.task}\nreceive information from: {t.receive_information_from}\n" for i,t in enumerate(self.plan)])

        self._logger.log("info",f"Task: {self.task}\n\nRevised Plan:\n{tasks_str}",color="bold_blue")


    def in_transit_revisions(self,current_task,current_response:str,model:str="gpt-4o-mini"):

        self._logger.log("info",f"Decide weather to assign extra tasks for {current_task.agent_name}")

        class ExtraTasks(BaseModel):
            add_extra_tasks:bool
            tasks:List[str]

        current_agent = next((m for m in self.env.members if m.name == current_task.agent_name), None)

        current_agent_description = f"- {current_agent.name} ({current_agent.role})" + (f" [tools available: {', '.join([x.__name__ for x in current_agent.tools])}]" if current_agent.tools else "")

        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Task Overview\n"
            f"```\n{self.task}\n```\n\n"
            f"### Proposed Task Plan\n"
            f"```\n{self.plan}\n```\n\n"
            f"### Current Agent Profile\n"
            f"{current_agent_description}\n\n"
            f"### Current Task Details\n"
            f"```\n{current_task.task}\n```\n\n"
            f"### Current Response\n"
            f"```\n{current_response}\n```\n\n"
            f"### Inquiry\n"
            f"Given the information above, do you think we should add any additional tasks for current agent? Yes/No for add extra tasks."
            f"Make sure to consider the agent's skills, availability, and the project's requirements."
            f"If you choose to add extra tasks,Please provide a clear and concise response, the task description should include sufficient details to let the agent do the task standalone."
        )

        if self.env.language is not None:
            prompt += f"\n\n### Response in Language: {self.env.language}\n"

        planner_assistant_prompt = (
            "As a planner assistant, you play a crucial role in supporting the planning process by providing valuable insights and suggestions."
            "Your feedback can help optimize task allocation and improve overall project efficiency."
            "Review the current task, agent response, and existing plan, then decide whether additional tasks are needed."
        )
        
        messages = [{"role": "system", "content": planner_assistant_prompt}]
        messages.extend([{"role": "user", "content": prompt}])

        completion = self.model_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0.0,
            response_format=ExtraTasks,
        )

        if completion.choices[0].message.parsed.add_extra_tasks:
            self._logger.log("info","Extra tasks are needed, adding extra tasks to the plan")
            extra_tasks = completion.choices[0].message.parsed.tasks
            return extra_tasks
        else:
            self._logger.log("info","No extra tasks needed")
            return None
