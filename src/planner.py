
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

        self.planner_prompt = (
            "As an experienced planner with strong analytical and organizational skills, your role is to analyze tasks and delegate sub-tasks to group members.\n"
            "1. Analyze Tasks:\n"
            "   - Break down the main task into smaller, manageable sub-tasks.\n"
            "   - Identify dependencies and determine the order in which tasks need to be completed.\n"
            "2. Assess Member Capabilities:\n"
            "   - Evaluate the skills and strengths of each team member to match them with appropriate tasks.\n"
            "3. Delegate Sub-Tasks:\n"
            "   - Assign each sub-task to the most suitable team member.\n"
            "   - Ensure that each task includes:\n"
            "     - The agent's name.\n"
            "     - A clear task description.\n"
            "     - A list of agents from whom they need to receive information (if applicable).\n"
            "4. Consider Task Order:\n"
            "   - Arrange tasks in a logical sequence, considering dependencies and the flow of information.\n"
        )

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



        prompt = (
            f"### Contextual Information\n"
            f"{self.env.description}\n\n"
            f"### Potential Members\n"
            f"{members_description}\n\n"
            f"### Members' Relationship\n"
            f"{self.env.relationships}\n\n"
            f"### Task for Planning\n"
            f"```\n{self.task}\n```\n\n"
            f"### Strategy\n"
            f"First, evaluate team members' skills and availability to form a balanced group, ensuring a mix of competencies and expertise."
            f"Then, break the main task into prioritized sub-tasks and assign them based on expertise"
        )

        if self.env.language is not None:
            prompt += f"\n\n### Response in Language: {self.env.language}\n"

        messages = [{"role": "system", "content": self.planner_prompt}]

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
            f"### Members' Relationship\n"
            f"{self.env.relationships}\n\n"
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

        messages = [{"role": "system", "content": self.planner_prompt}]
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

        self._logger.log("info",f"Decide weather to assign extra tasks before next task in the plan")


        members_description = "\n".join([f"- {m.name} ({m.role})" + (f" [tools available: {', '.join([x.__name__ for x in m.tools])}]" if m.tools else "") for m in self.env.members])

        member_list = f'"{current_task.agent_name}"' # for pydantic Literal

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
            f"### Task Overview\n"
            f"```\n{self.task}\n```\n\n"
            f"### Proposed Task Plan\n"
            f"```\n{self.plan}\n```\n\n"
            f"### Current Task Details\n"
            f"```\n{current_task.task}\n```\n\n"
            f"### Current Response\n"
            f"```\n{current_response}\n```\n\n"
            f"### Decision Point\n"
            f"Based on the information above, decide whether to assign extra tasks to current agent before proceeding with the next task in the plan. "
            f"Consider:\n"
            f"1. Does the current response fully meet the requirements of the current task? Identify any gaps or incomplete aspects.\n"
            f"2. Are there any unresolved dependencies or prerequisites that must be addressed before moving forward?\n"
            f"3. Is there a need for further clarification or information before proceeding to the next task? Determine if additional details are required.\n"
            f"Assign extra tasks only when necessary based on the evaluation above.\n"
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
            response_format=response_format,
        )
        
        extra_task = completion.choices[0].message.parsed.tasks
    
        if extra_task:

            tasks_str = "\n\n".join([f"Step {i+1}: {t.agent_name}\n{t.task}\nreceive information from: {t.receive_information_from}\n" for i,t in enumerate(extra_task)])

            self._logger.log("info",f"Extra Task:\n{tasks_str}",color="bold_blue")

        return extra_task