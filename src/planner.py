from typing import List
from pydantic import BaseModel,Field

class Plan(BaseModel):
    start_hour: int
    end_hour: int
    plan: str 
class OneDayPlan(BaseModel):
    plans: List[Plan]
class MinutePlan(BaseModel):
    start_minute: int
    end_minute: int
    plan: str
class OneHourPlan(BaseModel):
    plans: List[MinutePlan]
class Planner:
    def __init__(self,
                 model_client=None,
                 model:str="gpt-4o-mini",
                 language:str=None,
                 verbose:bool=False):
        self.model_client = model_client
        self.model = model
        self.language = language
        self.verbose = verbose 
        self.task_plan = []
        self.daily_plan = []
        self.one_hour_plan = []

    def plan_task(self, env_info: str,personal_info: str,memory: str,task:str):
        pass

    def plan_day(self, env_info: str,personal_info: str,memory: str):
        """
        env_info can be the time location, weather, and events happening in the environment.
        personal_info can be the user's name, occupation, preferences, goals, and frequent locations.
        memory can be the user's recent activities, conversations, and events or observations.
        """
        system_message = "You are skilled at planning daily activities based on environmental information, personal information, and memory."
        prompt = (
            "### Environment Information:\n"
            f"```{env_info}```\n"
            "### Personal Information:\n"
            f"```{personal_info}```\n"
            "### Memory:\n"
            f"```{memory}```\n\n"
            "Based on the environment information, personal information, and memory, create a detailed daily plan for the next 24 hours. "
            "Ensure the plan:\n"
            "- Reflects personal preferences, goals, and relevant events from memory.\n"
            "- Includes a specific activity for each hour.\n"
            "Ensure the updated plan is practical, well-balanced, and aligned with the provided information and every hour is covered from 0 to 24."
            "Plan examples: \n"
            "0:00 - 7:00: Sleep\n"
            "7:00 - 8:00: Morning Routine\n"
            "8:00 - 9:00: Breakfast\n"
            "..."
            "14:00 - 16:00: Work on Project\n"
            "21:00 - 22:00: Reading\n"
            "22:00 - 23:00: Prepare for Bed\n"
            "plans for 24 hours: "
        )

        if self.language:
            prompt += f"\n\n### Response in Language: {self.language}"

        messages = [{"role":"system","content":system_message}]
        messages.append({"role":"user","content":prompt})

        completion = self.model_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format=OneDayPlan
        )

        one_day_plan = completion.choices[0].message.parsed

        self.daily_plan = one_day_plan.plans

    def plan_hour(self, env_info: str,personal_info: str,memory: str,current_hour: int):
        current_hour_plan = self.get_current_hour_plan(current_hour)
        # plan the activity for the current hour based on the environment information, personal information, and memory
        system_message = "You are skilled at planning the activity for the current hour based on environmental information, personal information, memory, and current hour plan. decompose the hour into minutes and plan the activity for each minute."

        prompt = (
            "### Environment Information:\n"
            f"```{env_info}```\n"
            "### Personal Information:\n"
            f"```{personal_info}```\n"
            "### Memory:\n"
            f"```{memory}```\n"
            "### Current Hour:\n"
            f"```{current_hour}```\n\n"
            "### Current Plan for the Hour:\n"
            f"```{current_hour_plan}```\n\n"
            "Instructions:\n"
            "Use the provided information to create a detailed plan for the current hour.\n"
            "Divide the hour into smaller time blocks (e.g., 0-10, 10-40, 40-60).\n"
            "Assign specific activities to each block, ensuring alignment with the current hour plan and context.\n"
            "Example Format:\n"
            "0-20: Prepare breakfast\n"
            "20-40: Eat breakfast\n"
            "40-60: Clean up\n"
            "plans for 60 minutes: "
        )

        if self.language:
            prompt += f"\n\n### Response in Language: {self.language}"

        messages = [{"role":"system","content":system_message}]
        messages.append({"role":"user","content":prompt})

        completion = self.model_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format=OneHourPlan
        )

        one_hour_plan = completion.choices[0].message.parsed

        self.one_hour_plan = one_hour_plan.plans


    def update_day_plan(self, env_info: str,personal_info: str,memory: str,current_hour: int,extra_info: str):
        # update the plan after the current hour based on the extra information
        system_message = "You are skilled at updating the daily plan based on environmental information, personal information, memory, and current hour."
        prompt = (
            "### Environment Information:\n"
            f"```{env_info}```\n"
            "### Personal Information:\n"
            f"```{personal_info}```\n"
            "### Current Plan:\n"
            f"```{self.daily_plan}```\n"
            "### Memory:\n"
            f"```{memory}```\n"
            "### Current Hour:\n"
            f"```{current_hour}```\n"
            "### Extra Information:\n"
            f"```{extra_info}```\n\n"
            "Update the daily plan based on the current hour and extra information. Ensure the updated plan:\n"
            "- Reflects personal preferences, goals, and relevant events from memory.\n"
            "- Reassigns the activity for the current hour and adjusts activities for the following hours.\n"
            "- Includes a specific activity for each hour.\n"
            "- Only update the plan when necessary."
            "Ensure the updated plan is practical, well-balanced, and aligned with the provided information."
            "plans for 24 hours: "
        )

        if self.language:
            prompt += f"\n\n### Response in Language: {self.language}"

        messages = [{"role":"system","content":system_message}]
        messages.append({"role":"user","content":prompt})

        completion = self.model_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format=OneDayPlan
        )

        one_day_plan = completion.choices[0].message.parsed

        self.daily_plan = one_day_plan.plans
        
    def get_daily_plan(self):
        return self.daily_plan
    
    def get_daily_future_plan(self, current_hour: int):
        future_plan = []
        for hour_plan in self.daily_plan:
            if current_hour <= hour_plan.end_hour:
                future_plan.append(hour_plan)
        return future_plan

    def get_current_hour_plan(self,current_hour:int):
        for hour_plan in self.daily_plan:
            if current_hour >= hour_plan.start_hour and current_hour < hour_plan.end_hour:
                return hour_plan.plan
        return "Sleep"

    def get_day_plan_str(self):
        plan_str = ""
        for plan in self.daily_plan:
            plan_str += f"{plan.start_hour} - {plan.end_hour} : {plan.plan}\n"
        return plan_str

if __name__ == "__main__":
    
    from dotenv import load_dotenv
    from openai import OpenAI

    # load the environment variables
    load_dotenv()
    # create a model client
    model_client = OpenAI()

    planner = Planner(model_client=model_client,model="gpt-4o",verbose=True)
    
    env_info = "It's 2100-01-03, the weather is sunny, and there are no major events scheduled."

    personal_info = "Your name is Alice. You hold the position of a manager. You prefer mornings and enjoy running at that time. In the afternoons, you like to read books, while evenings are reserved for working on projects. This year, your personal objective is to maintain good health, be productive, and write a book. The places you frequently visit are the park, library, gym, and office."

    memory = "On 2100-01-01, you had a meeting with John at 2 PM to discuss the project. You went for a run in the morning and read a book in the afternoon. In the evening, you worked on the project until 10 PM. \n On 2100-01-02, you received a call from John and he wants to meet you at 2 PM to discuss the project tomorrow."

    one_day_plan = planner.plan_day(env_info=env_info,personal_info=personal_info,memory=memory)
    for plan in planner.get_daily_plan():
        print(f"{plan.start_hour} - {plan.end_hour} : {plan.plan}")

    print("====================================================")
    current_hour = 10

    extra_info = "You received a call from John and he what to reschedule the meeting to 4 PM."

    planner.update_plan(env_info=env_info,personal_info=personal_info,memory=memory,current_hour=current_hour,extra_info=extra_info)

    for plan in planner.get_daily_plan():
        print(f"{plan.start_hour} - {plan.end_hour} : {plan.plan}")

    print("====================================================")

    extra_info = "You suddenly want to eat some ice cream."

    planner.update_plan(env_info=env_info,personal_info=personal_info,memory=memory,current_hour=current_hour,extra_info=extra_info)

    for plan in planner.get_daily_plan():
        print(f"{plan.start_hour} - {plan.end_hour} : {plan.plan}")


    print("====================================================")

    current_hour_plan = planner.plan_hour(env_info=env_info,personal_info=personal_info,memory=memory,current_hour=14)

    for plan in current_hour_plan:
        print(f"{plan.start_minute} - {plan.end_minute} : {plan.plan}")

