from typing import List
from pydantic import BaseModel,Field

class Plan(BaseModel):
    start_hour: int
    end_hour: int
    plan: str 
class OneDayPlan(BaseModel):
    plans: List[Plan]

class Planner:
    def __init__(self,
                 model_client=None,
                 model:str="gpt-4o-mini",
                 verbose:bool=False):
        self.model_client = model_client
        self.model = model
        self.verbose = verbose 
        self.daily_plan = []

    def plan_day(self, env_info: str,personal_info: str,memory: str):
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
            "Ensure the updated plan is practical, well-balanced, and aligned with the provided information."
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

    def update_plan(self, env_info: str,personal_info: str,memory: str,current_hour: int,extra_info: str):
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
            "Ensure the updated plan is practical, well-balanced, and aligned with the provided information."
            "plans for 24 hours: "
        )

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

    def next_action(self,env_info: str,personal_info: str,memory: str,current_hour: int):
        current_hour_plan = self.get_current_hour_plan(current_hour)
        system_message = "You are skilled at determining the next action based on the current hour and the daily plan."
        prompt = (
            "### Environment Information:\n"
            f"```{env_info}```\n"
            "### Personal Information:\n"
            f"```{personal_info}```\n"
            "### Memory:\n"
            f"```{memory}```\n"
            "### Current Hour:\n"
            f"```{current_hour}```\n"
            "### Current Hour Plan:\n"
            f"```{current_hour_plan}```\n\n"
            "Based on the environment information, personal information, memory, and current hour plan, determine the next action. "
            "action can be go to somewhere, do something, or meet someone etc."
            "Ensure the next action is aligned with the current hour plan and the provided information."
        )

        messages = [{"role":"system","content":system_message}]
        messages.append({"role":"user","content":prompt})

        response = self.model_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=None,
                        tool_choice=None,
                    )
        
        return response.choices[0].message.content
        

    def get_daily_plan(self):
        return self.daily_plan
    
    def get_current_hour_plan(self,current_hour:int):
        for hour_plan in self.daily_plan:
            if current_hour >= hour_plan.start_hour and current_hour < hour_plan.end_hour:
                return hour_plan.plan
        return "Sleep"

if __name__ == "__main__":
    
    from dotenv import load_dotenv
    from openai import OpenAI

    # load the environment variables
    load_dotenv()
    # create a model client
    model_client = OpenAI()

    planner = Planner(model_client=model_client,model="gpt-4o",verbose=True)
    
    env_info = "This is year 2100,the world is a futuristic place with advanced technology. Today's weather is sunny and warm."

    personal_info = "Your name is Alice. You hold the position of a manager. You prefer mornings and enjoy running at that time. In the afternoons, you like to read books, while evenings are reserved for working on projects. This year, your personal objective is to maintain good health, be productive, and write a book. The places you frequently visit are the park, library, gym, and office."

    memory = "Today is 2025-01-21.yesterday you met John in the park and discussed your plans for summer vacation. You will have dinner with John at 7 PM at The Cheesecake Factory. You went to a party last night and danced all night."

    one_day_plan = planner.plan_day(env_info=env_info,personal_info=personal_info,memory=memory)
    for plan in planner.get_daily_plan():
        print(f"{plan.start_hour} - {plan.end_hour} : {plan.plan}")

    print("=====================================")
    current_hour = 10

    extra_info = "You received a call from John and he what to meet you at 2 PM to discuss the project."

    planner.update_plan(env_info=env_info,personal_info=personal_info,memory=memory,current_hour=current_hour,extra_info=extra_info)

    for plan in planner.get_daily_plan():
        print(f"{plan.start_hour} - {plan.end_hour} : {plan.plan}")


    print("=====================================")

    current_hour = 14
    print(planner.get_current_hour_plan(current_hour))
    current_hour = 20
    print(planner.get_current_hour_plan(current_hour))

    print("=====================================")

    next_action = planner.next_action(env_info=env_info,personal_info=personal_info,memory=memory,current_hour=1)
    print(next_action)