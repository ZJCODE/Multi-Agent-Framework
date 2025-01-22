from typing import List,Optional
from pydantic import BaseModel,Field

class LongTermMemory(BaseModel):
    """
    Long Term Memory: Combines both FactMemory and EventMemory into a unified structure.
    """
    fact_memory: Optional[List[str]] = Field(default_factory=list)  # List of fact memories, default to empty list
    event_memory: Optional[List[str]] = Field(default_factory=list)  # List of event memories, default to empty list

class Memory:
    """
    simple memory for demo
    """

    def __init__(self,
                 working_memory_threshold:int=10,
                 model_client=None,
                 model:str="gpt-4o-mini",
                 language:str=None,
                 verbose:bool=False):
        self.working_memory_threshold = working_memory_threshold
        self.working_memory:List[str] = []
        self.long_term_memory:LongTermMemory = LongTermMemory(fact_memory=[],event_memory=[])
        self.model_client = model_client
        self.model = model
        self.language = language
        self.verbose = verbose

    def add_working_memory(self, memory: str):
        self.working_memory.append(memory)
        if len(self.working_memory) > self.working_memory_threshold:
            memory = self.working_memory.pop(0)
            self._extract_long_term_memory(memory) # 后续改造成异步或者在其他线程中执行

    def _extract_long_term_memory(self, memory: str):

        system_message = "You are skilled at identifying and categorizing memories into Fact Memory (Semantic Memory) and Event Memory (Episodic Memory)."
    
        prompt = (
                "Extract and categorize memories into Fact Memory and Event Memory: \n"
                "### Memory:  \n"
                f"```{memory}```"
                "For every event memory,ensure that each event memory stands alone, allowing you to remember the complete event without needing additional context. Format like 'On January 21, 2025, at 3 PM, I met John in the park, where we discussed our plans for summer vacation, and afterward, we headed to the ice cream shop.' or 'On January 22, 2025, this morning, John invited me to have dinner with him tomorrow night at 7 PM at The Cheesecake Factory.' or 'On January 23, 2025, I went to the party at 7 PM, where I met my friend, Alice, and we danced all night.'.\n"
                "For every fact memory, make sure it is a general knowledge or fact like Nature Facts,Personal Facts, Common Knowledge, or Scientific Facts. Format like 'The sky is blue.' or 'John is my friend.' or 'My favorite color is green.'.  \n"
            )
        
        if self.language:
            prompt += f"\n\n### Response in Language: {self.language}"

        messages = [{"role":"system","content":system_message}]
        messages.append({"role":"user","content":prompt})
        
        completion = self.model_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format=LongTermMemory
        )
        long_term_memory = completion.choices[0].message.parsed

        if long_term_memory.fact_memory:
            self.long_term_memory.fact_memory.extend(long_term_memory.fact_memory)
        if long_term_memory.event_memory:
            self.long_term_memory.event_memory.extend(long_term_memory.event_memory)

    def retrieve_working_memory(self):
        return self.working_memory

    def retrieve_long_term_memory(self,max_results:int=5):
        fact_memory = []
        event_memory = []
        if len(self.long_term_memory.fact_memory) > 0:
            fact_memory = self.long_term_memory.fact_memory[-max_results:]
        if len(self.long_term_memory.event_memory) > 0:
            event_memory = self.long_term_memory.event_memory[-max_results:]
        return fact_memory,event_memory

    def get_memorys_str(self,max_results:int=5):
        working_memory = self.retrieve_working_memory()
        fact_memory, event_memory = self.retrieve_long_term_memory(max_results)
        memorys = []
        if fact_memory:
            memorys.append("### Recent Fact Memory:")
            memorys.extend([f"- {fact}" for fact in fact_memory])
        if event_memory:
            memorys.append("### Recent Event Memory:")
            memorys.extend([f"- {event}" for event in event_memory])
        if working_memory:
            memorys.append("### Working Memory:")
            memorys.extend([f"- {memory}" for memory in working_memory])
        memorys_str = "\n".join(memorys)
        return memorys_str