from typing import List,Optional
from pydantic import BaseModel,Field

class EventMemory(BaseModel):
    """
    Event Memory: Represents a specific personal experience tied to a time, location, and people involved.
    """
    event: str  # The event itself
    key_words: Optional[List[str]] = Field(default_factory=list)  # Keywords for indexing or searching, default to empty list

class FactMemory(BaseModel):
    """
    Fact Memory: Represents general knowledge or facts not tied to a specific time or place.
    """
    fact: str  # The fact itself
    key_words: Optional[List[str]] = Field(default_factory=list)  # Keywords for indexing or searching, default to empty list

class LongTermMemory(BaseModel):
    """
    Long Term Memory: Combines both FactMemory and EventMemory into a unified structure.
    """
    fact_memory: Optional[List[FactMemory]] = Field(default_factory=list)  # List of fact memories, default to empty list
    event_memory: Optional[List[EventMemory]] = Field(default_factory=list)  # List of event memories, default to empty list


class Memory:

    def __init__(self,
                 working_memory_threshold:int=10,
                 model_client=None,
                 model:str="gpt-4o-mini",
                 verbose:bool=False):
        self.working_memory_threshold = working_memory_threshold
        self.working_memory:List[str] = []
        self.long_term_memory:LongTermMemory = LongTermMemory(fact_memory=[],event_memory=[])
        self.model_client = model_client
        self.model = model
        self.verbose = verbose

    def add_working_memory(self, memory: str,date:str=None):
        self.working_memory.append(memory + f" Memory from {date}" if date else "")
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
            for fact in long_term_memory.fact_memory:
                fact.key_words = [keyword.lower() for keyword in fact.key_words]
        if long_term_memory.event_memory:
            for event in long_term_memory.event_memory:
                event.key_words = [keyword.lower() for keyword in event.key_words]

        if long_term_memory.fact_memory:
            self.long_term_memory.fact_memory.extend(long_term_memory.fact_memory)
        if long_term_memory.event_memory:
            self.long_term_memory.event_memory.extend(long_term_memory.event_memory)

    def retrieve_long_term_memory_by_recent(self,max_results:int=5):
        fact_memory = []
        event_memory = []
        if len(self.long_term_memory.fact_memory) > 0:
            fact_memory = self.long_term_memory.fact_memory[-max_results:]
        if len(self.long_term_memory.event_memory) > 0:
            event_memory = self.long_term_memory.event_memory[-max_results:]
        return fact_memory,event_memory

    def retrieve_long_term_memory_by_key_words(self,key_word:str,max_results:int=5):
        key_word = key_word.lower()
        fact_memory = []
        fact_memory_count = 0
        event_memory = []
        event_memory_count = 0
        for fact in self.long_term_memory.fact_memory[::-1]: # reverse to get the latest memory first
            if key_word in fact.key_words:
                fact_memory.append(fact)
                fact_memory_count += 1
                if fact_memory_count >= max_results:
                    break
        for event in self.long_term_memory.event_memory[::-1]:
            if key_word in event.key_words:
                event_memory.append(event)
                event_memory_count += 1
                if event_memory_count >= max_results:
                    break
        return fact_memory,event_memory