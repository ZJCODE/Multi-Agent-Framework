from typing import List,Optional
from pydantic import BaseModel,Field
import os
import uuid
import chromadb
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions

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
                 db_path:str=None, # for long term memory embedded database
                 verbose:bool=False):
        self.working_memory_threshold = working_memory_threshold
        self.working_memory:List[str] = []
        self.long_term_memory:LongTermMemory = LongTermMemory(fact_memory=[],event_memory=[])
        self.model_client = model_client
        self.model = model
        self.language = language
        self.db_path = db_path
        self.verbose = verbose
        self.memroy_unique_id = str(uuid.uuid4()) # can be replaced with a unique id provided by the agent
        load_dotenv()
        self._create_long_term_memory_db()

    def add_working_memory(self, memory: str):
        self.working_memory.append(memory)
        if len(self.working_memory) > self.working_memory_threshold:
            memory = self.working_memory.pop(0)
            self._extract_long_term_memory(memory) # 后续改造成异步或者在其他线程中执行

    def _create_long_term_memory_db(self):
        if self.db_path:
            if not os.path.exists(self.db_path):
                os.makedirs(self.db_path)
            client = chromadb.PersistentClient(self.db_path)
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                            api_key=os.environ.get("OPENAI_API_KEY"),
                            api_base=os.environ.get("OPENAI_BASE_URL"),
                            model_name="text-embedding-3-large" 
                        )
            self.db_collection = client.get_or_create_collection(self.memroy_unique_id,embedding_function=openai_ef)
        else:
            self.db_collection = None

    def _extract_long_term_memory(self, memory: str):

        system_message = "You are skilled at identifying and categorizing memories into Fact Memory (Semantic Memory) and Event Memory (Episodic Memory)."
    
        prompt = (
                "Extract and categorize memories into Fact Memory and Event Memory: \n"
                "### Memory:  \n"
                f"```{memory}```"
                "For every event memory,ensure that each event memory stands alone, allowing you to remember the complete event without needing additional context. Format like 'On January 21, 2025, at 3 PM, I met John in the park, where we discussed our plans for summer vacation, and afterward, we headed to the ice cream shop.' or 'On January 22, 2025, this morning, John invited me to have dinner with him tomorrow night at 7 PM at The Cheesecake Factory.' or 'On January 23, 2025, I went to the party at 7 PM, where I met my friend, Alice, and we danced all night.'.  \n"
                "For every fact memory, make sure it is a general knowledge or fact like Nature Facts,Personal Facts, Common Knowledge, or Scientific Facts. Format like 'The sky is blue.' or 'John is my friend.' or 'My favorite color is green.'.  \n"
                "Every Extracted Memory should only include information that is explicitly found within the memory; refrain from creating any details like who,time or place unless they are clearly mentioned in the memory.  \n"
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
            if self.db_collection:
                ids = [str(uuid.uuid4()) for _ in range(len(long_term_memory.fact_memory))]
                documents = long_term_memory.fact_memory
                self.db_collection.add(documents=documents,ids=ids)
        if long_term_memory.event_memory:
            self.long_term_memory.event_memory.extend(long_term_memory.event_memory)
            if self.db_collection:
                ids = [str(uuid.uuid4()) for _ in range(len(long_term_memory.event_memory))]
                documents = long_term_memory.event_memory
                self.db_collection.add(documents=documents,ids=ids)

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

    def retrieve_long_term_memory_by_query(self,query:str,max_results:int=5):
        res = self.db_collection.query(query_texts=[query],n_results=max_results)
        if res is None:
            return []
        try:
           return res['documents'][0]
        except:
            return []

    def get_memorys_str(self,query=None,max_results:int=3):

        working_memory = self.retrieve_working_memory()
        fact_memory, event_memory = self.retrieve_long_term_memory(max_results)
        semantic_matching = self.retrieve_long_term_memory_by_query(query,max_results)
        memorys = []
        if fact_memory:
            memorys.append("\n\n### Recent Fact Memory:\n")
            memorys.extend([f"{fact}\n---\n" for fact in fact_memory])
        if event_memory:
            memorys.append("\n\n### Recent Event Memory:\n")
            memorys.extend([f"{event}\n---\n" for event in event_memory])
        if working_memory:
            memorys.append("\n\n### Working Memory:\n")
            memorys.extend([f"{memory}\n---\n" for memory in working_memory])
        if semantic_matching:
            memorys.append("\n\n### Semantic Matching:\n")
            memorys.extend([f"{memory}\n---\n" for memory in semantic_matching])
        memorys_str = "\n".join(memorys)
        return memorys_str