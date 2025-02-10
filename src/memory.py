from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import os
import uuid
import chromadb
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions

class LongTermMemory(BaseModel):
    """
    Long Term Memory: Combines both FactMemory and EventMemory into a unified structure.
    """
    fact_memory: Optional[List[str]] = Field(default_factory=list)
    event_memory: Optional[List[str]] = Field(default_factory=list)

class Memory:
    """
    Simple memory for demo.
    """

    def __init__(self,
                 working_memory_threshold: int = 10,
                 model_client=None,
                 model: str = "gpt-4o-mini",
                 language: str = None,
                 db_path: str = None,  # For long term memory embedded database
                 verbose: bool = False):
        self.working_memory_threshold = working_memory_threshold
        self.working_memory: List[str] = []
        self.long_term_memory: LongTermMemory = LongTermMemory()
        self.model_client = model_client
        self.model = model
        self.language = language
        self.db_path = db_path
        self.verbose = verbose
        self.memory_unique_id = str(uuid.uuid4())  # Unique id for the agent
        load_dotenv()
        self._create_long_term_memory_db()

    def add_working_memory(self, memory: str) -> None:
        self.working_memory.append(memory)
        if len(self.working_memory) > self.working_memory_threshold:
            removed_memory = self.working_memory.pop(0)
            self._extract_long_term_memory(removed_memory)  # Can be updated to run asynchronously later

    def _create_long_term_memory_db(self) -> None:
        if self.db_path:
            if not os.path.exists(self.db_path):
                os.makedirs(self.db_path)
            client = chromadb.PersistentClient(self.db_path)
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                api_base=os.environ.get("OPENAI_BASE_URL"),
                model_name="text-embedding-3-large"
            )
            self.db_collection = client.get_or_create_collection(self.memory_unique_id, embedding_function=openai_ef)
        else:
            self.db_collection = None

    def _extract_long_term_memory(self, memory: str) -> None:
        system_message = "You are skilled at identifying and categorizing memories into Fact Memory (Semantic Memory) and Event Memory (Episodic Memory)."
    
        prompt = (
            "Extract and categorize memories into Fact Memory and Event Memory: \n"
            "### Memory:  \n"
            f"```{memory}```\n"
            "For every event memory, ensure that each event memory stands alone, allowing you to remember the complete event without needing additional context. "
            "Format like 'On January 21, 2025, at 3 PM, I met John in the park, where we discussed our plans for summer vacation, and afterward, we headed to the ice cream shop.' or "
            "'On January 22, 2025, this morning, John invited me to have dinner with him tomorrow night at 7 PM at The Cheesecake Factory.' or "
            "'On January 23, 2025, I went to the party at 7 PM, where I met my friend, Alice, and we danced all night.' \n"
            "For every fact memory, make sure it is general knowledge or a fact like Nature Facts, Personal Facts, Common Knowledge, or Scientific Facts. "
            "Format like 'The sky is blue.' or 'John is my friend.' or 'My favorite color is green.' \n"
            "Only include information explicitly provided without adding details unless they are clearly mentioned."
        )
        
        if self.language:
            prompt += f"\n\n### Response in Language: {self.language}"
    
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}]
        
        completion = self.model_client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format=LongTermMemory
        )
        long_term_memory: LongTermMemory = completion.choices[0].message.parsed

        if long_term_memory.fact_memory:
            self.long_term_memory.fact_memory.extend(long_term_memory.fact_memory)
            if self.db_collection:
                ids = [str(uuid.uuid4()) for _ in range(len(long_term_memory.fact_memory))]
                documents = long_term_memory.fact_memory
                self.db_collection.add(documents=documents, ids=ids)
        if long_term_memory.event_memory:
            self.long_term_memory.event_memory.extend(long_term_memory.event_memory)
            if self.db_collection:
                ids = [str(uuid.uuid4()) for _ in range(len(long_term_memory.event_memory))]
                documents = long_term_memory.event_memory
                self.db_collection.add(documents=documents, ids=ids)

    def retrieve_working_memory(self) -> List[str]:
        return self.working_memory

    def retrieve_long_term_memory(self, max_results: int = 5) -> Tuple[List[str], List[str]]:
        fact_memory = self.long_term_memory.fact_memory[-max_results:] if self.long_term_memory.fact_memory else []
        event_memory = self.long_term_memory.event_memory[-max_results:] if self.long_term_memory.event_memory else []
        return fact_memory, event_memory

    def retrieve_long_term_memory_by_query(self, query: str, max_results: int = 5) -> Any:
        if self.db_collection is None:
            return []
        # Retrieve available document count from the collection.
        available_count = self.db_collection.count() if hasattr(self.db_collection, "count") else max_results
        n_results = min(max_results, available_count)
        res = self.db_collection.query(query_texts=[query], n_results=n_results)
        if res is None:
            return []
        try:
            return res.get('documents', [])[0]
        except (IndexError, KeyError):
            return []

    def get_memorys_str(self, query: str = None, max_results: int = 3, enhanced_filter: bool = False) -> str:
        working_memory = self.retrieve_working_memory()
        fact_memory, event_memory = self.retrieve_long_term_memory(max_results)
        semantic_matching = self.retrieve_long_term_memory_by_query(query, max_results) if query else []

        sections = [
            ("Recent Fact Memory", fact_memory),
            ("Recent Event Memory", event_memory),
            ("Working Memory", working_memory),
            ("Semantic Matching", semantic_matching),
        ]

        memory_fragments = []
        for title, memories in sections:
            if memories:
                memory_fragments.append(f"\n\n### {title}:\n")
                memory_fragments.extend(f"{memory}\n---\n" for memory in memories)
        memories_res = "\n".join(memory_fragments)

        if query and enhanced_filter:
            system_message = (
                f"You are skilled at identifying and selecting relevant memories based on the context provided. "
                f"Here are the initial filtered memories:\n```{memories_res}```"
            )
            prompt = (
                "Select the most relevant memories based on the current context: \n"
                f"```{query}```\n"
                "Most relevant memories are those that are directly related to the context provided and can be used to answer the query effectively."
            )
            if self.language:
                prompt += f"\n\n### Response in Language: {self.language}"

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            
            completion = self.model_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=None,
                tool_choice=None,
                temperature=0.0,
            )
            return completion.choices[0].message.content

        return memories_res
    

if __name__ == "__main__":

    from dotenv import load_dotenv
    from openai import OpenAI

    # load the environment variables
    load_dotenv()
    # create a model client
    model_client = OpenAI()
    
    memory = Memory(working_memory_threshold=2, model_client=model_client, model="gpt-4o-mini", language="en", db_path="data")
    memory.add_working_memory("2025-01-21 15:00:00, I met John in the park, where we discussed our plans for summer vacation, and afterward, we headed to the ice cream shop.")
    memory.add_working_memory("2025-01-22 19:00:00, John invited me to have dinner with him tomorrow night at 7 PM at The Cheesecake Factory.")
    memory.add_working_memory("2025-01-23 19:00:00, I went to the party at 7 PM, where I met my friend, Alice, and we danced all night.")
    memory.add_working_memory("The sky is blue.")

    print(memory.get_memorys_str(query=None,enhanced_filter=True))
    print(memory.get_memorys_str(query="hey how are you",enhanced_filter=True))