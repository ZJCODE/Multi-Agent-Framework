from typing import List, Optional, Tuple, Any, Literal
from pydantic import BaseModel, Field
import os
import uuid
import chromadb
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions

from src.utilities.logger import Logger

# 增加记忆衰减机制
# 添加记忆整合机制
#   Find similar memories using semantic similarity
#   Merge related memories into more concise, comprehensive versions
#   Increase the importance of frequently referenced information
#   Update the vector database to reflect the consolidated memories

class MemoryItem(BaseModel):
    """
    Memory Item: Represents a single memory item.
    """
    content: str
    timestamp: Optional[str] = None
    category: Literal["episodic", "semantic","procedural","social"] = Field(...,description="Category of the memory item")
    retrieval_cue: str = Field(..., description="Retrieval cue for the memory item")
    importance: Literal[1,2,3,4,5] = Field(..., description="Importance of the memory item")
    emotional_valence: Literal[-2,-1,0,1,2] = Field(..., description="Emotional valence of the memory item")
class LongTermMemory(BaseModel):
    """
    Long Term Memory: Represents the long term memory of the agent.
    """
    memorys: Optional[List[MemoryItem]] = Field(default_factory=list)

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
                 memory_unique_id: str = None,
                 verbose: bool = False):
        self._logger = Logger(verbose=verbose)
        self.working_memory_threshold = working_memory_threshold
        self.working_memory: List[str] = []
        self.long_term_memory: LongTermMemory = LongTermMemory()
        self.model_client = model_client
        self.model = model
        self.language = language
        self.db_path = db_path
        self.verbose = verbose
        self.memory_unique_id = str(uuid.uuid4()) if memory_unique_id is None else memory_unique_id
        load_dotenv()
        self._create_long_term_memory_db()
        self._logger.log("info",f"Memory initialized with working memory threshold: {working_memory_threshold}")

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
            self._logger.log("info",f"Long term memory database created at: {self.db_path}")
        else:
            self.db_collection = None

    def _extract_long_term_memory(self, memory: str) -> None:
        system_message = "You are skilled at identifying and categorizing memories for long-term storage."
            
        prompt = (
            "Based on established principles in cognitive science, please analyze the following memory sample with an emphasis on memory systems and retrieval mechanisms:\n\n"
            "### Memory Sample:\n"
            "```\n"
            f"{memory}\n"
            "```\n\n"
            "### Analysis Requirements:\n"
            "1. **Summary:**\n"
            "   - Summarize the core content of the memory.\n"
            "   - Identify and extract any available timestamp (date and time).\n\n"
            "2. **Memory Categorization:**\n"
            "   - Classify the memory into one of the following types based on cognitive theories:\n"
            "     - **Episodic:** Personal experiences and specific events.\n"
            "     - **Semantic:** General knowledge or facts.\n"
            "     - **Procedural:** Skills or how-to knowledge.\n"
            "     - **Social:** Details regarding interpersonal relationships.\n\n"
            "3. **Retrieval Cue:**\n"
            "   - Propose a precise retrieval cue (keyword or phrase) that can efficiently trigger recall of this memory.\n\n"
            "4. **Memory Properties Evaluation:**\n"
            "   - Assign an **Importance Rating** on a scale from 1 to 5 (with 5 being the highest importance).\n"
            "   - Specify the **Emotional Valence** on a scale from -2 (very negative) to +2 (very positive), using only objective evidence from the memory content.\n\n"
            "Ensure that your analysis is strictly factual and directly derived from the memory sample, without introducing any additional speculation."
        )
 
        self._logger.log("info",f"Start Extracting Long Term Memory...")
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

        self._logger.log("info",f"Extract Long Term Memory Completed.")

        if long_term_memory.memorys:
            # self.long_term_memory.memorys.extend(long_term_memory.memorys)
            if self.db_collection:
                self._logger.log("info",f"Adding extracted memory to the long term memory vector database.")
                ids = [str(uuid.uuid4()) for _ in range(len(long_term_memory.memorys))]
                documents = [memory.content for memory in long_term_memory.memorys]
                metadatas = [{"category": memory.category ,
                              "timestamp": memory.timestamp or "",
                              "retrieval_cue": memory.retrieval_cue,
                              "importance": memory.importance,
                              "emotional_valence": memory.emotional_valence}
                              for memory in long_term_memory.memorys]
                self.db_collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def retrieve_working_memory(self) -> List[str]:
        return self.working_memory

    def retrieve_long_term_memory(self, query: str, max_results: int = 5) -> Any:
        if self.db_collection is None or not query:
            return []
        # Retrieve available document count from the collection.
        available_count = self.db_collection.count() if hasattr(self.db_collection, "count") else max_results
        if available_count == 0:
            return []
        n_results = min(max_results, available_count)
        res = self.db_collection.query(query_texts=[query], n_results=n_results)
        if res is None:
            return []
        try:
            return [{"content": doc, "time": meta.get('timestamp',None)} for doc, meta 
                    in zip(res.get('documents', [])[0],res.get('metadatas', [])[0])]
        except (IndexError, KeyError):
            return []

    def get_memorys_str(self, query: str = None, max_results: int = 3, enhanced_filter: bool = False) -> str:
        working_memory = self.retrieve_working_memory()
        semantic_matching = self.retrieve_long_term_memory(query, max_results)

        sections = [
            ("Working Memory", working_memory),
            ("Semantic Matching", semantic_matching),
        ]

        memory_fragments = []
        for title, memories in sections:
            if memories:
                memory_fragments.append(f"\n\n### {title}:\n")
                memory_fragments.extend(f"{memory}\n" for memory in memories)
        memories_res = "\n".join(memory_fragments)

        self._logger.log("info",f"Retrieved memories:\n{memories_res}")

        if query and enhanced_filter:
            system_message = (
                f"You are skilled at identifying and selecting relevant memories based on the context provided. "
                f"Here are the initial filtered memories:\n```{memories_res}```"
            )
            prompt = (
                "Select the most relevant memories based on the current context: \n"
                f"```{query}```\n"
                "Most relevant memories are those that are directly related to the context provided and can be used to answer the query effectively."
                "Just return the memories do not add any additional information and without code blocks."
                "If There is no relevant memory, please type 'No relevant memory'."
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

            filtered_memories = completion.choices[0].message.content

            self._logger.log("info",f"Filtered memories:\n\n{filtered_memories}",color="bold_blue")

            return filtered_memories

        return memories_res
    

if __name__ == "__main__":

    from dotenv import load_dotenv
    from openai import OpenAI

    # load the environment variables
    load_dotenv()
    # create a model client
    model_client = OpenAI()
    
    memory = Memory(working_memory_threshold=2, model_client=model_client, 
                    model="gpt-4o-mini", language="en", db_path="data",verbose=True)
    
    memory.add_working_memory("The sky is blue.")
    memory.add_working_memory("John and Alice are classmates.")
    memory.add_working_memory("2025-01-21 15:00:00, I met John in the park, where we discussed our plans for summer vacation, and afterward, we headed to the ice cream shop.")
    memory.add_working_memory("I attended the meeting at 10 AM, where we discussed the new project timeline and deliverables.")
    memory.add_working_memory("2025-01-22 19:00:00, John invited me to have dinner with him tomorrow night at 7 PM at The Cheesecake Factory.")
    memory.add_working_memory("2025-01-23 19:00:00, I went to the party at 7 PM, where I met my friend, Alice, and we danced all night.")

    memory.get_memorys_str(query="what did John do?",enhanced_filter=True)