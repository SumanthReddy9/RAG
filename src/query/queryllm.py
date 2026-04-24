from abc import ABC, abstractmethod
import chromadb
import ollama
import json

class LLMCaller(ABC):
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./db")
        self.collection = self.client.get_or_create_collection("docs")

    @abstractmethod
    def call_llm(self, query):
        pass

    def get_context(self, query):
        results = self.collection.query(query_texts = [query], n_results = 5)
        chunks = results["documents"][0] 
        context = ' '.join(chunks)
        return context

class OllamaCaller(LLMCaller):
    
    def call_llm(self, query):
        context = self.get_context(query)
        answer = ollama.generate(
            model = "tinyllama",
            prompt = f"Context: \n{context}\n\nQuestion: {query}\n\nAnswer clearly and concisely:"
        )

        return {"answer": answer["response"]}

class AgenticCaller:

    def __init__(self, model_caller = OllamaCaller):

        self.caller = model_caller

        self.tools = ["semantic_retrival", "graph_retrival"]

    def agent_loop(self, query):
        prompt = '''
                You're an intelligent retrival agent

                You've have access to two tools which are basically your allowed actions to perform 

                1. semantic_retrival 
                 - retrives semantically similar documents to the given query from the vector database
                 - use this method when the qeury is about general questioning or explanation
                
                2. graph_retrival
                 - retrives complex relations and dependencies from the graph database in the ingested data
                 - use this when the query is about retriving connections between entities, dependencies or multi step reading 

                Guidelines:
                1. if the query is about how things are related use graph retrival 
                2. if the query is about simple explanations and information use semantic retrival 
                3. You can use both if needed 
                4. You can also call tools multiple times 
                5. Write the accurate action inputs so that the given tools will understand and takes as input

                You can only respond in below specified JSON formats:

                {
                    "thought": ...,
                    "action": ...,
                    "action_input":
                } 

                OR 

                {
                    "thought": ...,
                    "final_answer": ....
                }
                 '''
        
        MAX_LOOPS = 5
        history = []
        for step in range(MAX_LOOPS):
            
            final_prompt = prompt + "\n\n"

            for h in history:
                final_prompt += h + "\n"

            final_prompt += f"User Query : {query}\n"

            try:
                response = self.caller.call_llm(final_prompt)
                response = json.loads(response)
            except Exception as e:
                return f"LLM call failed with an exception {e}"
            
            if "final_answer" in response:
                return response["final_answer"]

            action = response["action"]

            if action not in self.tools:
                return f"Invalid tool {action} is called"

            try:
                result = self.tools[action](response["action_input"])
            except Exception as e:
                result = f"Tool error {e}"

            
            observation = f"""
                            "result": {result}
                            "Action": {action}
                            "Action input": {response["action_input"]}
                            "thought": {response["thought"]}
                           """

            history.append(observation)

        return "Max steps reached without final answer"