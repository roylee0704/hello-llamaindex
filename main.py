from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def main():
    # Initialize OpenAI LLM
    llm = OpenAI(model="gpt-3.5-turbo")

    # Create some sample text
    with open("sample.txt", "w") as f:
        f.write("Hello, this is a sample document for LlamaIndex!")

    # Load the document
    documents = SimpleDirectoryReader(input_files=["sample.txt"]).load_data()

    # Create an index
    index = VectorStoreIndex.from_documents(documents)

    # Create query engine
    query_engine = index.as_query_engine()

    # Query the index
    response = query_engine.query("What does the document say?")
    print(response)

if __name__ == "__main__":
    main()