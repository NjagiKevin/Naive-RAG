import os
import ollama
import chromadb
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Constants
CHROMA_DB_PATH = "chroma_persistent_storage"
MODEL_NAME = "llama3.2:1b"

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Define a wrapper to ensure correct embedding function interface
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        # Use Ollama embeddings to generate embeddings for the input
        return OllamaEmbeddings(model=self.model).embed_documents(input)

# Use the wrapper to define the embedding function
embedding_function = EmbeddingFunctionWrapper(model="nomic-embed-text")

# Create or get the collection
collection_name = "doc_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function  # Corrected parameter
)

print(f"ChromaDB collection '{collection_name}' initialized successfully!")


# # Function to clear the collection by deleting all documents
# def clear_collection(collection):
#     # Fetch all documents from the collection (adjusting to get correct results)
#     documents = collection.get()  # This fetches the documents
    
#     # Check if documents are returned as expected (should be a list of dicts)
#     if isinstance(documents, dict) and 'documents' in documents:
#         ids_to_delete = [doc['id'] for doc in documents['documents']]  # Extract the IDs
        
#         # Delete all documents by their IDs
#         if ids_to_delete:
#             collection.delete(ids=ids_to_delete)
#             print(f"Cleared {len(ids_to_delete)} documents from the ChromaDB collection.")
#         else:
#             print("No documents to clear.")
#     else:
#         print("No documents found in the collection to delete.")

# # Clear the collection before inserting new data
# clear_collection(collection)


# Initialize ChatOllama 
chat_model = ChatOllama(model=MODEL_NAME)

# Load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# Updated function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"  
    )
    chunks = splitter.split_text(text)
    
    # Check and warn if a chunk exceeds the defined size
    for chunk in chunks:
        if len(chunk) > chunk_size:
            print(f"Warning: Created a chunk of size {len(chunk)}, which is longer than the specified {chunk_size}")
    
    return chunks

# Function to insert documents into Chroma DB
def insert_into_chroma_db(chunked_documents, embeddings, collection):
    # Extract the IDs and texts from the chunked documents
    ids = [doc['id'] for doc in chunked_documents]
    texts = [doc['text'] for doc in chunked_documents]

    # Flatten the embeddings (if they are a list of lists)
    flat_embeddings = [embedding[0] for embedding in embeddings]  # Flattening the embeddings list
    
    # Adding documents to the collection along with their embeddings
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=flat_embeddings  # Now passing flat embeddings
    )
    print("Documents successfully added to the ChromaDB collection!")

# Load documents from the directory
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

# Split documents into chunks and generate embeddings for each chunk
chunked_documents = []
embeddings = []  # List to store embeddings

for doc in documents:
    chunks = split_text(doc["text"])  # Split text into chunks
    print(f"==== Splitting doc '{doc['id']}' into chunks ====")
    
    for i, chunk in enumerate(chunks):
        # Generate embeddings for each chunk
        chunk_id = f"{doc['id']}_chunk{i+1}"
        chunk_embedding = embedding_function([chunk])  # Generate embeddings
        chunked_documents.append({"id": chunk_id, "text": chunk})
        embeddings.append(chunk_embedding)

# Now you have chunked_documents and embeddings
print(f"Split documents into {len(chunked_documents)} chunks and generated embeddings.")

# Insert the chunked documents and embeddings into the Chroma DB
insert_into_chroma_db(chunked_documents, embeddings, collection)


# Function to query documents
def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)

    # Flatten the list of lists of documents into a single list of strings
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    
    print("==== Returning relevant chunks ====")
    return relevant_chunks



# Function to generate a response from Ollama
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    # Query Ollama with the model name 'llama3.2:1b'
    response = ollama.chat(
        model="llama3.2:1b", 
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
    )

    # Print the response to debug
    print("Response from Ollama:", response)

    # Extract the response from Ollama
    try:
        answer = response['message']['content']  # Extracting the response text
    except KeyError:
        print("Error: Unexpected response format from Ollama.")
        answer = "Sorry, I couldn't generate a response."

    return answer


# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
question = "tell me about AI replacing TV writers strike."
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)


