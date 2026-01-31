from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

#import the .env file
from dotenv import load_dotenv
load_dotenv()

#configuration
DATA_PATH = R"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#initiate the model
llm = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-3-pro")

#Connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Setup the vector store to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={"k": num_results})

#Call this function for every message added to the Chatbot
def stream_response(message, history):
    docs = retriever.invoke(message)
    
    knowledge = ""
    
    for doc in docs:
        knowledge += doc.page_content + "\n\n"
    
    if message is not None:
        partial_message = ""
        
        rag_prompt = f"""
        You are an assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The Knowledge" section.
        You don't mention anything to the user about the provided knowledge.
        The question: {message}
        Conversation history: {history}
        The Knowledge: {knowledge}
        """
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
            
            
#initaite the Gradio App
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(lines=2, placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

#launch the Gradio App
chatbot.launch(share=True)