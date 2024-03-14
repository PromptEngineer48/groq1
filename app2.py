import requests
import json
import gradio as gr

#import the environment variable
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
memory=ConversationBufferWindowMemory(k=10)

def generate_response(prompt):
    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='mixtral-8x7b-32768'
    )

    conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )
    
    system_prompt = """Please respond in a very short and sweet manner. You are a helpful assistant. Reply for me the following question=> {}""".format(prompt)
    
    if prompt:
        response = conversation(system_prompt)
        response = response['response']
        return response


interface=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4,placeholder="Enter your Prompt"),
    outputs="text"
)
interface.launch()
