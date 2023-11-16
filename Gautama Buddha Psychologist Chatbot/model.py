from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain import HuggingFaceHub
from llama_cpp import Llama
import chainlit as cl
import os

 

DB_FAISS_PATH = 'vectorstore/buddha_data_faiss'

#prompt
custom_prompt_template = """Please provide answer to user's question using given context. The answer should be short and accurate. Do not try to over explain. Do not copy the prompt in your answer. Your answer should not be longer than 200 words. Do not hallucinate.
Context: {context}
Question: {question}
OUTPUT ONLY THE ANSWER
Your answer: 
"""

#prompt template
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


#model loading   
def load_model():
    local_llm = "TheBloke/zephyr-7B-beta-GGUF"

    config = {
    'max_new_tokens': 1024,
    'repetition_penalty': 1.1,
    'temperature': 0.4,
    'top_k': 50,
    'top_p': 0.9,
    'stream': True,
    'threads': int(os.cpu_count() / 2)
    }

    llm = CTransformers(
        model=local_llm,
        model_type="mistral",
        lib="avx2", #for CPU use
        **config
    )
   
    return llm   

#conversational_chain
def create_conversational_chain(llm, prompt, db):
    # Create llm
     
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=db.as_retriever(search_kwargs={"k": 2}), memory=memory,
                                                 combine_docs_chain_kwargs={"prompt": prompt})
    return chain


#chatbot
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_model()
    qa_prompt = set_custom_prompt()
    qa = create_conversational_chain(llm, qa_prompt, db)

    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"Chatbot": "Buddha ☸"}
    return rename_dict.get(orig_author, orig_author)


@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Gautam Buddha Psychologist Bot, please share with me what's bothering you?"
    await msg.update()

    cl.user_session.set("chain", chain)



@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=False, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]

    #sources = res["source_documents"]


    # if sources:
    #     answer += f"\n\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()
 


