import os
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
import gradio as gr

local_llm = "zephyr-7b-beta.Q4_K_S.gguf"


config = {
    "max_new_token": 1024,
    "repetition_penalty": 1.1,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2),
}

llm_init = CTransformers(model=local_llm, model_type="mistral", lib="avx2", **config)

prompt_template = """Use the following piece of information to answers the question asked by the user.
Don't try to make up the answer if you don't know the answer, simply say I don't know.

Context: {context}
Question: {question}

Only helpful answer below.
Helpful answer:
"""

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

load_vector_store = Chroma(
    persist_directory="stores/dino_cosine", embedding_function=embeddings
)

retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

# query = "How many genera of dinosaurs currently known?"

# semantic_search = retriever.get_relevant_documents(query)

# chain_type_kwargs = {"prompt": prompt}

# qa = RetrievalQA.from_chain_type(
#     llm=llm_init,
#     chain_type="stuff",
#     retriever=retriever,
#     verbose=True,
#     chain_type_kwargs=chain_type_kwargs,
#     return_source_documents=True,
# )

sample_query = [
    "How many genera of dinosaurs currently known?",
    "What methods are used to account for the incompleteness of the fossil record?",
    "Were Dinosaurs in Decline Before the Cretaceous or Tertiary Boundary?",
]


def get_response(input):
    query = input
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm_init,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    response = qa(query)
    return response


input = gr.Text(
    label="Query",
    show_label=True,
    max_lines=2,
    container=False,
    placeholder="Enter your question",
)

gIface = gr.Interface(
    fn=get_response,
    inputs=input,
    outputs="text",
    title="Dinosaurs Diversity RAG AI",
    description="RAG demo using Zephyr 7B Beta and Langchain",
    examples=sample_query,
    allow_flagging="never",
)

gIface.launch()

# llm_chain = LLMChain(prompt=prompt, llm=llm_init, verbose=True)
