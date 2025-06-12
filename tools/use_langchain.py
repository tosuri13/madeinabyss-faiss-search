import pickle
from pathlib import Path

import boto3
import faiss
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

if __name__ == "__main__":
    index = faiss.IndexFlatL2(1024)
    with open(Path("assets/index.pkl"), "rb") as f:
        chunks: list[str] = pickle.load(f)

    client = boto3.client("bedrock-runtime", "us-east-1")
    vectorstore = FAISS(
        embedding_function=BedrockEmbeddings(
            client=client,
            model_id="amazon.titan-embed-text-v2:0",
        ),
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    documents = [Document(chunk) for chunk in chunks]
    ids = [str(index) for index, _ in enumerate(chunks)]

    vectorstore.add_documents(documents, ids=ids)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    chain = RetrievalQA.from_llm(
        llm=ChatBedrockConverse(
            client=client,
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        ),
        retriever=retriever,
    )
    response = chain.run("深界二層の上昇負荷について教えて〜")
    print(response)
