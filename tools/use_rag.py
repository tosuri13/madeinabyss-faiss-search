import pickle
from pathlib import Path

import boto3
import faiss
import numpy as np
import xmltodict
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


def use_rag(question: str) -> None:
    index = faiss.read_index("assets/index.faiss")
    with open(Path("assets/index.pkl"), "rb") as f:
        chunks = pickle.load(f)

    client = boto3.client("bedrock-runtime", "us-east-1")
    embedding_model = BedrockEmbeddings(
        client=client,
        model_id="amazon.titan-embed-text-v2:0",
    )

    embedding = embedding_model.embed_query(question)
    embedding = np.array([embedding], dtype="float32")

    _, indices = index.search(embedding, k=3)
    documents = [chunks[i] for i in indices[0]]

    formatted_documents = xmltodict.unparse(
        {"documents": {"document": documents}},
        full_document=False,
        pretty=True,
    )

    messages = [
        SystemMessage(
            (
                "あなたは「メイドインアビス」と呼ばれるアニメに関する質問に回答するチャットボットです。\n"
                "これから与える<documents>の内容を参考にして、質問に対する回答を生成してください。\n"
                "\n"
                f"{formatted_documents}"
            )
        ),
        HumanMessage(question),
    ]

    model = ChatBedrockConverse(
        client=client,
        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    )
    response = model.invoke(messages)
    print(response.content)

    return None


if __name__ == "__main__":
    use_rag("深界二層の上昇負荷について教えて〜")
