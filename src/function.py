import io
import json
import pickle

import boto3
import faiss
import numpy as np
import xmltodict
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


def handler(event, context):
    request = json.loads(event["body"])
    question = request["question"]

    client = boto3.client("s3", "ap-northeast-1")

    response = client.get_object(
        Bucket="madeinabyss-faiss-search-index-bucket",
        Key="index.faiss",
    )
    index_data = response["Body"].read()
    index = faiss.deserialize_index(np.frombuffer(index_data, dtype="uint8"))

    response = client.get_object(
        Bucket="madeinabyss-faiss-search-index-bucket",
        Key="index.pkl",
    )
    chunks = pickle.load(io.BytesIO(response["Body"].read()))

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
    answer = response.content

    return {
        "statusCode": 200,
        "body": json.dumps({"answer": answer}, ensure_ascii=False),
    }
