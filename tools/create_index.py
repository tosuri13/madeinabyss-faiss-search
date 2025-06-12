import pickle
from pathlib import Path

import boto3
import faiss
import numpy as np
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from tqdm import tqdm


def create_index() -> None:
    source = Path("assets/source.txt").read_text()

    splitter = MarkdownTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
    )
    # NOTE: Wikiの後半部分は関連性が低いので省略する
    chunks = splitter.split_text(source)[:35]

    # 実コンテキストが格納されたオブジェクトを保存する
    with open(Path("assets/index.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    client = boto3.client("bedrock-runtime", "us-east-1")
    embedding_model = BedrockEmbeddings(
        client=client,
        model_id="amazon.titan-embed-text-v2:0",
    )

    embeddings = []
    print("● Creating FAISS index...")

    for chunk in tqdm(chunks):
        embedding = embedding_model.embed_query(chunk)
        embedding = np.array(embedding, dtype="float32")

        embeddings.append(embedding)

    # インデックスを構築して、何度も使用できるように保存しておく
    index = faiss.IndexFlatIP(1024)
    index.add(np.array(embeddings, dtype="float32"))  # type: ignore

    faiss.write_index(index, "assets/index.faiss")

    return None


if __name__ == "__main__":
    create_index()
