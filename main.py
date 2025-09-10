#!/usr/bin/env python
# coding: utf-8

from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from dotenv import load_dotenv
from google import genai


def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, "r", encoding="utf-8") as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]


def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk], embeddings=[embedding], ids=[str(i)]
        )


def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embed_chunk(query)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding], n_results=top_k
    )
    return results["documents"][0]


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)
    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks][:top_k]


def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{"\n\n".join(chunks)}

请基于上述内容作答，不要编造信息。"""

    print(f"{prompt}\n\n---\n")
    response = google_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text


if __name__ == "__main__":
    # 初始化组件
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    chromadb_client = chromadb.EphemeralClient()
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")
    load_dotenv()
    google_client = genai.Client()

    # 处理文档
    chunks = split_into_chunks("doc.md")
    for i, chunk in enumerate(chunks):
        print(f"[{i}] {chunk}\n")

    # 生成嵌入向量
    embedding = embed_chunk("测试内容")
    print(len(embedding))
    print(embedding)

    embeddings = [embed_chunk(chunk) for chunk in chunks]
    print(len(embeddings))
    print(embeddings[0])

    # 保存嵌入向量
    save_embeddings(chunks, embeddings)

    # 检索和重排序
    query = "大雄是谁"
    retrieved_chunks = retrieve(query, 5)
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")

    reranked_chunks = rerank(query, retrieved_chunks, 3)
    for i, chunk in enumerate(reranked_chunks):
        print(f"[{i}] {chunk}\n")

    # 生成回答
    answer = generate(query, reranked_chunks)
    print(answer)