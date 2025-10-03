#!/usr/bin/env python
# coding: utf-8

from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from dotenv import load_dotenv
from google import genai
import argparse  # 新增：导入argparse模块

# 全局初始化组件
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")
load_dotenv()
google_client = genai.Client()


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
    chunks_text = "\n\n".join(chunks)
    
    prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{chunks_text}

请基于上述内容作答，不要编造信息。"""

    print(f"{prompt}\n\n---\n")
    response = google_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text


if __name__ == "__main__":
    # 新增：设置命令行参数解析
    parser = argparse.ArgumentParser(description='RAG系统：基于文档的问答系统')
    parser.add_argument('--query', type=str, required=True, 
                       help='要查询的问题内容')
    parser.add_argument('--doc_file', type=str, default='doc-cfy.md',
                       help='文档文件路径（默认：doc-cfy.md）')
    parser.add_argument('--top_k', type=int, default=5,
                       help='检索返回的文档数量（默认：5）')
    parser.add_argument('--rerank_top_k', type=int, default=3,
                       help='重排序后保留的文档数量（默认：3）')
    
    args = parser.parse_args()

    # 处理文档（使用命令行参数指定的文件）
    chunks = split_into_chunks(args.doc_file)
    print(f"已加载文档：{args.doc_file}，共{len(chunks)}个段落")

    # 生成嵌入向量
    embeddings = [embed_chunk(chunk) for chunk in chunks]

    # 保存嵌入向量
    save_embeddings(chunks, embeddings)

    # 检索和重排序（使用命令行参数）
    retrieved_chunks = retrieve(args.query, args.top_k)
    print(f"检索到{len(retrieved_chunks)}个相关段落")
    
    reranked_chunks = rerank(args.query, retrieved_chunks, args.rerank_top_k)
    print(f"重排序后保留{len(reranked_chunks)}个最相关段落")

    # 生成回答
    answer = generate(args.query, reranked_chunks)
    print("\n=== 最终回答 ===")
    print(answer)