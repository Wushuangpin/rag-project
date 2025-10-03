from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import split_into_chunks, embed_chunk, save_embeddings, retrieve, rerank, generate
import argparse

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')
    doc_file = data.get('doc_file', 'doc-cfy.md')
    top_k = data.get('top_k', 5)
    rerank_top_k = data.get('rerank_top_k', 3)

    # 处理文档
    chunks = split_into_chunks(doc_file)
    print(f"已加载文档：{doc_file}，共{len(chunks)}个段落")

    # 生成嵌入向量
    embeddings = [embed_chunk(chunk) for chunk in chunks]

    # 保存嵌入向量
    save_embeddings(chunks, embeddings)

    # 检索和重排序
    retrieved_chunks = retrieve(query, top_k)
    print(f"检索到{len(retrieved_chunks)}个相关段落")
    
    reranked_chunks = rerank(query, retrieved_chunks, rerank_top_k)
    print(f"重排序后保留{len(reranked_chunks)}个最相关段落")

    # 生成回答
    answer = generate(query, reranked_chunks)
    print("\n=== 最终回答 ===")
    print(answer)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)