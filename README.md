# RAG 问答系统

基于文档的问答系统，支持用户输入问题并从指定文档中检索相关段落，生成回答。

## 功能特性

- **文档加载**：支持加载指定文档文件（如 `doc-cfy.md`）。
- **嵌入向量生成**：将文档段落转换为嵌入向量以便检索。
- **检索与重排序**：根据用户问题检索相关段落，并进行重排序。
- **回答生成**：基于重排序后的段落生成最终回答。
- **Web 界面**：提供类似 ChatGPT 的交互界面。

## 环境搭建

### 依赖安装

1. 确保已安装 Python 3.8+ 和 pip。
2. 克隆项目：
   ```bash
   git clone https://github.com/your-repo/rag-project.git
   cd rag-project
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

### 申请 GEMINI_API_KEY

1. 访问 [Gemini API 官网](https://ai.google.dev/) 并注册账号。
2. 在控制台中创建 API 密钥。
3. 将生成的 `GEMINI_API_KEY` 添加到环境变量中：
   ```bash
   export GEMINI_API_KEY="你的_API_密钥"
   ```
   - 或者将其添加到项目的 `.env` 文件中。

### 文档准备

将文档文件（如 `doc-cfy.md`）放置在项目根目录下。

## 服务启动

1. 启动后端服务：
   ```bash
   python app.py
   ```
   - 服务默认运行在 `http://localhost:5000`。

2. 访问前端界面：
   - 打开浏览器访问 `http://localhost:5000`。

## 接口说明

### 查询接口

- **URL**: `/query`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "用户问题",
    "doc_file": "doc-cfy.md",
    "top_k": 5,
    "rerank_top_k": 3
  }
  ```
- **Response**:
  ```json
  {
    "answer": "生成的回答"
  }
  ```

## 项目结构

```
/root/rag-project
├── app.py                # 后端服务入口
├── main.py               # 核心逻辑（文档处理、检索、生成）
├── templates
│   └── index.html        # 前端界面
├── requirements.txt      # 依赖列表
└── README.md             # 项目文档
```

## 后续计划

- 支持多文档加载。
- 优化检索算法。
- 增加用户认证功能。