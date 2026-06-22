"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys / LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Endpoint tương thích OpenAI (OpenAI/Gemini/gateway). LLM_HTTP_UA cho phép
# ghi đè User-Agent — một số gateway chặn UA mặc định của OpenAI SDK.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_HTTP_UA = os.getenv("LLM_HTTP_UA", "")


def _http_clients():
    """Trả về (sync_client, async_client) httpx với UA tùy chỉnh, hoặc (None, None)."""
    if not LLM_HTTP_UA:
        return None, None
    import httpx
    headers = {"User-Agent": LLM_HTTP_UA}
    return httpx.Client(headers=headers), httpx.AsyncClient(headers=headers)


def make_openai_client():
    """OpenAI SDK client cấu hình sẵn base_url/key + http_client (UA) nếu có."""
    from openai import OpenAI
    kwargs = {}
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    sync_client, _ = _http_clients()
    if sync_client is not None:
        kwargs["http_client"] = sync_client
    return OpenAI(**kwargs)


def make_chat_llm(temperature: float = 0.0):
    """langchain ChatOpenAI cho RAGAS — truyền cả http_client (sync) lẫn
    http_async_client (async) vì RAGAS gọi LLM bất đồng bộ."""
    from langchain_openai import ChatOpenAI
    kwargs = {"model": LLM_MODEL, "temperature": temperature}
    if OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    sync_client, async_client = _http_clients()
    if sync_client is not None:
        kwargs["http_client"] = sync_client
        kwargs["http_async_client"] = async_client
    return ChatOpenAI(**kwargs)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
# Gốc lab dùng BAAI/bge-m3 (1024-dim). Đổi sang multilingual-e5-large (cùng
# 1024-dim, đa ngữ, tốt cho tiếng Việt) vì đã có sẵn trong cache — tránh tải
# lại 2.2GB trên mạng chậm. Có thể đổi lại qua biến môi trường EMBEDDING_MODEL.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
