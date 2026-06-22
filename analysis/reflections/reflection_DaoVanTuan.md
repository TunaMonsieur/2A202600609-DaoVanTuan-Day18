# Individual Reflection — Lab 18: Production RAG

**Tên:** Đào Văn Tuấn
**Phụ trách:** Toàn bộ M1–M5 (bài cá nhân)

---

## Phần 1 — Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Embed câu bằng `all-MiniLM-L6-v2`, tách chunk khi cosine giữa 2 câu liền kề < threshold. Threshold thấp (0.5) → gộp nhiều câu, ít chunk hơn basic. |
| Hierarchical (parent/child) | M1 | `chunk_hierarchical()` | Parent ≤ 2048 char gom đoạn văn; child ≤ 256 char tách theo câu, mỗi child mang `parent_id`. Retrieve child (chính xác) → trả parent (đủ ngữ cảnh). 26 docs → 104 child chunks. |
| Structure-aware | M1 | `chunk_structure_aware()` | `re.split` theo header `#{1,3}` rồi ghép (header, body); giữ `section` trong metadata, không cắt giữa bảng/list. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF cộng `1/(k+rank+1)` từ 2 danh sách → hợp nhất lexical (BM25) và semantic (dense) mà không cần chuẩn hoá score khác thang. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | `underthesea` nối từ ghép bằng `_`; phải `replace("_"," ")` nếu không query 2 token "nghỉ phép" sẽ không khớp token "nghỉ_phép" trong BM25. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` chấm cặp (query, doc) → sort giảm dần, lấy top-3. Doc "nghỉ phép" được đẩy trên "VPN/mật khẩu" đúng như kỳ vọng. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | faithfulness/context_precision/context_recall dùng LLM làm giám khảo; answer_relevancy dùng embedding. Wrap try/except + NaN-guard để không vỡ khi 1 metric lỗi. |
| Diagnostic tree (failure) | M4 | `failure_analysis()` | Map metric thấp nhất → (chẩn đoán, cách sửa); sort theo điểm trung bình tăng dần lấy bottom-N. |
| Contextual embeddings | M5 | `contextual_prepend()` | Prepend 1 câu mô tả vị trí/chủ đề chunk (Anthropic: giảm ~49% retrieval failure). Có fallback "Trích từ {source}." khi không gọi LLM. |
| Cost-optimized enrichment | M5 | `_enrich_single_call()` | Gộp summary + câu hỏi + context + metadata vào **1 call/chunk** thay vì 4 → giảm 75% số request. |

**Số liệu RAGAS (20 câu, `gemini-3.1-flash-lite`):**

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8833 | 0.8500 | −0.0333 |
| Answer Relevancy | 0.6905 | 0.7876 | **+0.0971** |
| Context Precision | 0.7000 | 0.8000 | **+0.1000** |
| Context Recall | 0.7750 | 0.7917 | +0.0167 |

**Quan sát:** Reranking đẩy **context_precision +0.10** (lọc nhiễu top-20→top-3); enrichment + context tốt hơn đẩy **answer_relevancy +0.097**. Faithfulness giảm nhẹ trong nhiễu đo. Bottom-5 đều là câu **multi-hop/numeric/version** (laptop duyệt, thâm niên Senior, phạt tạm ứng, hoàn phí đào tạo, đổi mật khẩu v1/v2) — xem `analysis/failure_analysis.md`.

---

## Phần 2 — Khó khăn & cách giải quyết (thực tế buổi lab)

1. **Không có Python 3.11 như `.python-version` yêu cầu.** Máy chỉ có 3.10 và 3.13. `underthesea` yêu cầu Python `<3.13` nên 3.13 bị loại → chọn **3.10** (RAGAS 0.1.x vẫn chạy tốt, bài chấm không khoá phiên bản).

2. **`pip install` backtracking vô tận rồi chết ở jinja2.** Resolver kéo về các gói 2026 quá mới (`transformers 5.x`, `huggingface-hub 1.x`) gây xung đột, tụt xuống tận `jinja2 2.7` phải build từ source. **Fix:** ghim `transformers>=4.41,<5`, `huggingface-hub<1`, `jinja2>=3.1,<4`; cài `torch` bản CPU riêng từ `download.pytorch.org/whl/cpu`; tăng `--default-timeout 180 --retries 30` vì mạng chậm (timeout 15s liên tục).

3. **Lab viết cứng cho OpenAI nhưng chỉ có key Gemini.** Đấu nối qua endpoint tương thích OpenAI `https://generativelanguage.googleapis.com/v1beta/openai/`: đặt `OPENAI_BASE_URL` trong `.env` (OpenAI SDK tự đọc), thêm `LLM_MODEL=gemini-3.1-flash-lite`, thay các literal `gpt-4o-mini`. RAGAS mặc định gọi `gpt-4o` nên phải truyền `llm`/`embeddings` tường minh.

4. **`answer_relevancy = 0.0` — lỗi `400 Multiple candidates is not enabled`.** RAGAS sinh 3 câu hỏi bằng `n=3`, Gemini không hỗ trợ `n>1`. **Fix:** `answer_relevancy.strictness = 1` → dùng `n=1`. Sau fix metric lên ~0.95 ở smoke test.

5. **Không pull được image Qdrant (TLS handshake timeout tới Docker Hub).** **Fix:** `DenseSearch` thử server trước, không có thì fallback **Qdrant embedded in-memory** (`QdrantClient(":memory:")`) — không cần Docker và tránh tranh lock khi `main.py` tạo 2 instance trong cùng process.

6. **`bge-m3` (2.2GB) chưa cache, mạng quá chậm để tải.** Đổi `EMBEDDING_MODEL` sang `intfloat/multilingual-e5-large` (đã cache sẵn, cùng 1024-dim, đa ngữ tốt cho tiếng Việt).

---

## Phần 3 — Action plan cho project

### Hiện tại
- Pipeline RAG cơ bản: paragraph chunking + dense-only retrieval.
- Known issues: trả nhầm chính sách cũ (v2023/v1.0) khi có nhiều phiên bản; câu phủ định/đa bước trả lời yếu.

### Plan áp dụng
1. [ ] **Chunking:** dùng hierarchical (child 256 / parent 2048) — retrieve chính xác, trả đủ context; bổ sung structure-aware cho tài liệu có header rõ.
2. [ ] **Search:** hybrid BM25 (segment tiếng Việt) + dense, hợp nhất bằng RRF — xử lý cả từ khoá lẫn ngữ nghĩa.
3. [ ] **Reranking:** có — `bge-reranker-v2-m3`, top-20 → top-3 để tăng context precision trước khi đưa vào LLM.
4. [ ] **Evaluation:** RAGAS 4 metrics + diagnostic tree; theo dõi context_recall cho nhóm câu version/negation.
5. [ ] **Enrichment:** contextual prepend + auto metadata (lọc theo `version`/`effective_date` để loại tài liệu đã bị thay thế).

### Timeline
- Tuần 1: chunking + hybrid search + đánh giá baseline RAGAS.
- Tuần 2: reranking + enrichment + phân tích bottom-5, vá nhóm lỗi version/negation.

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | _ |
| Code quality | _ |
| Problem solving | _ |
| Hoàn thành | _ |

> _Lưu ý: điền số RAGAS ở Phần 1 và điểm tự chấm sau khi chạy xong `python main.py`._
