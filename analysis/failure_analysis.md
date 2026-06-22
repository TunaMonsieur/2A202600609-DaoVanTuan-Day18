# Failure Analysis — Lab 18: Production RAG

**Người thực hiện:** Đào Văn Tuấn (bài cá nhân — toàn bộ M1–M5)
**Cấu hình chạy:** Hierarchical chunking (104 child chunks) → M5 enrichment (1 call/chunk) → Hybrid BM25+Dense (e5-large, RRF) → CrossEncoder rerank (bge-reranker-v2-m3, top-3) → LLM `gemini-3.1-flash-lite` → RAGAS. Embedded Qdrant in-memory. Test set: 20 câu.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8833 | 0.8500 | −0.0333 |
| Answer Relevancy | 0.6905 | 0.7876 | **+0.0971** |
| Context Precision | 0.7000 | 0.8000 | **+0.1000** |
| Context Recall | 0.7750 | 0.7917 | +0.0167 |

> Production cải thiện rõ ở **context_precision** (+0.10, nhờ reranking) và **answer_relevancy** (+0.097, nhờ enrichment + context tốt hơn). Faithfulness giảm nhẹ (−0.033) nằm trong nhiễu đo (có 1 job RAGAS timeout → 1 ô tính 0.0). Naive baseline = paragraph chunking + dense-only, không rerank/enrichment.

---

## Bottom-5 Failures (theo avg 4 metrics, từ `reports/ragas_report.json`)

### #1 — avg 0.271 · worst: answer_relevancy = 0.00
- **Question:** "Nếu cần mua laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"
- **Expected:** Director phê duyệt (mức 5–50tr) + xác nhận cấu hình kỹ thuật từ CNTT + ≥3 báo giá (vì >10tr).
- **Worst metric:** answer_relevancy (câu trả lời lạc đề).
- **Error Tree:** Output sai → Context đúng một phần (chunk mua sắm) nhưng câu hỏi **multi-hop** (ghép 3 quy tắc: ngưỡng phê duyệt + quy trình CNTT + số báo giá) → LLM chỉ trả lời 1 phần → relevancy ≈ 0.
- **Root cause:** Câu hỏi đòi tổng hợp nhiều mục; retrieval lấy được 1 chunk chủ đạo, thiếu các điều kiện phụ.
- **Suggested fix:** Tăng top-k trước rerank để gom đủ mảnh; prompt yêu cầu liệt kê đầy đủ điều kiện (ai duyệt + giấy tờ + báo giá).

### #2 — avg 0.554 · worst: context_precision = 0.00
- **Question:** "Nhân viên Senior 9 năm thâm niên được nghỉ bao nhiêu ngày phép và lương khoảng nào?"
- **Expected:** 15 + 3 (9÷3) = **18 ngày**; lương Senior (P3–P4) **20–35tr/tháng**.
- **Worst metric:** context_precision (nhiều chunk nhiễu).
- **Error Tree:** Output sai → Context lẫn nhiều chunk không liên quan → câu cần ghép **2 nguồn** (chính sách phép + bảng lương) → kéo vào nhiều chunk thừa → precision tụt.
- **Root cause:** Cross-domain multi-hop: trộn tài liệu nghỉ phép v2024 và bảng lương; rerank chưa đủ tách nhiễu.
- **Suggested fix:** Metadata filter theo `category` (leave vs salary) rồi hợp nhất; hoặc tăng strictness của reranker.

### #3 — avg 0.610 · worst: faithfulness = 0.25
- **Question:** "Tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** Hạn 15 ngày; quá 5 ngày, phí 2%/tháng trên 15tr = 300k/tháng (~50k cho 5 ngày).
- **Worst metric:** faithfulness (bịa số liệu).
- **Error Tree:** Output sai → Context có quy định phí nhưng cần **tính toán số học pro-rata** → LLM hallucination con số → faithfulness thấp.
- **Root cause:** Câu numeric đòi suy luận tính toán; LLM tự "chế" số không có trong context.
- **Suggested fix:** Hạ temperature=0; prompt "chỉ dùng số có trong context, nêu công thức"; cân nhắc tool tính toán.

### #4 — avg 0.694 · worst: faithfulness = 0.00
- **Question:** "Được tài trợ khóa học 25 triệu, nghỉ sau 8 tháng. Phải hoàn trả bao nhiêu?"
- **Expected:** Cam kết tối thiểu 1 năm; nghỉ trước hạn → hoàn **100% = 25.000.000 VNĐ**.
- **Worst metric:** faithfulness.
- **Error Tree:** Output sai → Context đúng (chính sách hoàn chi phí đào tạo) → nhưng LLM suy luận điều kiện "8 tháng < 1 năm → 100%" sai/không bám context → faithfulness = 0.
- **Root cause:** Reasoning điều kiện thời hạn cam kết; mô hình flash-lite yếu ở suy luận nhiều bước.
- **Suggested fix:** Prompt few-shot ví dụ tính hoàn trả; hoặc model mạnh hơn cho câu numeric/điều kiện.

### #5 — avg 0.696 · worst: context_recall = 0.00
- **Question:** "Bao lâu phải đổi mật khẩu một lần?"
- **Expected:** Theo v2.0 hiện hành: **120 ngày** (v1.0 cũ 90 ngày đã bị thay thế).
- **Worst metric:** context_recall (thiếu chunk đúng).
- **Error Tree:** Output có thể sai → Context **thiếu** chunk chính sách v2.0 (lấy nhầm `mat_khau_v1.md` cũ) → recall = 0.
- **Root cause:** **Version conflict** — corpus có 2 bản (v1/v2); retrieval không phân biệt bản hiện hành.
- **Suggested fix:** Metadata `version`/`effective_date`, filter loại tài liệu superseded; hoặc enrichment đánh dấu "đã thay thế".

---

## Case Study (cho presentation)

**Question chọn phân tích:** #5 "Bao lâu phải đổi mật khẩu một lần?" — điển hình lỗi **version conflict**.

**Error Tree walkthrough:**
1. Output đúng? → Không chắc, dễ trả 90 ngày (bản cũ).
2. Context đúng? → **Không** — kéo nhầm `mat_khau_v1.md` (90 ngày) thay vì `mat_khau_v2.md` (120 ngày). context_recall = 0.
3. Query rewrite OK? → Query rõ ràng; vấn đề ở **ranking giữa 2 phiên bản gần như trùng ngữ nghĩa**.
4. Fix ở bước: **Indexing/Metadata** — gắn `version` + `is_current`, filter bỏ bản superseded trước khi rerank.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thêm metadata `version`/`effective_date` cho toàn corpus và filter `is_current=true` (giải quyết cả nhóm câu version: mật khẩu, nghỉ phép, MFA).
- Nhóm lỗi numeric (#3, #4): prompt few-shot + temperature=0 để giảm hallucination số liệu.
