# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Tống Duy An (2A202601995)
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Môi trường thực thi:** Local (Windows 11, Python 3.13 venv riêng) + Neo4j 5.26 chạy Docker container (`bolt://localhost:7687`). Extraction/Generation: OpenAI `gpt-4o-mini`. LLM-as-a-Judge: Groq `openai/gpt-oss-120b` (khác provider với generator để tránh self-preference bias).

**Quy mô dữ liệu thực tế:** 5.000 dòng đầu của `HackerNoon/tech-company-news-data-dump` (đúng scope của bộ Golden 50 câu) → lọc + exact dedup còn 2.119 bài → chọn 1.500 bài (giữ toàn bộ 51 bài chứa evidence của Golden) → 1.500 chunks → 400 chunks qua LLM extraction. Dataset thực tế chỉ có cột `description` (~17 từ/bài) thay vì full text, nên mỗi bài tạo 1 chunk (title được ghép vào text để tăng ngữ cảnh).

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)

- **Ví dụ từ dữ liệu:** chunk `5deb3381f482aefd657f::c0000`:
  > *"HPE to offer cloud computing service for artificial intelligence. HPE said **the company** will use its experience in supercomputers…"*
- **Hiện tượng:** Mô hình coref (chạy với conservative rule) đã resolve được "HPE said" nhưng **từ chối resolve "the company"** và ghi nó vào `unresolved_mentions` — dù người đọc thấy khá rõ đó là HPE. Tổng cộng 35/400 chunks có mention không resolve được (`it`, `we`, `the company`…). Đây là hành vi **chủ đích**: prompt yêu cầu chỉ resolve khi tiền ngữ được hỗ trợ rõ ràng trong cùng chunk.
- **Hậu quả đối với Graph nếu resolve sai:** Nếu chunk nói về 2 công ty (ví dụ tin M&A nhắc cả HPE và đối thủ), "the company" gán nhầm sẽ tạo **False Edge** — ví dụ gán nhầm quan hệ `DEVELOPED`/`ACQUIRED` cho đối thủ cạnh tranh. Cạnh sai trong đồ thị nguy hiểm hơn cạnh thiếu, vì GraphRAG sau đó trả lời sai kèm trích dẫn provenance trông rất đáng tin. Trade-off của conservative rule: chấp nhận mất một phần recall (bài giảng ước tính coref đóng góp 30–40% quan hệ) để giữ precision của đồ thị.

---

### 2. Entity Resolution Threshold & Lexical Guard

- **Ngưỡng cosine similarity:** `threshold = 0.90` (FAISS ANN trên embedding MiniLM-L6-v2 của tên thực thể, top-k = 5). Guard chuỗi: bỏ hậu tố công ty (Inc/Corp/Ltd…) rồi yêu cầu `SequenceMatcher.ratio() >= 0.72`. Bảng audit ghi cả các cặp 0.80–0.90 (`REJECT_THRESHOLD`) để minh bạch — tổng cộng **12 dòng audit**: 4 `MERGE_VECTOR`, 1 `REJECT_GUARD`, 7 `REJECT_THRESHOLD`.
- **Cặp bị Guard chặn (similarity > 0.85):** `Houston` vs `Houston Texas` — **similarity 0.945** nhưng `REJECT_GUARD`.
- **Lý do chặn:** Sau khi bỏ hậu tố, `SequenceMatcher("houston", "houston texas")` = 0.70 < 0.72 → guard từ chối. Điểm thú vị kép của ca này: (a) về ngữ nghĩa hai mention này *có thể* cùng chỉ một địa danh, nhưng cả hai đều bị NER gán nhầm type `Company` (lỗi upstream) — việc guard chặn gộp vô tình ngăn lỗi extraction lan rộng; (b) nó minh họa đúng nguyên tắc thiết kế: **vector similarity đề cử, lexical guard phủ quyết** — embedding rất dễ cho điểm cao với các cặp "cùng ngữ cảnh nhưng khác thực thể" (Sam Altman/Steve Altman, Apple/Apple Watch), tầng chữ là phanh an toàn cuối trước khi Union-Find gộp không thể đảo ngược.
- **Ví dụ merge đúng:** `L&T Technology Services Limited` → `L&T Technology Services` (0.926, `MERGE_VECTOR`), `Fidelity National Information Services Inc.` → `Fidelity National Information Services` (0.925).

---

### 3. Đồ thị & Super-node Mitigation

Đồ thị cuối: **382 nodes** (236 Company, 105 Technology, 41 Person), **251 edges**, phân bố quan hệ: PARTNERED_WITH 76, DEVELOPED 57, WORKED_AT 36, ACQUIRED 22, USES 22, FOUNDED 14, INVESTED_IN 12, LEADS 10. Sanity check: **0 cạnh thiếu `source_chunk_id`/`published_date`**.

- **Top 3 Super-nodes (bậc cao nhất):**

| Hạng | Tên thực thể | Loại | Degree |
|------|--------------|------|--------|
| 1 | Microsoft | Company | 17 |
| 2 | L&T Technology Services | Company | 7 |
| 3 | ServiceNow | Company | 6 |

- **Lưu ý trung thực về scale:** Ở quy mô lab (400 chunks extraction), degree cao nhất chỉ 17 — **chính sách cap (degree > 100 → 50 cạnh mới nhất) chưa từng kích hoạt trên dữ liệu thật** (0 supernode_events trong toàn bộ evaluation). Để chứng minh cơ chế hoạt động đúng, notebook có **synthetic test**: tạo hub giả 120 cạnh trong Neo4j → xác nhận `degree=120, fetched=50` (đúng cap) → dọn sạch. Ở scale 350MB, Microsoft/Google chắc chắn vượt 100 và chính sách này trở thành bắt buộc.
- **Ưu điểm của Temporal Mitigation (lấy 50 cạnh `published_date` mới nhất):** chặn bùng nổ context window và chi phí BFS; với tin tức công nghệ, fact mới thường có giá trị trả lời cao hơn (tin cập nhật thay thế tin cũ); kết hợp trần toàn cục `GLOBAL_EDGE_CAP=250` và `MAX_GRAPH_CONTEXT_CHARS=14000` tạo 3 tầng phanh.
- **Rủi ro:** câu hỏi lịch sử ("ai sáng lập X năm 2010?") có thể bị cắt mất cạnh cũ; bias hệ thống nghiêng về sự kiện gần đây; cạnh thiếu `published_date` (sort về cuối) gần như không bao giờ được chọn dù có thể quan trọng. Giải pháp sản xuất: cap theo *relevance score* (kết hợp độ mới + confidence + match với query) thay vì chỉ độ mới.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

Đánh giá trên **12 câu Golden** (4 factoid + 4 multi-hop + 4 cross-doc, chọn từ bộ 50 câu của giảng viên theo evidence-coverage, tất cả đạt coverage 100%). Judge: Groq `gpt-oss-120b`, temperature 0, chấm 1–5.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge)

| Tiêu chí | Flat RAG | GraphRAG | Δ | Nhận xét phân tích |
|---|---|---|---|---|
| **Comprehensiveness — factoid** | 5.00 | 5.00 | 0 | Hoà. Đáp án nằm gọn 1 chunk, vector search đủ |
| **Comprehensiveness — multi-hop** | 3.00 | **4.00** | **+1.0** | GraphRAG gom đủ thực thể qua traversal |
| **Comprehensiveness — cross-doc** | 5.00 | 5.00 | 0 | Hoà ở scale này (evidence chỉ 2–4 bài/câu) |
| **Faithfulness — multi-hop** | 3.00 | **4.00** | **+1.0** | Provenance trên cạnh giúp bám evidence |
| **Multi-hop Reasoning — multi-hop** | 3.00 | **4.00** | **+1.0** | Đúng kỳ vọng lý thuyết: quan hệ nằm ở cấu trúc |
| **Latency trung bình (s)** | 3.69 | 2.32 | −1.37 | Xem ghi chú đo lường bên dưới |
| **Token usage trung bình** | 752 | 910 | **+21%** | Graph context (linearized edges) cộng thêm vào prompt |

*Ghi chú đo lường:* `latency_s` chỉ đo **call generation cuối** — chưa tính call seed-extraction LLM và các query Neo4j của GraphRAG (chi phí retrieval thật của GraphRAG cao hơn số trong bảng). GraphRAG generation lại nhanh hơn vì context có cấu trúc giúp model trả lời gọn hơn. Token +21% là chi phí thật phải trả cho graph context.

#### Phân tích 2 Ca lỗi Điển hình

**1. Flat RAG thất bại — GraphRAG thành công: `G5000-08` (multi-hop, Flat 1/1/1 vs Graph 5/5/5)**
- *Câu hỏi:* "Which external organizations are connected to ServiceNow's generative-AI efforts, and what distinct role does each play?"
- *Tại sao Flat RAG thất bại:* Top-6 chunk theo cosine similarity chỉ phủ NVIDIA và Deloitte — chunk về **AI Lighthouse (có Accenture)** không đủ "giống" câu hỏi về mặt embedding nên không lọt top-k. Judge: *"The candidate omits Accenture… not comprehensive."* Đây chính là ranh giới của similarity search: đủ để tìm "cái giống", không đủ để **liệt kê đầy đủ theo quan hệ**.
- *GraphRAG giải quyết thế nào:* Seed = `ServiceNow` → BFS 2-hop đi theo các cạnh `PARTNERED_WITH` quanh node ServiceNow → gom trọn NVIDIA + Accenture + Deloitte kèm evidence từng cạnh. Bài toán "liệt kê mọi X có quan hệ R với Y" là dạng truy vấn đồ thị thuần túy — traversal thắng tuyệt đối.

**2. Cả hai cùng thất bại: `G5000-06` (multi-hop temporal, Flat 1/1/1 vs Graph 1/1/1)**
- *Câu hỏi:* Trace tiến hoá generative-AI của ServiceNow từ tháng 5→7/2023 (đối tác tháng 5? tính năng tháng 6? chương trình tháng 7?).
- *Nguyên nhân:* Cả hai đều trả lời đúng mốc tháng 5 (NVIDIA) và tháng 7 (AI Lighthouse) nhưng **cùng trượt mốc tháng 6 — "Now Assist for Virtual Agent"**. Root cause nằm ở **tầng extraction**: schema 8 quan hệ không có loại quan hệ diễn tả "ra mắt tính năng cho khách hàng" đủ cụ thể; sự kiện tháng 6 hoặc không thành cạnh, hoặc thành cạnh `DEVELOPED` chung chung không mang ngữ nghĩa "customer-facing feature tháng 6". GraphRAG thậm chí trả lời "evidence insufficient" cho mốc này — từ chối trung thực nhưng judge vẫn trừ điểm thiếu ý. Đây là minh họa **GIGO**: graph không thể trả lời fact mà extraction chưa từng ghi vào.
- *Đề xuất khắc phục:* (a) mở rộng schema có kiểm soát (`LAUNCHED`, `RELEASED` với thuộc tính `event_date`); (b) với câu hỏi temporal, retrieval nên lọc/sort cạnh theo khoảng thời gian trong câu hỏi thay vì chỉ "mới nhất"; (c) tầng self-correction đã có sẵn trong bonus chính là chỗ cắm fix này (phát hiện thiếu mốc tháng 6 → mở rộng hop hoặc rơi về vector với query viết lại).

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

**Đánh đổi Quality vs Cost vs Latency (số đo thật của run):**
- *Indexing overhead:* Flat RAG index 1.500 chunks bằng MiniLM local mất ~1 phút, gần như miễn phí. GraphRAG phải đẩy 400 chunks qua LLM 2 lần (coref 8,7 phút + NER/RE 8,3 phút, ~167k tokens) rồi mới có đồ thị — **chi phí xây graph ≈ toàn bộ chi phí của Flat RAG nhân nhiều lần**, đúng cảnh báo "extraction bottleneck" của bài giảng.
- *Query time:* GraphRAG tốn thêm 1 call seed-extraction + các query Neo4j mỗi câu hỏi, và +21% token generation. Đổi lại multi-hop tăng +1.0 điểm trên cả 3 tiêu chí.
- *Kết luận vận hành:* factoid → Flat RAG (rẻ, đủ); multi-hop/liệt-kê-quan-hệ → GraphRAG xứng đáng chi phí; corpus dùng một lần → không bao giờ xây graph.

**Các quyết định từ chối/điều chỉnh đề xuất của AI Coding Agent trong buổi lab:**
1. **Từ chối stream dataset theo dung lượng MB (mặc định của notebook gốc):** golden dataset của giảng viên đánh scope "5.000 dòng đầu tiên", stream theo MB-priority sẽ lệch scope làm golden mất giá trị. Đổi sang stream đúng 5.000 dòng đầu theo thứ tự gốc.
2. **Chặn nguy cơ dùng ground truth sai mục đích:** quy tắc đặt ra là `reference_answer`/`reference_evidence` **chỉ được vào prompt của judge**, tuyệt đối không vào retrieval/generation. `evidence_row_ids` (metadata) chỉ dùng để scoping corpus — áp dụng **đồng đều cho cả hai hệ** nên so sánh A/B vẫn công bằng, và giới hạn này được khai báo minh bạch tại đây: kết quả benchmark phản ánh corpus đã được đảm bảo chứa evidence, không phản ánh khả năng tìm evidence trong corpus ngẫu nhiên.
3. **Đổi provider giữa chừng có kiểm soát:** Groq free tier chỉ 8.000 tokens/phút khiến pipeline ~70 phút chỉ để chờ throttle; quyết định chuyển extraction/generation sang OpenAI `gpt-4o-mini` (~$0.15 cho cả pipeline) và **chuyển judge sang Groq** để giữ nguyên tắc judge khác provider với generator.
4. **Yêu cầu bằng chứng thay vì tự nhận:** super-node cap không kích hoạt tự nhiên ở scale lab → bổ sung synthetic test (hub 120 cạnh) để cơ chế được *chứng minh* chứ không chỉ *được cài*.

**Giải pháp kiến trúc khi scale 350MB (~100.000 bài):**
- *Bottleneck 1 — LLM extraction* (nghiêm trọng nhất): 100k bài × ~2 call ≈ chi phí và thời gian không chạy tuần tự được. Giải pháp: hàng đợi async worker (batch API rẻ hơn ~50%), ưu tiên hoá tài liệu (extract trước tài liệu giàu entity), cache theo content-hash (đã làm trong lab: `data_local/*_cache.csv` — re-run miễn phí), model nhỏ cho coref + model lớn chỉ cho RE.
- *Bottleneck 2 — Entity Resolution:* số mention tăng, pairwise O(N²) chết trước tiên. Lab đã dùng đúng nền: ANN (FAISS) k-NN thay vì pairwise → giữ nguyên kiến trúc, thêm blocking theo type + ký tự đầu, và HNSW index thay FlatIP.
- *Bottleneck 3 — Super-node thật:* Microsoft/Google sẽ có hàng nghìn cạnh — cap 50 theo ngày là bắt buộc, nâng cấp thành cap theo relevance; thêm community detection (Leiden) để trả lời câu hỏi toàn cục bằng community summary thay vì BFS.
- *Hạ tầng:* Neo4j đơn node chịu được hàng triệu edge (không phải bottleneck ở 350MB); embedding chuyển sang GPU (RTX 4060 local) khi số chunk × 60.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module | Hàm / Khối code | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `run_coref()` | 35/400 chunks có unresolved mentions — mô hình thà bỏ qua còn hơn đoán; ví dụ "the company" của HPE bị giữ nguyên dù ngữ cảnh khá rõ. Đúng tinh thần "false coref → false edge" |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, filter trong `run_extraction()` | 400 chunks chỉ ra 252 triples — precision-over-recall hoạt động; mọi triple ngoài schema bị vứt im lặng. Đổi lại ca G5000-06 lộ đúng điểm mù: schema thiếu quan hệ "ra mắt sản phẩm" |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` (UNWIND, batch 1000) | Toàn bộ graph nạp trong 0,5s. Đối chứng: từng MERGE đơn lẻ sẽ tốn 251 round-trip |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `merge_guard()`, class `UF` | Audit 12 dòng; ca Houston vs Houston Texas (0.945) chứng minh guard chữ phủ quyết được vector |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `recent_edges()` (ORDER BY published_date DESC LIMIT) | Không kích hoạt tự nhiên (max degree 17); synthetic test 120 cạnh → fetch đúng 50 |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `run_evaluation()` | Judge khác provider với generator; temperature 0 + JSON mode cho điểm ổn định giữa 2 lần chạy (multi-hop 3.0/4.0 lặp lại y hệt) |

### 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật đáng nhớ nhất:** `CypherSyntaxError: Invalid input 'row'` khi chạy synthetic super-node test — nhìn query trong notebook thì hoàn toàn đúng. Root cause: đoạn code được patch vào notebook qua lệnh shell dùng chuỗi double-quote, **bash đã nuốt `$rows`** (biến shell rỗng) biến `UNWIND $rows AS row` thành `UNWIND  AS row`. Lỗi nằm ở tầng *công cụ đưa code vào file*, không nằm ở code. Xử lý: viết patch bằng file script Python (không đi qua shell string) và tìm cell theo nội dung thay vì theo index (papermill chèn cell làm lệch index).
- **Các blocker môi trường đã vượt:** dataset HackerNoon là gated repo (phải Agree trên HF trước); model `llama-3.3-70b-versatile` đã bị Groq gỡ (đổi sang `gpt-oss-120b`); dataset thật không có cột `text` — chỉ có `description` ngắn (patch loader ghép title + description); Groq free tier 8.000 TPM buộc viết throttle chủ động theo usage thay vì đợi lỗi 429.
- **Bài học:** (1) đo rate limit *trước* khi thiết kế pipeline, đừng để 429 dạy; (2) cache mọi output LLM trung gian theo content-hash — lần re-run thứ hai của lab tốn $0 phần extraction; (3) khi benchmark, kiểm soát đường đi của ground truth nghiêm ngặt như kiểm soát secrets.

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

*(Khung đề xuất — cần chốt lại khi xác định đồ án cuối khóa)*

- **Dạng bài toán phù hợp GraphRAG từ trải nghiệm lab:** hệ hỏi-đáp trên corpus tin tức/tài liệu doanh nghiệp tiếng Việt, nơi câu hỏi dạng "liệt kê mọi đối tác của X", "chuỗi sự kiện của Y theo thời gian" xuất hiện thường xuyên. Nếu đồ án chỉ là FAQ/tra cứu tài liệu đơn lẻ → Flat RAG hoặc Hybrid (BM25 + dense + rerank) là đủ, không xây graph.
- **Cấu trúc Node/Relation dự kiến (nếu làm QA tin tức doanh nghiệp VN):**
  - Nodes: `Company`, `Person`, `Product`, `Regulation` (đặc thù VN: nghị định/thông tư liên quan doanh nghiệp)
  - Relations: `ACQUIRED`, `INVESTED_IN`, `FOUNDED`, `LEADS`, `LAUNCHED` (có `event_date` — rút bài học từ ca G5000-06), `REGULATED_BY`
- **Chiến lược Entity Resolution:** tiếng Việt cần thêm normalize không dấu + alias map cho tên viết tắt phổ biến (VinGroup/VIC, FPT Corp/FPT); giữ kiến trúc ANN + lexical guard + Union-Find + audit table của lab.
- **Chiến lược Super-node:** các hub như "Chính phủ", "Ngân hàng Nhà nước" sẽ là super-node thật — cap theo relevance score (recency + confidence + độ khớp query) thay vì chỉ recency.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm chắc pipeline 5 bước, extraction bottleneck, trade-off Flat/Graph; cần trải nghiệm thêm Leiden/Global Search ở scale thật |
| Khả năng kiểm soát AI Coding Agent | 4 | 4 quyết định từ chối/điều chỉnh có căn cứ (scope dữ liệu, ground truth, provider, synthetic test); ghi lại minh bạch trong báo cáo |
| Chất lượng đồ thị tri thức xây dựng | 3.5 | 0 lỗi provenance, audit đầy đủ; nhưng 252 edges còn thưa do description ngắn — muốn dày hơn cần full-text corpus |
| Khả năng phân tích và debug hệ thống | 4 | Truy được root cause xuyên tầng (bash ăn `$rows`, schema thiếu quan hệ gây fail G5000-06, giới hạn phép đo latency) |
