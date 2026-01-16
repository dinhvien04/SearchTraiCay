# Orchard - Hệ thống tìm kiếm trái cây Việt Nam

Ứng dụng web sử dụng AI để tìm kiếm và nhận diện trái cây Việt Nam thông qua từ khóa, hình ảnh hoặc giọng nói.

## Tính năng

- Tìm kiếm bằng từ khóa/mô tả (semantic search)
- Tìm kiếm bằng hình ảnh (image similarity)
- Tìm kiếm bằng giọng nói (tiếng Việt)
- Bộ lọc theo màu sắc, mùa vụ, nguồn gốc
- So sánh thông tin giữa 2 loại trái cây
- Gợi ý trái cây tương tự
- Chatbot AI hỏi đáp (RAG + LLM)

## Công nghệ

- Backend: Flask (Python)
- Vector Database: Qdrant
- Text Embedding: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
- Image Embedding: EfficientNet-B0 (PyTorch)
- LLM: MegaLLM API
- Frontend: HTML, CSS, JavaScript

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/dinhvien04/SearchTraiCay.git
cd SearchTraiCay
```

### 2. Tạo virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình environment

Tạo file `.env`:

```env
MEGALLM_API_KEY=your_api_key_here
MEGALLM_MODEL=gpt-5-mini
```

### 5. Khởi động Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 6. Insert dữ liệu vào database

```bash
$env:PYTHONPATH = "."  # PowerShell
python qdrant_utils/insert_text_vectors.py
python qdrant_utils/insert_image_vectors.py
```

### 7. Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

## Cấu trúc thư mục

```
├── app.py                     # Flask application
├── requirements.txt
├── .env
├── data/
│   └── metadata/
│       ├── fruit_metadata.csv
│       └── fruit_metadata.json
├── embedding/
│   ├── generate_text_vec.py
│   └── generate_image_vec.py
├── qdrant_utils/
│   ├── connect.py
│   ├── insert_text_vectors.py
│   ├── insert_image_vectors.py
│   ├── search_text_vectors.py
│   └── search_image_vectors.py
├── static/images/
└── templates/
    ├── layout.html
    ├── home.html
    ├── search_text.html
    ├── search_image.html
    ├── fruit_detail.html
    ├── compare.html
    └── chatbot.html
```

## Cách hoạt động

### Tìm kiếm văn bản
Text được chuyển thành vector embedding bằng SentenceTransformers, sau đó tìm kiếm vector tương đồng trong Qdrant.

### Tìm kiếm hình ảnh
Ảnh được chuyển thành vector đặc trưng bằng EfficientNet-B0, so sánh với các vector ảnh trong database để tìm kết quả tương tự.

### Chatbot (RAG)
Tìm kiếm thông tin liên quan từ database, gửi context kèm câu hỏi cho LLM để tạo câu trả lời.

## Lưu ý

Tìm kiếm ảnh hoạt động tốt nhất với ảnh quả trái cây đơn lẻ, rõ ràng. Ảnh chụp cả cây hoặc có nhiều đối tượng có thể cho kết quả không chính xác do model so sánh đặc trưng hình ảnh tổng thể.

## License

MIT License

## Tác giả

GitHub: [@dinhvien04](https://github.com/dinhvien04)
