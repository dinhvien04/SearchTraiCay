# 🍎 FruitGo - Hệ thống tìm kiếm trái cây Việt Nam

Ứng dụng web sử dụng AI để tìm kiếm và nhận diện trái cây Việt Nam thông qua từ khóa/mô tả hoặc hình ảnh.

## ✨ Tính năng

- **Tìm kiếm bằng từ khóa/mô tả**: Nhập tên, đặc điểm, mô tả trái cây để tìm kiếm (semantic search)
- **Tìm kiếm bằng hình ảnh**: Upload ảnh trái cây, hệ thống sẽ nhận diện và trả về kết quả tương tự
- **Bộ lọc nâng cao**: Lọc theo màu sắc, mùa vụ, nguồn gốc
- **So sánh trái cây**: So sánh thông tin chi tiết giữa 2 loại trái cây
- **Gợi ý tương tự**: Xem các loại trái cây có đặc điểm tương tự

## 🛠️ Công nghệ sử dụng

- **Backend**: Flask (Python)
- **Vector Database**: Qdrant
- **Text Embedding**: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Image Embedding**: EfficientNet-B0 (PyTorch)
- **Frontend**: HTML, CSS, Jinja2

## 📦 Cài đặt

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

### 4. Khởi động Qdrant (Docker)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Tạo vector và insert vào database

```bash
# Set PYTHONPATH
$env:PYTHONPATH = "."  # PowerShell
# export PYTHONPATH="."  # Linux/Mac

# Insert text vectors
python qdrant_utils/insert_text_vectors.py

# Insert image vectors
python qdrant_utils/insert_image_vectors.py
```

### 6. Chạy ứng dụng

```bash
python app.py
```

Truy cập: http://localhost:5000

## 📁 Cấu trúc thư mục

```
SearchTraiCay/
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── data/
│   └── metadata/
│       ├── fruit_metadata.csv
│       └── fruit_metadata.json
├── embedding/
│   ├── generate_text_vec.py  # Tạo text embeddings
│   └── generate_image_vec.py # Tạo image embeddings
├── qdrant_utils/
│   ├── connect.py            # Kết nối Qdrant
│   ├── insert_text_vectors.py
│   ├── insert_image_vectors.py
│   ├── search_text_vectors.py
│   └── search_image_vectors.py
├── static/
│   └── images/               # Ảnh trái cây
├── templates/
│   ├── layout.html
│   ├── home.html
│   ├── search_text.html
│   ├── search_image.html
│   ├── fruit_detail.html
│   └── compare.html
└── README.md
```

## 📊 Dataset

- **358 loại trái cây** Việt Nam
- Thông tin bao gồm: tên, mô tả, đặc điểm, nguồn gốc, màu sắc, mùa vụ, category
- Hình ảnh minh họa cho từng loại

## 🔍 Cách hoạt động

### Tìm kiếm văn bản (Semantic Search)
1. Người dùng nhập từ khóa/mô tả
2. Text được chuyển thành vector embedding bằng SentenceTransformers
3. Tìm kiếm vector tương đồng trong Qdrant
4. Trả về kết quả có độ tương đồng cao nhất

### Tìm kiếm hình ảnh
1. Người dùng upload ảnh trái cây
2. Ảnh được chuyển thành vector embedding bằng EfficientNet-B0
3. Tìm kiếm vector tương đồng trong collection ảnh
4. Trả về top 5 kết quả giống nhất

## 📝 License

MIT License

## 👤 Tác giả

- GitHub: [@dinhvien04](https://github.com/dinhvien04)
