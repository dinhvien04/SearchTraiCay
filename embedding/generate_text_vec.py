#xử lý phần embedding text (mô tả, keywords) cho từng quả.
import os
import json
import pickle
from sentence_transformers import SentenceTransformer

#đọc file json, trả về sd đối tượng dict
def load_metadata(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)

#load mô hình encode mô tả
def generate_text_vectors(metadata_list):
    print("🚀 Bắt đầu sinh vector mô tả...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    vectors = []

    for item in metadata_list:
        # Nối description và keywords để tạo text cho encode
        combined_text = (item.get('description', '') + " " + item.get('keywords', '')).strip()

        try:
            vec = model.encode(combined_text)
        except Exception as e:
            print(f"⚠️ Lỗi encode văn bản cho id={item.get('fruit_id') or item.get('id')}: {e}")
            continue

        # Chuẩn hóa fruit_id (ưu tiên fruit_id, fallback id)
        fruit_id = item.get("fruit_id") or item.get("id") or "unknown_id"
        try:
            fruit_id = int(fruit_id)
        except (ValueError, TypeError):
            fruit_id = str(fruit_id)

        # Tạo payload bắt buộc có đủ trường (dict)
        payload = {
            "fruit_id": fruit_id,
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "keywords": item.get("keywords", ""),
            "image_url": item.get("image_url", ""),
            "origin": item.get("origin", ""),
            "season": item.get("season", ""),
            "color": item.get("color", ""),
            "category": item.get("category", "")
        }

        vectors.append({
            "id": fruit_id,
            "vector": vec.tolist(),
            "payload": payload
        })

    return vectors

#•	Lưu list vector embedding vào file .pkl bằng pickle.
def save_vectors(vectors, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(vectors, f)
    print(f"✅ Đã lưu {len(vectors)} vector mô tả vào: {output_path}")

#khối main chạy file
if __name__ == "__main__":
    metadata = load_metadata("data/metadata/fruit_metadata.json")
    text_vectors = generate_text_vectors(metadata)
    save_vectors(text_vectors, "data/vectors/text_vectors.pkl")

#	Sử dụng mô hình SentenceTransformer nhỏ, hiệu quả và hỗ trợ đa ngôn ngữ.
#	Kết quả là một list vector embedding có cấu trúc rõ ràng, phù hợp để đẩy vào cơ sở dữ liệu vector hoặc dùng cho tìm kiếm tương tự (semantic search).
#	Kết quả lưu dưới dạng pickle để tái sử dụng.
