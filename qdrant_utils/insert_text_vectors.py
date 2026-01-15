#Chèn vector đặc trưng của văn bản mô tả (text) vào hệ thống.
import json
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Distance

from qdrant_utils.connect import connect_qdrant
from qdrant_utils.insert import create_collection, insert_vectors

def main():
    # Kết nối tới Qdrant server
    try:
        print("⏳ Đang kết nối Qdrant...")
        client = connect_qdrant()
        print("✅ Kết nối Qdrant thành công.")
    except Exception as e:
        print("❌ Lỗi kết nối Qdrant:", e)
        sys.exit(1)

    # Đọc file metadata JSON
    try:
        with open("data/metadata/fruit_metadata.json", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Đã đọc file metadata với {len(data)} mục.")
    except Exception as e:
        print("❌ Lỗi đọc file metadata:", e)
        sys.exit(1)

    # Tải mô hình SentenceTransformer
    print("⏳ Đang tải mô hình SentenceTransformer...")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("✅ Mô hình đã sẵn sàng.")
    #Khởi tạo list chứa dữ liệu
    vectors = [] #chứa các vector embedding (list các list số thực).
    ids = [] 
    payloads = [] 
#•	Với mỗi item (trái cây), lấy 2 trường text để tạo vector: description + keyword
    for item in data:
        text = (item.get("description") or "") + " " + (item.get("keywords") or "")
        text = text.strip()
        if not text:
            print(f"⚠️ Mục id={item.get('id', item.get('fruit_id', 'unknown'))} không có nội dung text để encode, bỏ qua.")
            continue

        # Ưu tiên lấy fruit_id, nếu không có lấy id
        raw_id = item.get("fruit_id") or item.get("id")
        if raw_id is None:
            print(f"⚠️ Mục không có id hoặc fruit_id, bỏ qua: {item}")
            continue
# chuyển id về int hoặc str
#qdrant yêu cầu id phải là chuỗi, thống nhất, hạn chế lỗi
        try:
            item_id = int(raw_id)
        except Exception:
            if raw_id == "" or raw_id is None:
                print(f"⚠️ id không hợp lệ (rỗng hoặc None), bỏ qua: {raw_id}")
                continue
            else:
                item_id = str(raw_id)

        if item_id is None or item_id == "":
            print(f"⚠️ id chuyển đổi thành None hoặc rỗng, bỏ qua: {raw_id}")
            continue

        # Encode văn bản thành vector: biến text thành vector số.
        try:
            vec = model.encode(text)
        except Exception as e:
            print(f"⚠️ Lỗi encode văn bản cho id={item_id}: {e}")
            continue

        vectors.append(vec.tolist())
        ids.append(item_id)

        # Chuẩn bị payload kèm theo vector
        payload = {
    "fruit_id": item_id,
    "name": item.get("name", ""),
    "type": "text",
    "description": item.get("description", ""),
    "keywords": item.get("keywords", ""),
    "image_url": item.get("image_url", ""),
    "origin": item.get("origin", ""),
    "season": item.get("season", ""),
    "color": item.get("color", "")
}
        if "category" in item:
            payload["category"] = item["category"]
        if "tags" in item:
            payload["tags"] = item["tags"]
        payloads.append(payload)

    if not vectors:
        print("❌ Không có vector nào được tạo, dừng chương trình.")
        sys.exit(1)

    print(f"✅ Tạo được {len(vectors)} vector embedding.")

    # Tạo collection Qdrant (nếu chưa có)
    print(f"⏳ Đang tạo collection 'fruit_text' với kích thước vector {len(vectors[0])} và distance metric 'Cosine'...")
    create_collection(client, "fruit_text", vector_size=len(vectors[0]), distance=Distance.COSINE)
    print("✅ Collection đã được tạo.")

    # Chèn vector vào collection
    print(f"⏳ Đang chèn {len(vectors)} vector vào collection...")
    insert_vectors(client, "fruit_text", vectors, payloads, ids)
    print("✅ Đã chèn vector text vào collection 'fruit_text'.")

if __name__ == "__main__":
    main()
