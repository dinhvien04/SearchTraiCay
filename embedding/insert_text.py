import os
import pickle
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

def load_text_vectors(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def main():
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "fruit_text"

    # Tạo collection mới "fruit_text" với vector size = 384 (đúng với output của paraphrase-multilingual-MiniLM-L12-v2
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✅ Đã tạo collection '{collection_name}' với vector size=384")

    # Tải vector từ file pickle
    text_vectors = load_text_vectors("data/vectors/text_vectors.pkl")

    points = []
    for item in text_vectors:
        # Chuẩn hóa ID
        raw_id = item.get("id") or item.get("fruit_id")
        try:
            item_id = int(raw_id)
        except:
            item_id = str(raw_id)

        # Chuẩn hóa vector
        vector = item["vector"].tolist() if hasattr(item["vector"], "tolist") else item["vector"]

        # Chuẩn hóa payload
        payload = item.get("payload", {})
        payload.setdefault("fruit_id", item_id)
        payload.setdefault("name", "")
        payload.setdefault("description", "")
        payload.setdefault("keywords", "")
        payload.setdefault("image_url", "")
        payload.setdefault("category", "")
        payload.setdefault("origin", "")
        payload.setdefault("season", "")
        payload.setdefault("color", "")

        # Tạo PointStruct
        point = PointStruct(id=item_id, vector=vector, payload=payload)
        points.append(point)

    # Upsert vào collection
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Đã upsert {len(points)} vector mô tả vào collection '{collection_name}'")

if __name__ == "__main__":
    main()
