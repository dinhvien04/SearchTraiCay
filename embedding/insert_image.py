#Tải dữ liệu vector ảnh trái cây từ file .pkl và đưa vào cơ sở dữ liệu vector Qdrant (tên collection: fruit_image).
import pickle
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

# Load vector từ file pickle
with open("data/vectors/image_vectors.pkl", "rb") as f:
    data = pickle.load(f)

print("📦 Kiểu dữ liệu vector:", type(data))
if isinstance(data, list):
    print(f"➡️ Dữ liệu dạng list, số phần tử: {len(data)}")
elif isinstance(data, dict):
    print(f"➡️ Dữ liệu dạng dict, số keys: {len(data)}")
else:
    raise Exception("⚠️ Dữ liệu không đúng định dạng list hoặc dict")

#•	Collection tên "fruit_image" dùng để lưu vector ảnh
client = QdrantClient("localhost", port=6333)
collection_name = "fruit_image"

# Tạo collection nếu chưa tồn tại
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1280, distance=Distance.COSINE)
    )

points = []

#Chuẩn bị dữ liệu dạng PointStruct để upsert
def safe_convert_fruit_id(fid_raw):
    if isinstance(fid_raw, (list, tuple)):
        fid_raw = fid_raw[0]
    if hasattr(fid_raw, "item"):
        fid_raw = fid_raw.item()
    try:
        return int(fid_raw)
    except Exception as e:
        print(f"⚠️ Lỗi chuyển fruit_id sang int: {fid_raw} - {e}")
        return None

if isinstance(data, list):
    for entry in data:
        fruit_id = safe_convert_fruit_id(entry.get("fruit_id", None))
        if fruit_id is None:
            print("⚠️ Bỏ qua phần tử không có fruit_id hợp lệ:", entry)
            continue

        vector = entry.get("vector")
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        points.append(
            PointStruct(
                id=fruit_id,
                vector=vector,
                payload={
                    "name": entry.get("name", ""),
                    "image_url": entry.get("image_url", ""),
                    "origin": entry.get("origin", ""),
                    "color": entry.get("color", ""),
                    "season": entry.get("season", ""),
                    "category": entry.get("category", "")
                }
            )
        )
else:
    for fruit_id_str, entry in data.items():
        try:
            fruit_id = int(fruit_id_str)
        except Exception as e:
            print(f"⚠️ Lỗi chuyển fruit_id sang int: {fruit_id_str} - {e}")
            continue

        vector = entry.get("vector")
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        points.append(
            PointStruct(
                id=fruit_id,
                vector=vector,
                payload={
                    "name": entry.get("name", ""),
                    "image_url": entry.get("image_url", ""),
                    "origin": entry.get("origin", ""),
                    "color": entry.get("color", ""),
                    "season": entry.get("season", ""),
                    "category": entry.get("category", "")
                }
            )
        )

try:
#Thêm hoặc cập nhật điểm vector vào collection.
#In ra số lượng điểm đã chèn thành công.

    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Đã thêm {len(points)} vector vào collection '{collection_name}'")
except Exception as e:
    print(f"❌ Lỗi khi upsert vector vào Qdrant: {e}")

#Mỗi điểm chứa: id: fruit_id duy nhất, vector: vector ảnh, payload: các trường thông tin bổ sung như name, image_url, origin, color, season, category.
