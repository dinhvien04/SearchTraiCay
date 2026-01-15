#chuyển đổi ảnh thành vector embedding bằng mô hình EfficientNet_B0,
#  sau đó lưu vector này kèm thông tin mô tả (payload) vào Qdrant
#Chèn vector đặc trưng của ảnh vào hệ thống
import os
import json
import pickle
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from qdrant_utils.connect import connect_qdrant
from qdrant_utils.insert import create_collection, insert_vectors

#: lấy embedding vector của ảnh chứ không lấy kết quả phân loại).
def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()  # Loại bỏ lớp phân loại cuối
    model.eval()
    return model
#o	Chuyển mô hình sang chế độ đánh giá (eval()) để không cập nhật trọng số, giảm dùng bộ nhớ GPU.
#mô hình này khi input ảnh sẽ trả về vector embedding kích thước 1280.
def image_to_vec(image_path, model, transform):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Lỗi khi mở ảnh {image_path}: {e}")
        return None
#o	Dùng transform (định nghĩa bên dưới) để resize và chuẩn hóa ảnh thành tensor chuẩn đầu vào model.
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = model(tensor)
    return vec.squeeze().numpy()


def main():
    print("🚀 Bắt đầu sinh vector ảnh để chèn vào Qdrant...")
    #o	Resize ảnh về kích thước 224x224 px 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    model = load_model()
    client = connect_qdrant()

    # Đọc metadata json
    metadata_path = "data/metadata/fruit_metadata.json"
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    vector_data = []
#•	Lặp qua từng ảnh trong file JSON.
    print(f"🔎 Tổng số ảnh cần xử lý: {len(data)}")
    for idx, item in enumerate(data):
        image_url = item.get("image_url")
        if not image_url:
            print(f"⚠️ Mục không có image_url, bỏ qua: {item}")
            continue

        print(f"🖼️ [{idx+1}/{len(data)}] Đang xử lý ảnh: {image_url}")

        # Chuyển URL thành đường dẫn file local nếu cần
        local_path = image_url.replace("http://localhost:5000", ".")
        if not os.path.exists(local_path):
            print(f"⚠️ Không tìm thấy ảnh: {local_path}, bỏ qua")
            continue
#•	Gọi hàm chuyển ảnh thành vector embedding.
        vec = image_to_vec(local_path, model, transform)
        if vec is None:
            continue

        # Lấy id an toàn, ưu tiên fruit_id, nếu không có thì id
        raw_id = item.get("fruit_id") or item.get("id")
        if raw_id is None:
            print(f"⚠️ Mục không có id hoặc fruit_id, bỏ qua: {item}")
            continue

        try:
            fruit_id = int(raw_id)
        except Exception:
            print(f"⚠️ Không thể chuyển id: {raw_id} sang int, bỏ qua")
            continue

#•	Tạo payload (dữ liệu mô tả đi kèm với vector) để sau này dùng tìm kiếm có thể trả về thông tin chi tiết.
        payload = {
    "fruit_id": fruit_id,
    "name": item.get("name", ""),
    "description": item.get("description", ""),
    "keywords": item.get("keywords", ""),
    "image_url": image_url,
    "origin": item.get("origin", ""),
    "season": item.get("season", ""),
    "color": item.get("color", ""),
    "category": item.get("category", ""),
    "type": "image"
}

#•	vector_data sẽ lưu trữ danh sách tuple (vector_embedding, payload_info).
        vector_data.append((vec, payload))

    if not vector_data:
        print("❌ Không có vector nào được sinh ra, thoát.")
        return

    # Tạo hoặc ghi đè collection
    vector_size = len(vector_data[0][0])
    create_collection(client, "fruit_image", vector_size=vector_size)

    # Chèn vectors và payloads vào Qdrant
    vectors = [v for v, _ in vector_data]
    payloads = [p for _, p in vector_data]
    insert_vectors(client, "fruit_image", vectors, payloads)

    # Lưu vectors + payloads để dùng lại
#•	Tách riêng vector embedding và payload thành 2 list.
#Gọi hàm insert_vectors để thêm toàn bộ vector và payload vào collection "fruit_image".

    os.makedirs("data/vectors", exist_ok=True)
    with open("data/vectors/image_vectors.pkl", "wb") as f:
        pickle.dump(vector_data, f)

    print("💾 Đã lưu vector ảnh vào: data/vectors/image_vectors.pkl")
    print("✅ Đã chèn vector ảnh mới vào collection 'fruit_image'.")


if __name__ == "__main__":
    main()
