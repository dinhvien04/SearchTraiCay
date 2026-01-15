#Chuyển đổi các ảnh (fruit images) thành vector embedding dùng mạng nơ-ron EfficientNet (được train sẵn).
import os
import torch
import json
import pickle
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from torchvision.models import efficientnet_b0

#•	Tải mạng EfficientNet-B0 đã được train sẵn trên ImageNet.
def load_model():
    model = efficientnet_b0(pretrained=True)
    model.classifier = torch.nn.Identity()  # Bỏ lớp phân loại để lấy embedding
    model.eval()
    return model #tính toán embedding ảnh

#chuyển đổi ảnh thành vector, •	Trả về vector dạng numpy array 1 chiều.
def image_to_vec(image_path, model, transform):
    try:
        image = Image.open(image_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            vec = model(img_tensor)
        return vec.squeeze().numpy()
    except (FileNotFoundError, UnidentifiedImageError):
        print(f"❌ Không thể xử lý ảnh: {image_path}")
        return None
#•	Kiểm tra từng bản ghi trong JSON có đủ các trường bắt buộc 
def check_required_fields(row, required_fields):
    missing = [f for f in required_fields if not row.get(f)]
    if missing:
        print(f"⚠️ Bỏ qua dòng thiếu trường bắt buộc {missing}: {row}")
        return False
    return True
#đọc file json, transform cho ảnh, Tải model EfficientNet.
# khởi tạo dict lưu kết quả vector

def process_images_from_json(json_path, static_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    model = load_model()
    vectors = {}

    required_fields = ["fruit_id", "name", "description", "keywords", "image_url", "category", "origin", "color", "season"]

    for row in data:
        # Kiểm tra các trường bắt buộc
        if not check_required_fields(row, required_fields):
            continue

        image_filename = os.path.basename(row["image_url"])
        image_path = os.path.join(static_dir, image_filename)

        vec = image_to_vec(image_path, model, transform)
        if vec is not None:
            fruit_id = row["fruit_id"]
            #list vectors là một list chứa nhiều dict, mỗi dict lưu vector và thông tin của một loại trái cây
            #dict là kiểu dl bst các cặp key-value dùng để lưu dữ liệu có cấu trúc, ví dụ: thông tin một loại trái cây gồm tên, mô tả, id, màu sắc, mùa vụ, ...
            vectors[fruit_id] = {
                "vector": vec,
                "name": row["name"],
                "description": row["description"],
                "keywords": row["keywords"],
                "image_url": row["image_url"],
                "category": row["category"],
                "origin": row["origin"],
                "color": row["color"],
                "season": row["season"]
            }

    return vectors

if __name__ == "__main__":
    json_input = "data/metadata/fruit_metadata.json"   # Đường dẫn file JSON đầu vào
    static_image_dir = "static/images"                 # Thư mục chứa ảnh
    output_vector_file = "data/vectors/image_vectors.pkl"  # File lưu vector ảnh

    os.makedirs(os.path.dirname(output_vector_file), exist_ok=True)
    print("🚀 Bắt đầu sinh vector ảnh...")

    image_vectors = process_images_from_json(json_input, static_image_dir)

    with open(output_vector_file, "wb") as f:
        pickle.dump(image_vectors, f)

    print(f"✅ Đã lưu {len(image_vectors)} vector ảnh vào: {output_vector_file}")
