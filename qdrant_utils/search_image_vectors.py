#load model chuyển ảnh thành vector đặc trưng
#dùng qdrant để tìm kiếm ảnh có vector tương tự
#cho phép tìm kiếm ảnh dựa trên ảnh query.
from qdrant_client import QdrantClient 
from qdrant_utils.connect import connect_qdrant
from PIL import Image
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os


class ImageVectorSearch:
    def __init__(self, device='cpu'):
        # Khởi tạo model EfficientNet-B0 và transform preprocess một lần
        self.device = device
        self.weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=self.weights).to(self.device) #chạy trên cpu hoặc gpu
        self.model.eval() #chuyển model sang chế độ đánh giá (inference)
        self.preprocess = self.weights.transforms() #pipeline transform ảnh chuẩn hóa, resize, tensor hóa theo model.

#chuyển ảnh thành vector đặc trưng
    def get_image_vector(self, image_path: str) -> list[float]: #kiểm tra file ảnh
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File ảnh không tồn tại: {image_path}")

        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # shape [1,3,224,224]

        with torch.no_grad():
            features = self.model.features(input_tensor)
            vector = torch.nn.functional.adaptive_avg_pool2d(features, 1).squeeze().cpu().numpy()
        return vector.tolist() #•	Đưa về CPU và chuyển thành numpy array rồi thành list.

#hàm tìm kiếm ảnh tương tự
    def search_similar_images(self, collection_name: str, query_image_path: str, top_k: int = 1):
        client = connect_qdrant()
        query_vector = self.get_image_vector(query_image_path)

        try:
            results = client.search(
                collection_name=collection_name, 
                query_vector=query_vector, #vector ảnh query để so sánh (vector truy vấn dạng số hóa)
                limit=top_k,
                with_payload=True  # chắc chắn lấy payload
            )
            return results
        except Exception as e:
            print(f"❌ Lỗi khi tìm kiếm ảnh tương tự: {e}")
            return []


if __name__ == "__main__":
    searcher = ImageVectorSearch()
    query_img = "static/images/xoai_cat_chu.jpg"  # chỉnh đường dẫn phù hợp

    try:
        hits = searcher.search_similar_images("fruit_image", query_img, top_k=1)
        print("✅ Kết quả tìm ảnh tương tự:")
        for hit in hits:
            print(f"ID: {hit.id}, Score: {hit.score:.4f}, Payload: {hit.payload}")
    except FileNotFoundError as fnf_err:
        print(f"❌ Lỗi: {fnf_err}")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")
 #Mỗi ảnh được biểu diễn dưới dạng vector số thực có kích thước cố định
 # (ở đây là kích thước output của EfficientNet-B0, khoảng 1280 chiều). 
 # Vector này tóm tắt đặc trưng của ảnh về mặt hình ảnh, màu sắc, hình dạng,...
#Mục đích là để so sánh ảnh bằng khoảng cách hoặc cosine similarity giữa các vector.
