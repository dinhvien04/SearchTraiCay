from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_utils.connect import connect_qdrant
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchText

# Load model 1 lần duy nhất
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#nhận vào 1 câu text, mã hóa câu thành 1 vector số thực
def get_text_vector(text: str) -> list[float]:
    """Encode câu thành vector embedding."""
    return model.encode(text).tolist()

#tìm kiếm dựa trên câu truy vấn query text
def search_fruits_and_images(query_text: str, top_k_text: int = 400, top_k_images: int = 400): #số lượng kết quả trả về
    client = connect_qdrant()
    query_vector = get_text_vector(query_text)

    # Tạo filter tìm theo 4 trường metadata
    metadata_filter = Filter(
        should=[ #nếu 1 trong các điều kiện đúng thì phù hợp
            FieldCondition(key="season", match=MatchText(text=query_text)),
            FieldCondition(key="category", match=MatchText(text=query_text)),
            FieldCondition(key="origin", match=MatchText(text=query_text)),
            FieldCondition(key="color", match=MatchText(text=query_text)),
        ],
    )

    # Tìm kiếm vector kết hợp filter metadata
    results_text = client.search(
        collection_name="fruit_text", #chứa dữ liệu text trả về
        query_vector=query_vector,
        limit=top_k_text,
        with_payload=True,
        query_filter=metadata_filter
    )

    fruit_ids = [item.payload.get("fruit_id") for item in results_text if item.payload.get("fruit_id") is not None]

    if not fruit_ids:
        print("⚠️ Không tìm thấy fruit_id trong kết quả văn bản.")
        results_image = []
    else:
        # Lấy ảnh liên quan dựa trên fruit_id
        #dùng croll để tìm trong collection rồi lấy kèm payload
        scroll_result = client.scroll(
            collection_name="fruit_image",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="fruit_id",
                        match=models.MatchAny(any=fruit_ids)
                    )
                ]
            ),
            with_payload=True,
            limit=top_k_images
        )
        results_image = scroll_result[0] #lấy danh sách ảnh (mảng)

#ghép kết quả text và ảnh
    combined_results = []
    for text_item in results_text:
        fruit_id = text_item.payload.get("fruit_id")
        related_images = [img for img in results_image if img.payload.get("fruit_id") == fruit_id]

        if not related_images:
            combined_results.append({
                'payload': {
                    'name': text_item.payload.get("name", "Không tên"),
                    'image_url': "/static/default.jpg"
                },
                'score': round(text_item.score, 4)
            })
        else:
            for img in related_images:
                combined_results.append({
                    'payload': {
                        'name': text_item.payload.get("name", "Không tên"),
                        'image_url': img.payload.get("image_url", "/static/default.jpg")
                    },
                    'score': round(text_item.score, 4)
                })

    return combined_results


if __name__ == "__main__":
    keyword = "mùa hè"
    results = search_fruits_and_images(keyword)
    print(f"✅ Kết quả tìm kiếm cho từ khóa '{keyword}':")
    for item in results:
        print(f"- Name: {item['payload']['name']}, Image URL: {item['payload']['image_url']}, Score: {item['score']}")
