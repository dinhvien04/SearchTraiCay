#tìm kiếm dựa trên text văn bản
#embedding vector text, dùng mô hình để mã hóa câu thành vector số
#tìm kiếm vector tương đồng, tìm kiếm chính xác theo văn bản
#•	Kết hợp tìm kiếm vector với filter metadata (như season, origin, color, category)
from sentence_transformers import SentenceTransformer
from qdrant_utils.connect import connect_qdrant
from qdrant_client.http.models import Filter, FieldCondition, MatchText

# Load model 1 lần duy nhất
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#chuyển vb sang dạng số (đầu vào: text, đầu ra: 1 vector số (listfloat))
def get_text_vector(query_text: str) -> list[float]:
    """Encode câu thành vector embedding."""
    return model.encode(query_text).tolist()

#hàm tìm kiếm vector tương đồng trong 1 collection
def search_by_vector(collection_name: str, query_vector: list[float], top_k: int = 1):
    """Tìm kiếm tương đồng vector trong collection."""
    client = connect_qdrant()
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k, #số kết quả trả về
        with_payload=True
    ).points

#hàm tìm kiếm lọc văn bản chính xác trên trường metadata
#chỉ so sánh text chính xác
def search_by_text_filter(collection_name: str, keyword: str, top_k: int = 1):
    """Tìm kiếm chính xác theo text trong trường 'name' (không dùng vector)."""
    client = connect_qdrant()
    filter_ = Filter(
        must=[FieldCondition(key="name", match=MatchText(text=keyword))]
    )

    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=filter_,
        limit=top_k,
        with_payload=True
    )
    return results

#mở rộng tìm kiếm keyword trên nhiều trường 
def search_all_by_text_filter_multiple_fields(collection_name: str, keyword: str):
    """
    Tìm kiếm theo keyword trong nhiều trường: season, origin, color, category
    Trả về toàn bộ kết quả không giới hạn.
    """
    client = connect_qdrant()

    filter_ = Filter(
        should=[
            FieldCondition(key="season", match=MatchText(text=keyword)),
            FieldCondition(key="origin", match=MatchText(text=keyword)),
            FieldCondition(key="color", match=MatchText(text=keyword)),
            FieldCondition(key="category", match=MatchText(text=keyword)),
        ]
        # ✅ Không dùng min_should_match ở đây
    )

    all_results = []
    offset = 0
    batch_size = 300  # Số lượng lấy mỗi lần

    while True:
        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_,
            limit=batch_size,
            offset=offset,
            with_payload=True
        )
        if not results:
            break
        all_results.extend(results)
        if len(results) < batch_size:
            break
        offset += batch_size

    return all_results

#•	Kết hợp tìm kiếm vector embedding + lọc theo metadata text.
def search_vector_with_metadata_filter(collection_name: str, query_text: str, top_k: int = 20):
    """
    Tìm kiếm vector kèm filter metadata season, category, origin, color chứa keyword query_text.
    """
    client = connect_qdrant()
    query_vector = get_text_vector(query_text)
    
    filter_ = Filter(
        should=[
            FieldCondition(key="season", match=MatchText(text=query_text)),
            FieldCondition(key="category", match=MatchText(text=query_text)),
            FieldCondition(key="origin", match=MatchText(text=query_text)),
            FieldCondition(key="color", match=MatchText(text=query_text)),
        ]
        # ❌ min_should_match bị loại bỏ vì không được hỗ trợ
    )
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=filter_ 
    ).points
    return results

def pretty_print_hits(hits):
    for i, hit in enumerate(hits, 1):
        print(f"{i}. ID: {hit.id}, Score: {getattr(hit, 'score', 0):.4f}")
        print(f"   Name: {hit.payload.get('name', '')}")
        print(f"   Description: {hit.payload.get('description', '')[:60]}...")
        print(f"   Season: {hit.payload.get('season', '')}")
        print(f"   Origin: {hit.payload.get('origin', '')}")
        print(f"   Color: {hit.payload.get('color', '')}")
        print(f"   Category: {hit.payload.get('category', '')}")
        print()

if __name__ == "__main__":
    keyword = "mùa hè"

    print("\n🔍 Vector Search:")
    vector = get_text_vector(keyword)
    hits = search_by_vector("fruit_text", vector, top_k=100)
    pretty_print_hits(hits)

    print("\n🔎 Text Filter Search (field 'name'):")
    hits_text = search_by_text_filter("fruit_text", keyword, top_k=100)
    pretty_print_hits(hits_text)

    print("\n🔎 Text Filter Search (fields season, origin, color, category):")
    hits_multi = search_all_by_text_filter_multiple_fields("fruit_text", keyword)
    pretty_print_hits(hits_multi)

    print("\n🔍 Vector Search + Metadata Filter (season, category, origin, color):")
    hits_vector_filter = search_vector_with_metadata_filter("fruit_text", keyword, top_k=100)
    pretty_print_hits(hits_vector_filter)
