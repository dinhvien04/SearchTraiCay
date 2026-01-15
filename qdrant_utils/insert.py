from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Optional, Union, Dict

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    distance: models.Distance = models.Distance.COSINE
) -> None:
    """
    Tạo hoặc tạo lại collection mới trên Qdrant.
    Nếu collection đã tồn tại thì recreate (xóa + tạo mới).
    """
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance),
        )
        print(f"✅ Tạo collection '{collection_name}' thành công.")
    except Exception as e:
        print(f"❌ Lỗi khi tạo collection '{collection_name}': {e}")

#chèn vector vào collection
def insert_vectors(
    client: QdrantClient,
    collection_name: str,
    vectors: List[Union[np.ndarray, list]],
    payloads: List[Dict],
    ids: Optional[List[Union[int, str]]] = None
) -> None:
    """
    Chèn danh sách vector cùng payload (và id tùy chọn) vào collection.
    
    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection Qdrant.
        vectors: Danh sách vector numpy array hoặc list.
        payloads: Danh sách dict chứa thông tin metadata tương ứng vector.
        ids: Danh sách id cho từng vector (int hoặc str). Nếu None thì tự generate id theo index.
        
    Raises:
        ValueError nếu kích thước các list không khớp.
    """
    if ids is not None and len(ids) != len(vectors):
        raise ValueError("Length of ids must match length of vectors.")
    if len(payloads) != len(vectors):
        raise ValueError("Length of payloads must match length of vectors.")

    try:
        points = []
        for idx, (vec, payload) in enumerate(zip(vectors, payloads)):
            point_id = ids[idx] if ids is not None else idx

            # Chuyển vec thành list nếu là numpy array
            if isinstance(vec, np.ndarray):
                vector_list = vec.tolist()
            elif isinstance(vec, list):
                vector_list = vec
            else:
                vector_list = list(vec)  # fallback

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector_list,
                    payload=payload
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Đã chèn {len(points)} vector vào collection '{collection_name}'. "
              f"(Ví dụ id đầu: {points[0].id if points else 'N/A'})")
    except Exception as e:
        print(f"❌ Lỗi khi chèn vector vào collection '{collection_name}': {e}")
