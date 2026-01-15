from qdrant_client import QdrantClient

def connect_qdrant(host="localhost", port=6333):
    return QdrantClient(host=host, port=port)
