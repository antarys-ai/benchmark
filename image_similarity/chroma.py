import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import chromadb
from chromadb.config import Settings
import os
import matplotlib.pyplot as plt
import time
import uuid
import json
from pathlib import Path


class FeatureExtractor:
    def __init__(self, modelname):
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()
        self.input_size = self.model.default_cfg["input_size"]
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        input_image = Image.open(imagepath).convert("RGB")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        feature_vector = output.squeeze().numpy()
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten().tolist()


def display_results(query_image_path, result_images, query_time):
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 6, 1)
    query_img = Image.open(query_image_path).resize((150, 150))
    plt.imshow(query_img)
    plt.title("Query Image")
    plt.axis('off')

    for i, img_path in enumerate(result_images, start=2):
        plt.subplot(2, 6, i)
        img = Image.open(img_path).resize((150, 150))
        plt.imshow(img)
        plt.title(f"Result {i - 1}")
        plt.axis('off')

    plt.suptitle(f"ChromaDB Query time: {query_time:.4f} seconds", y=1.05)
    plt.tight_layout()

    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    plt.savefig(results_dir / f"result_chromadb_{timestamp}.png")
    plt.show()


def main():
    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    collection_name = f"image_embeddings_{int(time.time())}"

    try:
        for col in client.list_collections():
            if col.name == collection_name:
                client.delete_collection(collection_name)
                break
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 100
        }
    )

    extractor = FeatureExtractor("resnet34")

    metrics = {
        "timestamp": int(time.time()),
        "collection_name": collection_name,
        "model": "resnet34",
        "dimensions": 512,
        "distance": "cosine"
    }

    root = "./train"
    insert = True
    if insert:
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".JPEG"):
                    filepath = os.path.join(dirpath, filename)
                    image_embedding = extractor(filepath)

                    ids.append(str(uuid.uuid4()))
                    embeddings.append(image_embedding)
                    metadatas.append({"filename": filepath})
                    documents.append(f"Image: {filename}")

        upsert_start = time.time()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        upsert_end = time.time()

        upsert_time = upsert_end - upsert_start
        total_vectors = len(ids)
        wps = total_vectors / upsert_time if upsert_time > 0 else 0

        metrics.update({
            "upsert_time": upsert_time,
            "total_vectors": total_vectors,
            "wps": wps
        })

    query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
    query_embedding = extractor(query_image)

    query_start = time.time()
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["metadatas", "distances"]
    )
    query_end = time.time()

    print("=" * 60)
    print(query_embedding)
    print("=" * 60)
    print(search_results)

    query_time = query_end - query_start
    qps = 1 / query_time if query_time > 0 else 0

    metrics.update({
        "query_time": query_time,
        "qps": qps,
        "query_image": query_image,
        "results_count": len(search_results["ids"][0]) if search_results["ids"] else 0
    })

    print(f"Query executed in {query_time:.4f} seconds")

    result_images = []
    if search_results["metadatas"] and search_results["metadatas"][0]:
        for metadata in search_results["metadatas"][0]:
            result_images.append(metadata["filename"])

    display_results(query_image, result_images, query_time)

    metrics_file = results_dir / f"metrics_chromadb_{metrics['timestamp']}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_file}")

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


if __name__ == "__main__":
    main()
