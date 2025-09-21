import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import matplotlib.pyplot as plt
import time
import uuid
import json
from pathlib import Path
import math


DISPLAY = 100


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
    global DISPLAY

    display_images = result_images[:DISPLAY]
    total_images = len(display_images) + 1

    cols = min(6, total_images)
    rows = math.ceil(total_images / cols)

    plt.figure(figsize=(cols * 2.5, rows * 2.5))

    plt.subplot(rows, cols, 1)
    query_img = Image.open(query_image_path).resize((150, 150))
    plt.imshow(query_img)
    plt.title("Query Image", fontsize=10, fontweight='bold')
    plt.axis('off')

    for i, img_path in enumerate(display_images, start=2):
        plt.subplot(rows, cols, i)
        img = Image.open(img_path).resize((150, 150))
        plt.imshow(img)
        plt.title(f"Result {i - 1}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()

    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    plt.savefig(
        results_dir / f"result_qdrant_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    global DISPLAY

    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    client = QdrantClient(
        host="localhost",
        port=6333,
        prefer_grpc=True,
        grpc_port=6334
    )

    collection_name = f"image_embeddings_{int(time.time())}"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=512,
            distance=models.Distance.COSINE
        )
    )

    extractor = FeatureExtractor("resnet34")

    metrics = {
        "timestamp": int(time.time()),
        "collection_name": collection_name,
        "model": "resnet34",
        "dimensions": 512,
        "distance": "COSINE",
        "display_count": DISPLAY
    }

    root = "./train"
    insert = True
    if insert is True:
        points = []
        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".JPEG"):
                    filepath = os.path.join(dirpath, filename)
                    image_embedding = extractor(filepath)
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=image_embedding,
                            payload={"filename": filepath}
                        )
                    )

        upsert_start = time.time()
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        upsert_end = time.time()

        upsert_time = upsert_end - upsert_start
        total_vectors = len(points)
        wps = total_vectors / upsert_time if upsert_time > 0 else 0

        metrics.update({
            "upsert_time": upsert_time,
            "total_vectors": total_vectors,
            "wps": wps
        })

    query_image = "./test/Afghan_hound/n02088094_4261.JPEG"

    query_start = time.time()
    query_embedding = extractor(query_image)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=max(DISPLAY * 2, 50),
        with_payload=["filename"]
    )
    query_end = time.time()

    print("=" * 60)
    print(query_embedding)
    print("=" * 60)
    print(search_result)

    query_time = query_end - query_start
    qps = 1 / query_time if query_time > 0 else 0

    metrics.update({
        "query_time": query_time,
        "qps": qps,
        "query_image": query_image,
        "results_count": len(search_result),
        "displayed_count": min(DISPLAY, len(search_result))
    })

    print(f"Query executed in {query_time:.4f} seconds")

    result_images = [hit.payload["filename"] for hit in search_result]

    print(
        f"Found {len(result_images)} results, displaying top {min(DISPLAY, len(result_images))}")

    display_results(query_image, result_images, query_time)

    metrics_file = results_dir / f"metrics_qdrant_{metrics['timestamp']}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":

    main()
