import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
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
        results_dir / f"result_milvus_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    global DISPLAY

    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    connections.connect("default", host="localhost", port="19530")

    collection_name = f"image_embeddings_{int(time.time())}"

    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64,
                    is_primary=True, auto_id=False),
        FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512)
    ]

    schema = CollectionSchema(fields, "Image embeddings collection")
    collection = Collection(collection_name, schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 100}
    }
    collection.create_index("vector", index_params)

    extractor = FeatureExtractor("resnet34")

    metrics = {
        "timestamp": int(time.time()),
        "collection_name": collection_name,
        "model": "resnet34",
        "dimensions": 512,
        "distance": "COSINE",
        "index_type": "HNSW",
        "display_count": DISPLAY
    }

    root = "./train"
    insert = True
    if insert:
        ids = []
        filenames = []
        embeddings = []

        counter = 0
        for dirpath, foldername, filenames_list in os.walk(root):
            for filename in filenames_list:
                if filename.endswith(".JPEG"):
                    filepath = os.path.join(dirpath, filename)
                    image_embedding = extractor(filepath)

                    ids.append(counter)
                    filenames.append(filepath)
                    embeddings.append(image_embedding)
                    counter += 1

        data = [ids, filenames, embeddings]

        upsert_start = time.time()
        collection.insert(data)
        collection.flush()
        collection.load()
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
    search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
    search_results = collection.search(
        data=[query_embedding],
        anns_field="vector",
        param=search_params,
        limit=max(DISPLAY * 2, 50),
        output_fields=["filename"]
    )
    query_end = time.time()

    print("=" * 60)
    print(f"Query embedding first 10 values: {query_embedding[:10]}")
    print("=" * 60)
    print(f"Search results: {search_results}")

    query_time = query_end - query_start
    qps = 1 / query_time if query_time > 0 else 0

    metrics.update({
        "query_time": query_time,
        "qps": qps,
        "query_image": query_image,
        "results_count": len(search_results[0]) if search_results and len(search_results) > 0 else 0,
        "displayed_count": min(DISPLAY, len(search_results[0]) if search_results and len(search_results) > 0 else 0)
    })

    print(f"Query executed in {query_time:.4f} seconds")

    result_images = []
    if search_results and len(search_results) > 0 and len(search_results[0]) > 0:
        for hit in search_results[0]:
            filename = hit.entity.get("filename")
            if filename:
                result_images.append(filename)

    display_results(query_image, result_images, query_time)

    metrics_file = results_dir / f"metrics_milvus_{metrics['timestamp']}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_file}")

    try:
        utility.drop_collection(collection_name)
    except Exception:
        pass


if __name__ == "__main__":
    main()
