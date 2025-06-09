import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import asyncio
import os
import matplotlib.pyplot as plt
from antarys import create_client
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
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


def display_results(query_image_path, result_images, query_time=None):
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

    plt.suptitle(f"Antarys Query time: {query_time:.4f} seconds", y=1.05)
    plt.tight_layout()

    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = int(time.time())
    plt.savefig(results_dir / f"result_antarys_{timestamp}.png")
    plt.show()


async def main():
    results_dir = Path("../query_results")
    results_dir.mkdir(exist_ok=True)

    client = await create_client(
        host="http://localhost:8080",
        use_http2=True,
        cache_size=1000
    )

    collection_name = f"image_embeddings_{int(time.time())}"
    await client.create_collection(
        name=collection_name,
        dimensions=512,
    )

    vector_ops = client.vector_operations(collection_name)
    extractor = FeatureExtractor("resnet34")

    root = "./train"
    insert = True

    metrics = {
        "timestamp": int(time.time()),
        "collection_name": collection_name,
        "model": "resnet34",
        "dimensions": 512
    }

    if insert:
        records = []
        for dirpath, foldername, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".JPEG"):
                    filepath = os.path.join(dirpath, filename)
                    image_embedding = extractor(filepath)
                    record = {
                        "id": str(uuid.uuid4()),
                        "values": image_embedding.tolist(),
                        "metadata": {"filename": filepath}
                    }
                    records.append(record)

        print(records)

        upsert_start = time.time()
        await vector_ops.upsert(
            records,
            batch_size=1000,
            show_progress=True
        )
        await client.commit()
        upsert_end = time.time()

        upsert_time = upsert_end - upsert_start
        total_vectors = len(records)
        wps = total_vectors / upsert_time if upsert_time > 0 else 0

        metrics.update({
            "upsert_time": upsert_time,
            "total_vectors": total_vectors,
            "wps": wps
        })

    query_image = "./test/Afghan_hound/n02088094_4261.JPEG"
    query_embedding = extractor(query_image)

    query_start = time.time()
    search_results = await vector_ops.query(
        vector=query_embedding.tolist(),
        include_metadata=True,
        include_values=False,
        use_ann=True,
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
        "results_count": len(search_results["matches"])
    })

    print(f"Query executed in {query_time:.4f} seconds")

    result_images = []
    for result in search_results["matches"]:
        result_images.append(result["metadata"]["filename"])

    display_results(query_image, result_images, query_time)

    metrics_file = results_dir / f"metrics_antarys_{metrics['timestamp']}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {metrics_file}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
