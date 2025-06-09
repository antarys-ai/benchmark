import os
import time
import json
import uuid
import platform
import psutil
import numpy as np

from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from datasets import load_dataset

COLLECTION_NAME = "dbpedia_benchmark_100k"
VECTOR_SIZE = 1536
BATCH_SIZES = [10, 100, 1000]
QUERY_COUNTS = [10, 100, 1000]
SAMPLE_LIMIT = 100000
MAX_RETRIES = 3
RESULTS_DIR = "../results/Chroma"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_system_info() -> Dict[str, Any]:
    try:
        return {
            "os": f"{platform.system()} {platform.release()}",
            "cpu": platform.processor(),
            "cpu_cores": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Could not get system info: {e}"}


async def load_huggingface_dataset(limit: int = SAMPLE_LIMIT) -> List[Dict[str, Any]]:
    print(f"Loading {limit} samples from Hugging Face dataset...")
    dataset = load_dataset("KShivendu/dbpedia-entities-openai-1M", split='train')

    samples = []
    for i, item in tqdm(enumerate(dataset), total=limit, desc="Processing dataset"):
        if i >= limit:
            break
        samples.append({
            "id": str(uuid.uuid4()),
            "values": item["openai"],
            "metadata": {
                "title": item["title"],
                "text": item["text"],
                "source": "dbpedia",
                "sample_id": i
            }
        })

    return samples


def initialize_chroma():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    try:
        for col in client.list_collections():
            if col.name == COLLECTION_NAME:
                print(f"Deleting existing collection '{COLLECTION_NAME}'...")
                client.delete_collection(COLLECTION_NAME)
                break
    except Exception as e:
        print(f"Error checking collections: {e}")

    print(f"Creating collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
            "hnsw:search_ef": 100,
            "hnsw:num_threads": os.cpu_count() or 4
        }
    )
    return client, collection


def benchmark_writes(collection, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking writes with batch size {batch_size}")
        batch_times = []
        total_points = 0

        for i in tqdm(range(0, len(samples), batch_size), desc=f"Batch size {batch_size}"):
            batch = samples[i:i + batch_size]
            ids = [s["id"] for s in batch]
            embeddings = [s["vector"] for s in batch]
            metadatas = [s["metadata"] for s in batch]
            documents = [f"{s['title']}: {s['text'][:500]}" for s in batch]

            for attempt in range(MAX_RETRIES):
                try:
                    start = time.time()
                    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                    elapsed = time.time() - start
                    batch_times.append(elapsed)
                    total_points += len(batch)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                    else:
                        print(f"  Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2 ** attempt)

        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            results.append({
                "operation": "write",
                "batch_size": batch_size,
                "total_points": total_points,
                "successful_batches": len(batch_times),
                "avg_batch_time": avg_time,
                "qps": batch_size / avg_time if avg_time else 0,
                "batch_times": batch_times,
                "percentiles": {
                    "p50": float(np.percentile(batch_times, 50)),
                    "p90": float(np.percentile(batch_times, 90)),
                    "p99": float(np.percentile(batch_times, 99))
                }
            })

    return results


def benchmark_reads(collection, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    try:
        if collection.count() == 0:
            print("No data in collection, skipping reads.")
            return results
    except Exception as e:
        print(f"Could not get collection count: {e}")
        return results

    for count in QUERY_COUNTS:
        print(f"\nBenchmarking {count} queries")
        query_vectors = [s["vector"] for s in samples[:count]]
        query_times = []

        for i, vector in enumerate(tqdm(query_vectors, desc="Running queries")):
            try:
                start = time.time()
                if i % 4 == 0:
                    collection.query(query_embeddings=[vector], n_results=10,
                                     include=["metadatas", "documents", "distances"])
                elif i % 4 == 1:
                    collection.query(query_embeddings=[vector], n_results=5, where={"source": "dbpedia"},
                                     include=["metadatas", "documents", "distances"])
                elif i % 4 == 2:
                    collection.query(query_embeddings=[vector], n_results=7, include=["metadatas", "distances"])
                else:
                    collection.query(query_embeddings=[vector], n_results=5, where={"sample_id": {"$gte": 0}},
                                     include=["metadatas", "distances"])
                query_times.append(time.time() - start)
            except Exception as e:
                print(f"  Query {i + 1} failed: {e}")

        if query_times:
            total_time = sum(query_times)
            results.append({
                "operation": "read",
                "query_count": count,
                "successful_queries": len(query_times),
                "avg_query_time": total_time / len(query_times),
                "qps": len(query_times) / total_time if total_time else 0,
                "query_times": query_times,
                "percentiles": {
                    "p50": float(np.percentile(query_times, 50)),
                    "p90": float(np.percentile(query_times, 90)),
                    "p99": float(np.percentile(query_times, 99))
                }
            })

    return results


def calculate_recall(collection, samples: List[Dict[str, Any]], top_k: int = 100) -> Dict[str, float]:
    print(f"\nCalculating recall with top_k={top_k}...")
    test_queries = samples[:50]
    recalls = []

    for i, query in enumerate(tqdm(test_queries, desc="Calculating recall")):
        try:
            r1 = collection.query(query_embeddings=[query["vector"]], n_results=top_k, include=["distances"])
            r2 = collection.query(query_embeddings=[query["vector"]], n_results=top_k, include=["distances"])
            ids1 = set(r1["ids"][0]) if r1["ids"] else set()
            ids2 = set(r2["ids"][0]) if r2["ids"] else set()
            if ids2:
                recalls.append(len(ids1 & ids2) / len(ids2))
        except Exception as e:
            print(f"Recall failed for query {i}: {e}")

    return {
        "recall_at_100": float(np.mean(recalls)) if recalls else 0.0,
        "recall_std": float(np.std(recalls)) if recalls else 0.0,
        "recall_samples": len(recalls),
        "note": "Consistency check – Chroma doesn’t support exact search"
    }


def save_results(results: List[Dict[str, Any]], system_info: Dict, recall_info: Dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_data = {
        "database": "Chroma",
        "system_info": system_info,
        "benchmark_config": {
            "collection_name": COLLECTION_NAME,
            "vector_size": VECTOR_SIZE,
            "sample_limit": SAMPLE_LIMIT,
            "batch_sizes": BATCH_SIZES,
            "query_counts": QUERY_COUNTS
        },
        "results": results,
        "recall": recall_info,
        "timestamp": timestamp
    }

    json_file = os.path.join(RESULTS_DIR, "benchmark.json")
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=2)

    txt_file = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.txt")
    with open(txt_file, "w") as f:
        f.write(f"Chroma Vector Database Benchmark Results - {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")
        f.write("SYSTEM INFORMATION:\n")
        for key, value in system_info.items():
            f.write(f"- {key}: {value}\n")
        f.write("\n")

        for result in results:
            if result["operation"] == "write":
                f.write(f"WRITE PERFORMANCE (batch size {result['batch_size']})\n")
                f.write(f"- Total points inserted: {result['total_points']}\n")
                f.write(f"- Successful batches: {result['successful_batches']}\n")
                f.write(f"- Average batch time: {result['avg_batch_time']:.4f}s\n")
                f.write(f"- Throughput: {result['qps']:.2f} vectors/sec\n")
                p = result['percentiles']
                f.write(f"- Percentiles: P50={p['p50']:.4f}s, P90={p['p90']:.4f}s, P99={p['p99']:.4f}s\n\n")
            else:
                f.write(f"READ PERFORMANCE ({result['query_count']} queries)\n")
                f.write(f"- Successful queries: {result['successful_queries']}\n")
                f.write(f"- Average query time: {result['avg_query_time']:.4f}s\n")
                f.write(f"- Throughput: {result['qps']:.2f} queries/sec\n")
                p = result['percentiles']
                f.write(f"- Percentiles: P50={p['p50']:.4f}s, P90={p['p90']:.4f}s, P99={p['p99']:.4f}s\n\n")

        if recall_info:
            f.write("RECALL PERFORMANCE\n")
            f.write(f"- Recall@100: {recall_info.get('recall_at_100', 0):.4f}\n")
            f.write(f"- Standard deviation: {recall_info.get('recall_std', 0):.4f}\n")
            f.write(f"- Sample size: {recall_info.get('recall_samples', 0)}\n\n")

        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n")

        write_results = [r for r in results if r["operation"] == "write"]
        if write_results:
            best_write = max(write_results, key=lambda x: x["qps"])
            f.write(
                f"Best write performance: {best_write['qps']:.2f} vectors/sec (batch size {best_write['batch_size']})\n")

        read_results = [r for r in results if r["operation"] == "read"]
        if read_results:
            best_read = max(read_results, key=lambda x: x["qps"])
            f.write(f"Best read performance: {best_read['qps']:.2f} queries/sec ({best_read['query_count']} queries)\n")


def main():
    print("Starting Chroma benchmark...")

    system_info = get_system_info()
    samples = load_huggingface_dataset()

    client, collection = initialize_chroma()

    all_results = []
    recall_info = {}

    try:
        print("\n=== Starting Write Benchmarks ===")
        write_results = benchmark_writes(collection, samples)
        all_results.extend(write_results)

        print("Waiting for indexing to complete...")
        time.sleep(5)

        print("\n=== Starting Read Benchmarks ===")
        read_results = benchmark_reads(collection, samples)
        all_results.extend(read_results)

        print("\n=== Calculating Recall ===")
        recall_info = calculate_recall(collection, samples)

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        save_results(all_results, system_info, recall_info)
        print(f"\nBenchmark complete! Results saved to {RESULTS_DIR}")

        try:
            print(f"Cleaning up collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        except Exception as e:
            print(f"Failed to cleanup collection: {e}")


if __name__ == "__main__":
    main()
