import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any
import antarys
from tqdm.asyncio import tqdm
import numpy as np
import uuid
from datasets import load_dataset
import os
import platform
import psutil

COLLECTION_NAME = "dbpedia_benchmark_100k"
VECTOR_SIZE = 1536
BATCH_SIZES = [10, 100, 1000]
QUERY_COUNTS = [10, 100, 1000]
SAMPLE_LIMIT = 100000
MAX_RETRIES = 3

RESULTS_DIR = "../results/Antarys"
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_system_info():
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


async def initialize_antarys(client):
    try:
        collections = await client.list_collections()
        collection_names = [col.get("name", col) if isinstance(col, dict) else col for col in collections]
        if COLLECTION_NAME in collection_names:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            await client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error checking collections: {e}")

    print(f"Creating collection '{COLLECTION_NAME}'...")
    await client.create_collection(
        name=COLLECTION_NAME,
        dimensions=VECTOR_SIZE,
        enable_hnsw=True,
        shards=16,
        m=16,
        ef_construction=100
    )


async def benchmark_writes(client, samples: List[Dict[str, Any]]):
    results = []
    vector_ops = client.vector_operations(COLLECTION_NAME)

    for batch_size in BATCH_SIZES:
        print(f"\nBenchmarking writes with batch size {batch_size}")
        batch_times = []
        total_points = 0
        successful_batches = 0

        for i in tqdm(range(0, len(samples), batch_size), desc=f"Batch size {batch_size}"):
            batch = samples[i:i + batch_size]

            for attempt in range(MAX_RETRIES):
                try:
                    start_time = time.time()
                    result = await vector_ops.upsert(
                        batch,
                        batch_size=batch_size,
                        show_progress=False
                    )
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)
                    successful_batches += 1
                    total_points += result.get("upserted_count", len(batch))
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                        break
                    print(f"  Attempt {attempt + 1} failed, retrying...")
                    await asyncio.sleep(2 ** attempt)

        if batch_times:
            avg_time = sum(batch_times) / len(batch_times)
            qps = batch_size / avg_time if avg_time > 0 else 0
            results.append({
                "operation": "write",
                "batch_size": batch_size,
                "total_points": total_points,
                "successful_batches": successful_batches,
                "avg_batch_time": avg_time,
                "qps": qps,
                "batch_times": batch_times,
                "percentiles": {
                    "p50": float(np.percentile(batch_times, 50)),
                    "p90": float(np.percentile(batch_times, 90)),
                    "p99": float(np.percentile(batch_times, 99))
                }
            })

    return results


async def benchmark_reads(client, samples: List[Dict[str, Any]]):
    results = []
    vector_ops = client.vector_operations(COLLECTION_NAME)

    for query_count in QUERY_COUNTS:
        print(f"\nBenchmarking {query_count} queries")
        query_times = []
        successful_queries = 0
        query_vectors = [sample["values"] for sample in samples[:query_count]]

        for i, vector in enumerate(tqdm(query_vectors, desc="Running queries")):
            try:
                start_time = time.time()
                if i % 4 == 0:
                    await vector_ops.query(vector=vector, top_k=10, include_metadata=True)
                elif i % 4 == 1:
                    await vector_ops.query(vector=vector, top_k=5, include_metadata=True,
                                           filter={"metadata.source": "dbpedia"})
                elif i % 4 == 2:
                    await vector_ops.query(vector=vector, top_k=7, include_metadata=True, threshold=0.5)
                else:
                    await vector_ops.query(vector=vector, top_k=5, include_metadata=True,
                                           use_ann=True, ef_search=200, threshold=0.3)

                query_time = time.time() - start_time
                query_times.append(query_time)
                successful_queries += 1
            except Exception as e:
                print(f"  Query {i + 1} failed: {e}")
                continue

        if query_times:
            avg_time = sum(query_times) / len(query_times)
            qps = successful_queries / sum(query_times) if query_times else 0
            results.append({
                "operation": "read",
                "query_count": query_count,
                "successful_queries": successful_queries,
                "avg_query_time": avg_time,
                "qps": qps,
                "query_times": query_times,
                "percentiles": {
                    "p50": float(np.percentile(query_times, 50)),
                    "p90": float(np.percentile(query_times, 90)),
                    "p99": float(np.percentile(query_times, 99))
                }
            })

    return results


async def calculate_recall(client, samples: List[Dict[str, Any]], top_k: int = 100) -> Dict[str, float]:
    print(f"\nCalculating recall with top_k={top_k}...")
    vector_ops = client.vector_operations(COLLECTION_NAME)
    test_queries = samples[:50]
    recalls = []

    for i, query_sample in enumerate(tqdm(test_queries, desc="Calculating recall")):
        try:
            ann_results = await vector_ops.query(
                vector=query_sample["values"],
                top_k=top_k,
                use_ann=True,
                include_metadata=False
            )
            bf_results = await vector_ops.query(
                vector=query_sample["values"],
                top_k=top_k,
                use_ann=False,
                include_metadata=False
            )
            ann_ids = set(match["id"] for match in ann_results.get("matches", []))
            bf_ids = set(match["id"] for match in bf_results.get("matches", []))

            if bf_ids:
                recall = len(ann_ids.intersection(bf_ids)) / len(bf_ids)
                recalls.append(recall)

        except Exception as e:
            print(f"Recall calculation failed for query {i}: {e}")
            continue

    if recalls:
        return {
            "recall_at_100": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "recall_samples": len(recalls)
        }
    else:
        return {"recall_at_100": 0.0, "recall_std": 0.0, "recall_samples": 0}


def save_results(results: List[Dict[str, Any]], system_info: Dict, recall_info: Dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_data = {
        "database": "Antarys",
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
        f.write(f"Antarys Vector Database Benchmark Results - {datetime.now().isoformat()}\n")
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


async def main():
    print("Starting Antarys benchmark...")

    system_info = get_system_info()

    samples = await load_huggingface_dataset()

    client = await antarys.create_client(
        host="http://localhost:8080",
        timeout=120,
        debug=False,
        use_http2=True,
        cache_size=1000
    )

    await initialize_antarys(client)

    all_results = []
    recall_info = {}

    try:
        print("\n=== Starting Write Benchmarks ===")
        write_results = await benchmark_writes(client, samples)
        all_results.extend(write_results)

        print("\n=== Starting Read Benchmarks ===")
        read_results = await benchmark_reads(client, samples)
        all_results.extend(read_results)

        print("\n=== Calculating Recall ===")
        recall_info = await calculate_recall(client, samples)

    except Exception as e:
        print(f"\nBenchmark failed: {e}")
    finally:
        save_results(all_results, system_info, recall_info)
        print(f"\nBenchmark complete! Results saved to {RESULTS_DIR}")
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
