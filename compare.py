#!/usr/bin/env python3

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics
import glob

RESULTS_DIR = "results"
COMPARISON_FILE = os.path.join(RESULTS_DIR, "comparison.json")
TYPESCRIPT_FILE = os.path.join(RESULTS_DIR, "dbpedia.ts")


def load_benchmark_results() -> Dict[str, Dict]:
    results = {}

    if not os.path.exists(RESULTS_DIR):
        print(f"Results directory '{RESULTS_DIR}' not found!")
        return results

    db_dirs = [d for d in os.listdir(RESULTS_DIR)
               if os.path.isdir(os.path.join(RESULTS_DIR, d))]

    for db_dir in db_dirs:
        db_path = os.path.join(RESULTS_DIR, db_dir)
        json_file = os.path.join(db_path, "benchmark.json")

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results[db_dir] = data
                    print(f"Loaded results for {db_dir}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        else:
            print(f"No benchmark.json found for {db_dir}")

    return results


def extract_performance_metrics(results: Dict[str, Dict]) -> Dict[str, Any]:
    comparison_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "databases": list(results.keys()),
            "comparison_version": "1.0"
        },
        "write_performance": {},
        "read_performance": {},
        "recall_performance": {},
        "system_info": {},
        "charts": {
            "write_throughput": [],
            "read_throughput": [],
            "latency_comparison": [],
            "recall_comparison": [],
            "percentile_comparison": []
        }
    }

    for db_name, data in results.items():
        write_results = [r for r in data.get("results", [])
                         if r.get("operation") == "write" and r.get("batch_size") == 1000 or r.get("batch_size") == 100]
        if write_results:
            best_write = max(write_results, key=lambda x: x.get("qps", 0))
            comparison_data["write_performance"][db_name] = {
                "throughput_vectors_per_sec": best_write.get("qps", 0),
                "avg_batch_time": best_write.get("avg_batch_time", 0),
                "batch_size": best_write.get("batch_size", 0),
                "total_points": best_write.get("total_points", 0),
                "percentiles": best_write.get("percentiles", {}),
                "successful_batches": best_write.get("successful_batches", 0)
            }

        read_results = [r for r in data.get("results", []) if r.get("operation") == "read"]
        if read_results:
            read_metrics = {}
            for read_result in read_results:
                query_count = read_result.get("query_count", 0)
                read_metrics[f"queries_{query_count}"] = {
                    "throughput_queries_per_sec": read_result.get("qps", 0),
                    "avg_query_time": read_result.get("avg_query_time", 0),
                    "successful_queries": read_result.get("successful_queries", 0),
                    "percentiles": read_result.get("percentiles", {})
                }
            comparison_data["read_performance"][db_name] = read_metrics

        recall_data = data.get("recall", {})
        if recall_data:
            comparison_data["recall_performance"][db_name] = {
                "recall_at_100": recall_data.get("recall_at_100", 0),
                "recall_std": recall_data.get("recall_std", 0),
                "recall_samples": recall_data.get("recall_samples", 0),
                "note": recall_data.get("note", "")
            }

        system_info = data.get("system_info", {})
        comparison_data["system_info"][db_name] = system_info

    comparison_data["charts"] = generate_chart_data(comparison_data)

    return comparison_data


def generate_chart_data(comparison_data: Dict[str, Any]) -> Dict[str, List]:
    charts = {
        "write_throughput": [],
        "read_throughput": [],
        "latency_comparison": [],
        "recall_comparison": [],
        "percentile_comparison": []
    }

    for db_name, write_perf in comparison_data["write_performance"].items():
        charts["write_throughput"].append({
            "database": db_name,
            "throughput": write_perf.get("throughput_vectors_per_sec", 0),
            "batch_size": write_perf.get("batch_size", 0)
        })

    for db_name, read_perf in comparison_data["read_performance"].items():
        query_data = read_perf.get("queries_1000") or next(iter(read_perf.values()), {})
        charts["read_throughput"].append({
            "database": db_name,
            "throughput": query_data.get("throughput_queries_per_sec", 0),
            "avg_latency": query_data.get("avg_query_time", 0) * 1000
        })

    for db_name, read_perf in comparison_data["read_performance"].items():
        query_data = read_perf.get("queries_1000") or next(iter(read_perf.values()), {})
        percentiles = query_data.get("percentiles", {})
        charts["latency_comparison"].append({
            "database": db_name,
            "p50": percentiles.get("p50", 0) * 1000,
            "p90": percentiles.get("p90", 0) * 1000,
            "p99": percentiles.get("p99", 0) * 1000
        })

    for db_name, recall_perf in comparison_data["recall_performance"].items():
        charts["recall_comparison"].append({
            "database": db_name,
            "recall": recall_perf.get("recall_at_100", 0),
            "std_dev": recall_perf.get("recall_std", 0),
            "samples": recall_perf.get("recall_samples", 0)
        })

    for db_name, write_perf in comparison_data["write_performance"].items():
        percentiles = write_perf.get("percentiles", {})
        charts["percentile_comparison"].append({
            "database": db_name,
            "operation": "write",
            "p50": percentiles.get("p50", 0) * 1000,
            "p90": percentiles.get("p90", 0) * 1000,
            "p99": percentiles.get("p99", 0) * 1000
        })

    return charts


def calculate_rankings(comparison_data: Dict[str, Any]) -> Dict[str, List]:
    rankings = {
        "write_throughput": [],
        "read_throughput": [],
        "recall": [],
        "overall": []
    }

    write_data = [(db, perf.get("throughput_vectors_per_sec", 0))
                  for db, perf in comparison_data["write_performance"].items()]
    write_data.sort(key=lambda x: x[1], reverse=True)
    rankings["write_throughput"] = [{"rank": i + 1, "database": db, "value": val}
                                    for i, (db, val) in enumerate(write_data)]

    read_data = []
    for db, perf in comparison_data["read_performance"].items():
        query_data = perf.get("queries_1000") or next(iter(perf.values()), {})
        throughput = query_data.get("throughput_queries_per_sec", 0)
        read_data.append((db, throughput))
    read_data.sort(key=lambda x: x[1], reverse=True)
    rankings["read_throughput"] = [{"rank": i + 1, "database": db, "value": val}
                                   for i, (db, val) in enumerate(read_data)]

    recall_data = [(db, perf.get("recall_at_100", 0))
                   for db, perf in comparison_data["recall_performance"].items()]
    recall_data.sort(key=lambda x: x[1], reverse=True)
    rankings["recall"] = [{"rank": i + 1, "database": db, "value": val}
                          for i, (db, val) in enumerate(recall_data)]

    overall_scores = {}
    databases = set()

    for ranking in rankings.values():
        for entry in ranking:
            databases.add(entry["database"])

    for db in databases:
        scores = []

        write_rank = next((r["rank"] for r in rankings["write_throughput"] if r["database"] == db), len(databases))
        write_score = (len(databases) - write_rank + 1) / len(databases)
        scores.append(write_score * 0.3)

        read_rank = next((r["rank"] for r in rankings["read_throughput"] if r["database"] == db), len(databases))
        read_score = (len(databases) - read_rank + 1) / len(databases)
        scores.append(read_score * 0.4)

        recall_rank = next((r["rank"] for r in rankings["recall"] if r["database"] == db), len(databases))
        recall_score = (len(databases) - recall_rank + 1) / len(databases)
        scores.append(recall_score * 0.3)

        overall_scores[db] = sum(scores)

    overall_data = list(overall_scores.items())
    overall_data.sort(key=lambda x: x[1], reverse=True)
    rankings["overall"] = [{"rank": i + 1, "database": db, "score": score}
                           for i, (db, score) in enumerate(overall_data)]

    return rankings


def generate_summary_statistics(comparison_data: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "total_databases": len(comparison_data["metadata"]["databases"]),
        "write_stats": {},
        "read_stats": {},
        "recall_stats": {}
    }

    write_throughputs = [perf.get("throughput_vectors_per_sec", 0)
                         for perf in comparison_data["write_performance"].values()]
    if write_throughputs:
        summary["write_stats"] = {
            "max_throughput": max(write_throughputs),
            "min_throughput": min(write_throughputs),
            "avg_throughput": statistics.mean(write_throughputs),
            "median_throughput": statistics.median(write_throughputs),
            "std_dev": statistics.stdev(write_throughputs) if len(write_throughputs) > 1 else 0
        }

    read_throughputs = []
    for perf in comparison_data["read_performance"].values():
        query_data = perf.get("queries_1000") or next(iter(perf.values()), {})
        read_throughputs.append(query_data.get("throughput_queries_per_sec", 0))

    if read_throughputs:
        summary["read_stats"] = {
            "max_throughput": max(read_throughputs),
            "min_throughput": min(read_throughputs),
            "avg_throughput": statistics.mean(read_throughputs),
            "median_throughput": statistics.median(read_throughputs),
            "std_dev": statistics.stdev(read_throughputs) if len(read_throughputs) > 1 else 0
        }

    recall_values = [perf.get("recall_at_100", 0)
                     for perf in comparison_data["recall_performance"].values()]
    if recall_values:
        summary["recall_stats"] = {
            "max_recall": max(recall_values),
            "min_recall": min(recall_values),
            "avg_recall": statistics.mean(recall_values),
            "median_recall": statistics.median(recall_values),
            "std_dev": statistics.stdev(recall_values) if len(recall_values) > 1 else 0
        }

    return summary


def generate_typescript_file(comparison_data: Dict[str, Any]) -> str:
    databases = comparison_data["metadata"]["databases"]
    database_union = " | ".join([f'"{db}"' for db in databases])

    ts_content = f"""// @/lib/bench/dbpedia/index.ts

export const data = {json.dumps(comparison_data, indent=2)} as const;

export type DatabaseName = {database_union};

export type BenchmarkData = typeof data;

export type WritePerformance = BenchmarkData["write_performance"][DatabaseName];

export type ReadPerformance = BenchmarkData["read_performance"][DatabaseName];

export type RecallPerformance =
  BenchmarkData["recall_performance"][DatabaseName];
"""

    return ts_content


def generate_detailed_report() -> str:
    results = load_benchmark_results()
    if not results:
        return "No benchmark results found!"

    comparison_data = extract_performance_metrics(results)
    rankings = calculate_rankings(comparison_data)
    summary = generate_summary_statistics(comparison_data)

    report = []
    report.append("=" * 80)
    report.append("VECTOR DATABASE BENCHMARK COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append(f"Databases tested: {', '.join(comparison_data['metadata']['databases'])}")
    report.append("NOTE: Write performance based on batch_size=1000 only")
    report.append("")

    report.append("OVERALL PERFORMANCE RANKINGS")
    report.append("-" * 40)
    for entry in rankings["overall"]:
        report.append(f"{entry['rank']}. {entry['database']} (Score: {entry['score']:.3f})")
    report.append("")

    report.append("WRITE PERFORMANCE RANKINGS (batch_size=1000)")
    report.append("-" * 40)
    for entry in rankings["write_throughput"]:
        report.append(f"{entry['rank']}. {entry['database']}: {entry['value']:.2f} vectors/sec")
    report.append("")

    report.append("READ PERFORMANCE RANKINGS")
    report.append("-" * 40)
    for entry in rankings["read_throughput"]:
        report.append(f"{entry['rank']}. {entry['database']}: {entry['value']:.2f} queries/sec")
    report.append("")

    report.append("RECALL PERFORMANCE RANKINGS")
    report.append("-" * 40)
    for entry in rankings["recall"]:
        report.append(f"{entry['rank']}. {entry['database']}: {entry['value']:.4f}")
    report.append("")

    report.append("SUMMARY STATISTICS")
    report.append("-" * 40)
    if summary["write_stats"]:
        report.append(f"Write Throughput (batch_size=1000) - Max: {summary['write_stats']['max_throughput']:.2f}, "
                      f"Avg: {summary['write_stats']['avg_throughput']:.2f} vectors/sec")
    if summary["read_stats"]:
        report.append(f"Read Throughput - Max: {summary['read_stats']['max_throughput']:.2f}, "
                      f"Avg: {summary['read_stats']['avg_throughput']:.2f} queries/sec")
    if summary["recall_stats"]:
        report.append(f"Recall@100 - Max: {summary['recall_stats']['max_recall']:.4f}, "
                      f"Avg: {summary['recall_stats']['avg_recall']:.4f}")

    return "\n".join(report)


def main():
    print("Vector Database Benchmark Comparison Tool")
    print("=" * 50)

    results = load_benchmark_results()
    if not results:
        print("No benchmark results found!")
        return

    print(f"Found results for: {', '.join(results.keys())}")

    comparison_data = extract_performance_metrics(results)

    rankings = calculate_rankings(comparison_data)

    summary = generate_summary_statistics(comparison_data)

    comparison_data["rankings"] = rankings
    comparison_data["summary"] = summary

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(COMPARISON_FILE, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison data saved to: {COMPARISON_FILE}")

    ts_content = generate_typescript_file(comparison_data)
    with open(TYPESCRIPT_FILE, 'w') as f:
        f.write(ts_content)
    print(f"TypeScript file saved to: {TYPESCRIPT_FILE}")

    report = generate_detailed_report()
    report_file = os.path.join(RESULTS_DIR, "comparison_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Detailed report saved to: {report_file}")

    print("\n" + "=" * 50)
    print("QUICK SUMMARY")
    print("=" * 50)

    if rankings["overall"]:
        print("Overall Winner:", rankings["overall"][0]["database"])

    if rankings["write_throughput"]:
        best_write = rankings["write_throughput"][0]
        print(
            f"Best Write Performance (batch_size=1000): {best_write['database']} ({best_write['value']:.2f} vectors/sec)")

    if rankings["read_throughput"]:
        best_read = rankings["read_throughput"][0]
        print(f"Best Read Performance: {best_read['database']} ({best_read['value']:.2f} queries/sec)")

    if rankings["recall"]:
        best_recall = rankings["recall"][0]
        print(f"Best Recall: {best_recall['database']} ({best_recall['value']:.4f})")

    print(f"\nFull comparison data available in:")
    print(f"  - JSON: {COMPARISON_FILE}")
    print(f"  - TypeScript: {TYPESCRIPT_FILE}")
    print(f"  - Report: {report_file}")
    print("\nThe TypeScript file can be imported directly in React applications!")
    print("Usage: import { data } from '@/lib/bench/dbpedia'")


if __name__ == "__main__":
    main()
