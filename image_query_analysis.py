import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any
import re


def extract_database_name(filename: str) -> str:
    match = re.search(r'metrics_(\w+)_\d+\.json', filename)
    return match.group(1) if match else 'unknown'


def load_metrics_files(results_dir: str = "./query_results") -> List[Dict[str, Any]]:
    metrics_files = glob.glob(os.path.join(results_dir, "metrics_*.json"))
    all_metrics = []

    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            filename = os.path.basename(file_path)
            db_name = extract_database_name(filename)
            data['database'] = db_name

            all_metrics.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return all_metrics


def calculate_summary_stats(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics:
        return {}

    db_groups = {}
    for metric in metrics:
        db_name = metric.get('database', 'unknown')
        if db_name not in db_groups:
            db_groups[db_name] = []
        db_groups[db_name].append(metric)

    summary = {
        "total_tests": len(metrics),
        "databases_tested": list(db_groups.keys()),
        "database_count": len(db_groups),
        "performance_comparison": {},
        "database_details": {}
    }

    for db_name, db_metrics in db_groups.items():
        if not db_metrics:
            continue

        latest_metric = max(db_metrics, key=lambda x: x.get('timestamp', 0))

        wps = latest_metric.get('wps', 0)
        qps = latest_metric.get('qps', 0)
        upsert_time = latest_metric.get('upsert_time', 0)
        query_time = latest_metric.get('query_time', 0)
        total_vectors = latest_metric.get('total_vectors', 0)

        summary["performance_comparison"][db_name] = {
            "wps": round(wps, 2),
            "qps": round(qps, 2),
            "upsert_time": round(upsert_time, 4),
            "query_time": round(query_time, 4),
            "total_vectors": total_vectors
        }

        summary["database_details"][db_name] = {
            "collection_name": latest_metric.get('collection_name', ''),
            "model": latest_metric.get('model', ''),
            "dimensions": latest_metric.get('dimensions', 0),
            "distance_metric": latest_metric.get('distance', ''),
            "index_type": latest_metric.get('index_type', ''),
            "results_count": latest_metric.get('results_count', 0),
            "timestamp": latest_metric.get('timestamp', 0)
        }

    if summary["performance_comparison"]:
        best_wps = max(summary["performance_comparison"].items(), key=lambda x: x[1]['wps'])
        best_qps = max(summary["performance_comparison"].items(), key=lambda x: x[1]['qps'])
        fastest_upsert = min(summary["performance_comparison"].items(), key=lambda x: x[1]['upsert_time'])
        fastest_query = min(summary["performance_comparison"].items(), key=lambda x: x[1]['query_time'])

        summary["best_performers"] = {
            "highest_wps": {"database": best_wps[0], "value": best_wps[1]['wps']},
            "highest_qps": {"database": best_qps[0], "value": best_qps[1]['qps']},
            "fastest_upsert": {"database": fastest_upsert[0], "value": fastest_upsert[1]['upsert_time']},
            "fastest_query": {"database": fastest_query[0], "value": fastest_query[1]['query_time']}
        }

    return summary


def prepare_chart_data(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    chart_data = {
        "performance_chart": [],
        "time_comparison": [],
        "throughput_comparison": [],
        "database_overview": []
    }

    db_groups = {}
    for metric in metrics:
        db_name = metric.get('database', 'unknown')
        if db_name not in db_groups:
            db_groups[db_name] = []
        db_groups[db_name].append(metric)

    for db_name, db_metrics in db_groups.items():
        if not db_metrics:
            continue

        latest_metric = max(db_metrics, key=lambda x: x.get('timestamp', 0))

        chart_data["performance_chart"].append({
            "database": db_name,
            "wps": round(latest_metric.get('wps', 0), 2),
            "qps": round(latest_metric.get('qps', 0), 2),
            "total_vectors": latest_metric.get('total_vectors', 0)
        })

        chart_data["time_comparison"].append({
            "database": db_name,
            "upsert_time": round(latest_metric.get('upsert_time', 0), 4),
            "query_time": round(latest_metric.get('query_time', 0), 4)
        })

        chart_data["throughput_comparison"].append({
            "database": db_name,
            "writes_per_second": round(latest_metric.get('wps', 0), 2),
            "queries_per_second": round(latest_metric.get('qps', 0), 2)
        })

        chart_data["database_overview"].append({
            "name": db_name,
            "model": latest_metric.get('model', ''),
            "dimensions": latest_metric.get('dimensions', 0),
            "distance": latest_metric.get('distance', ''),
            "index_type": latest_metric.get('index_type', ''),
            "collection": latest_metric.get('collection_name', ''),
            "results": latest_metric.get('results_count', 0),
            "timestamp": latest_metric.get('timestamp', 0)
        })

    return chart_data


def generate_analysis_report(results_dir: str = "./query_results") -> Dict[str, Any]:
    print("Loading metrics files...")
    metrics = load_metrics_files(results_dir)

    if not metrics:
        return {
            "error": "No metrics files found",
            "results_directory": results_dir,
            "files_found": 0
        }

    print(f"Found {len(metrics)} test results")

    print("Calculating summary statistics...")
    summary = calculate_summary_stats(metrics)

    print("Preparing chart data...")
    chart_data = prepare_chart_data(metrics)

    analysis_report = {
        "metadata": {
            "generated_at": int(Path().stat().st_mtime) if Path().exists() else 0,
            "results_directory": results_dir,
            "files_processed": len(metrics),
            "analysis_version": "1.0"
        },
        "summary": summary,
        "chart_data": chart_data,
        "raw_metrics": metrics
    }

    return analysis_report


def save_analysis_report(output_file: str = "./analysis_report.json"):
    report = generate_analysis_report()

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Analysis report saved to {output_file}")
    return report


def print_summary_table(report: Dict[str, Any]):
    summary = report.get('summary', {})
    performance = summary.get('performance_comparison', {})

    if not performance:
        print("No performance data available")
        return

    print(f"{'Database':<15} {'WPS':<12} {'QPS':<12} {'Upsert(s)':<12} {'Query(s)':<12}")

    for db_name, metrics in performance.items():
        print(f"{db_name:<15} {metrics['wps']:<12.1f} {metrics['qps']:<12.1f} "
              f"{metrics['upsert_time']:<12.4f} {metrics['query_time']:<12.4f}")

    best = summary.get('best_performers', {})
    if best:
        print("\n" + "=" * 50)
        print("BEST PERFORMERS")
        print("=" * 50)
        print(f"Highest WPS: {best['highest_wps']['database']} ({best['highest_wps']['value']:.1f})")
        print(f"Highest QPS: {best['highest_qps']['database']} ({best['highest_qps']['value']:.1f})")
        print(f"Fastest Upsert: {best['fastest_upsert']['database']} ({best['fastest_upsert']['value']:.4f}s)")
        print(f"Fastest Query: {best['fastest_query']['database']} ({best['fastest_query']['value']:.4f}s)")


if __name__ == "__main__":
    report = save_analysis_report()
    print_summary_table(report)

    print(f"\nTotal databases tested: {report['summary']['database_count']}")
    print(f"Databases: {', '.join(report['summary']['databases_tested'])}")
    print(f"Analysis report saved to: ./analysis_report.json")
