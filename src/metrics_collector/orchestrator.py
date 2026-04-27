"""
Orchestrator for running multiple metric collectors concurrently and repeatedly over a time window.
All collectors run simultaneously in each cycle and results are aggregated.
"""

import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.metadata.metadata import Metadata
from src.metrics_collector import collectors


async def collect_all_metrics(
    metadata: Metadata,
    metrics_interval: float = 20.0,
    collection_interval_s: float = 2.0,
    output_csv: Optional[str | Path] = None,
):
    """Run all collectors concurrently and repeatedly over a time window.

    Executes each collector simultaneously in repeated cycles, stores results in CSV.

    Args:
        metadata: Metadata with process_id and gpu_index
        metrics_interval: Total collection duration in seconds
        collection_interval_s: Seconds between collection cycles
        output_csv: Optional path to export results as CSV

    Returns:
        Dictionary with:
            - results: List of all metric records
            - summary: Collection statistics
    """
    all_results = []
    start_time = time.time()
    cycle = 0

    print(f"Starting collection: {len(collectors)} collectors for {metrics_interval}s")
    print(f"Collection interval: {collection_interval_s}s")

    # Repeated collection cycles
    while time.time() - start_time < metrics_interval:
        cycle += 1
        cycle_start = time.time()

        print(f"Cycle {cycle}...", end=" ", flush=True)

        try:
            # Create task for each collector and run concurrently
            tasks = [collector(metadata).collect_metrics() for collector in collectors]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process and store results
            for i, result in enumerate(results):
                collector_name = collectors[i].__name__

                if isinstance(result, Exception):
                    record = {
                        "cycle": cycle,
                        "collector": collector_name,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(result),
                    }
                else:
                    # Flatten result for CSV
                    record = _flatten_result(result, collector_name, cycle)

                all_results.append(record)

            print("✓", flush=True)

        except Exception as e:
            print(f"✗ Cycle {cycle} failed: {e}", flush=True)

        # Wait for next collection interval
        elapsed = time.time() - cycle_start
        if cycle * collection_interval_s < metrics_interval:
            wait_time = max(0, collection_interval_s - elapsed)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    total_duration = time.time() - start_time

    # Export to CSV if specified
    if output_csv and all_results:
        _export_to_csv(all_results, output_csv)

    # Prepare summary
    summary = {
        "total_collectors": len(collectors),
        "total_cycles": cycle,
        "total_records": len(all_results),
        "successful_records": sum(1 for r in all_results if "error" not in r),
        "failed_records": sum(1 for r in all_results if "error" in r),
        "requested_duration_s": metrics_interval,
        "actual_duration_s": round(total_duration, 2),
        "collection_interval_s": collection_interval_s,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "output_csv": str(output_csv) if output_csv else None,
    }

    print(f"\n✓ Collection complete: {cycle} cycles, {len(all_results)} records in {total_duration:.2f}s")

    return {
        "results": all_results,
        "summary": summary,
    }


def _flatten_result(result: Dict[str, Any], collector_name: str, cycle: int) -> Dict[str, Any]:
    """Flatten nested result dictionary for CSV export."""
    flat = {
        "cycle": cycle,
        "collector": collector_name,
        "timestamp": datetime.now().isoformat(),
    }

    for key, value in result.items():
        if isinstance(value, (list, tuple)):
            # For list metrics, compute statistics
            if value and isinstance(value[0], (int, float)):
                flat[f"{key}_count"] = len(value)
                flat[f"{key}_min"] = min(value)
                flat[f"{key}_max"] = max(value)
                flat[f"{key}_avg"] = round(sum(value) / len(value), 4)
            else:
                flat[f"{key}"] = str(value)
        elif isinstance(value, dict):
            # Expand nested dictionaries
            for k, v in value.items():
                flat[f"{key}_{k}"] = v
        else:
            flat[key] = value

    return flat


def _export_to_csv(results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Export results to CSV file."""
    output_path = Path(output_path)

    if not results:
        print("No results to export")
        return

    # Collect all possible keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    fieldnames = sorted(list(all_keys))

    # Write to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"✓ Exported {len(results)} records to {output_path}")
