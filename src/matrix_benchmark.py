"""
Matrix benchmark runner: runs 1,2,4 workers across Uniform, Dynamic, Sharded,
collects aggregate and nodewise metrics, writes CSVs, and generates plots.
"""
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import from src
sys.path.append(str(Path(__file__).parent))
from approaches.uniform_partitioning import run_uniform_partitioning
from approaches.dynamic_partitioning import run_dynamic_partitioning
from approaches.sharded_partitioning import run_sharded_partitioning
from config import PartitioningConfig


def ensure_dirs():
    out_root = Path(__file__).parent.parent / 'results'
    (out_root / 'matrix').mkdir(parents=True, exist_ok=True)
    (out_root / 'plots').mkdir(parents=True, exist_ok=True)
    return out_root


def run_case(approach: str, workers: int, num_samples: int, batch_size: int) -> dict:
    # For fair comparison, all approaches should process the same total workload
    # Use standardized configuration system
    
    # Create base configuration with randomized port to avoid conflicts
    base_config = PartitioningConfig({
        'num_workers': workers,
        'num_samples': num_samples,
        'batch_size': batch_size,
        'chunk_size': batch_size,
        'num_shards': workers,
        'master_port': str(12000 + (os.getpid() % 1000) + workers)
    })
    
    if approach == 'uniform':
        cfg = base_config.get_uniform_config()
        res = run_uniform_partitioning(cfg)
        # new return format: {'aggregate':..., 'nodewise':[...]}
        if 'aggregate' not in res:
            # backward compat
            res = {'aggregate': res, 'nodewise': []}
        return res
    elif approach == 'dynamic':
        cfg = base_config.get_dynamic_config()
        res = run_dynamic_partitioning(cfg)
        return {'aggregate': res, 'nodewise': res.get('nodewise', [])}
    elif approach == 'sharded':
        # Use multiprocessing-based sharded partitioning (no Dask configuration needed)
        sharded_config = base_config.get_sharded_config()
        # Remove Dask-specific parameters that are no longer used
        sharded_config = {k: v for k, v in sharded_config.items() 
                         if not k.startswith('worker_memory_') and k != 'dask_scheduler_address'}
        
        # Configure chunking based on worker count for optimal performance
        if workers == 1:
            sharded_config.update({
                'process_in_chunks': True,
                'chunk_size': 1000  # Smaller chunks for single worker
            })
        else:
            sharded_config.update({
                'process_in_chunks': True,
                'chunk_size': 500   # Moderate chunks for multiple workers
            })
        
        res = run_sharded_partitioning(sharded_config)
        # Convert flat result to expected format with aggregate and nodewise
        aggregate = {
            'throughput': res['throughput'],
            'latency_ms': (res['total_duration'] * 1000) / res['total_inferences'] if res['total_inferences'] > 0 else 0,
            'avg_cpu_percent': res['avg_cpu_percent'],
            'avg_memory_mb': res['avg_memory_mb']
        }
        # Convert worker results to nodewise format
        nodewise = []
        for worker_result in res.get('worker_results', []):
            nodewise.append({
                'worker_id': worker_result['worker_id'],
                'throughput': worker_result['throughput'],
                'latency_ms': (worker_result['inference_time'] * 1000) / worker_result['total_inferences'] if worker_result['total_inferences'] > 0 else 0,
                'avg_cpu_percent': worker_result['avg_cpu_percent'],
                'avg_memory_mb': worker_result['avg_memory_mb']
            })
        return {'aggregate': aggregate, 'nodewise': nodewise}
    else:
        raise ValueError(f"Unknown approach: {approach}")


def to_rows(approach: str, workers: int, bundle: dict):
    rows = []
    agg = bundle['aggregate']
    rows.append({
        'scope': 'aggregate',
        'approach': approach,
        'workers': workers,
        'node': 'all',
        'throughput': agg['throughput'],
        'latency_ms': agg['latency_ms'],
        'cpu_percent': float(agg['avg_cpu_percent']),
        'memory_mb': float(agg['avg_memory_mb'])
    })
    for idx, nw in enumerate(bundle['nodewise']):
        node_id = nw.get('rank') if 'rank' in nw else nw.get('actor_id', nw.get('worker_id', idx))
        rows.append({
            'scope': 'node',
            'approach': approach,
            'workers': workers,
            'node': str(node_id),
            'throughput': nw['throughput'],
            'latency_ms': nw['latency_ms'],
            'cpu_percent': float(nw['avg_cpu_percent']),
            'memory_mb': float(nw['avg_memory_mb'])
        })
    return rows


def plot_per_workers(df: pd.DataFrame, workers: int, out_dir: Path):
    sub = df[(df['scope'] == 'aggregate') & (df['workers'] == workers)]
    metrics = ['throughput', 'latency_ms', 'cpu_percent', 'memory_mb']
    titles = ['Throughput (inf/s)', 'Latency (ms)', 'CPU %', 'Memory (MB)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, m in enumerate(metrics):
        vals = [sub[sub['approach'] == a][m].values[0] if not sub[sub['approach'] == a].empty else 0 for a in ['uniform', 'dynamic', 'sharded']]
        axes[i].bar(['Uniform', 'Dynamic', 'Sharded'], vals, color=['#4e79a7', '#f28e2b', '#59a14f'])
        axes[i].set_title(titles[i])
        axes[i].grid(True, axis='y', alpha=0.3)
    fig.suptitle(f'Aggregate Benchmarks - {workers} Worker(s)')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / f'aggregate_{workers}w.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Matrix benchmark runner')
    parser.add_argument('--sizes', type=str, default='1,2,4', help='Comma-separated worker sizes')
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    out_root = ensure_dirs()

    sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]
    approaches = ['uniform', 'dynamic', 'sharded']

    all_rows = []
    for w in sizes:
        for a in approaches:
            print(f"Running {a} with {w} worker(s)...")
            bundle = run_case(a, w, args.num_samples, args.batch_size)
            rows = to_rows(a, w, bundle)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    matrix_dir = out_root / 'matrix'
    df.to_csv(matrix_dir / 'combined_results.csv', index=False)

    # Speedup/Efficiency per approach (using workers 1 as baseline)
    speed_rows = []
    for a in approaches:
        base = df[(df['approach'] == a) & (df['scope'] == 'aggregate') & (df['workers'] == sizes[0])]
        if base.empty:
            continue
        base_thr = base['throughput'].values[0]
        for w in sizes:
            thr = df[(df['approach'] == a) & (df['scope'] == 'aggregate') & (df['workers'] == w)]['throughput']
            if thr.empty:
                continue
            thr_val = thr.values[0]
            speedup = thr_val / base_thr if base_thr > 0 else 0
            efficiency = (speedup / w) * 100
            speed_rows.append({'approach': a, 'workers': w, 'speedup': speedup, 'efficiency_percent': efficiency})
    sp = pd.DataFrame(speed_rows)
    sp.to_csv(matrix_dir / 'speedup_efficiency.csv', index=False)

    # Plots per workers
    plots_dir = out_root / 'plots'
    for w in sizes:
        plot_per_workers(df, w, plots_dir)
    print(f"Saved plots to {plots_dir}")


if __name__ == '__main__':
    main()


