"""Figure builders for TensorBoard logging."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_result_fig(result: dict, generation: int, fitness: float):
    """Build the 2x2 result figure. Returns the figure."""
    e = result['e']
    y = result['y']
    u = result['u']
    partitions = result['partitions']
    total_fitness = result['fitness']
    w_ref = e + y

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax1, ax_pie = axes[0]
    ax2, ax_bar = axes[1]

    # Top-left: Reference vs Output
    ax1.plot(w_ref, 'r--', linewidth=2, label='Reference (w)')
    ax1.plot(y, 'b-', linewidth=1.5, label='System Output (y)')
    ax1.set_title(f'Gen {generation} | Best Fitness: {fitness:.4f}')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)

    # Partition data
    partition_names = list(partitions.keys())
    partition_values = [float(v) if np.isfinite(float(v)) else 0.0 for v in partitions.values()]
    cmap = plt.cm.Set2
    colors = [cmap(i / max(len(partition_names), 1)) for i in range(len(partition_names))]

    # Top-right: Pie chart — guard against all-zero (pie([0,0]) computes 0/0 = NaN)
    total_pv = sum(partition_values)
    if total_pv > 0 and np.isfinite(total_fitness) and total_fitness > 0:
        partition_labels = [
            f"{name.replace('_', ' ').title()}\n{val:.1f} ({100*val/total_fitness:.1f}%)"
            for name, val in zip(partition_names, partition_values)
        ]
        ax_pie.pie(partition_values, labels=partition_labels, colors=colors,
                   autopct='', startangle=90)
    else:
        ax_pie.text(0.5, 0.5, f'fitness={total_fitness:.2f}',
                    ha='center', va='center', transform=ax_pie.transAxes, fontsize=11)
    ax_pie.set_title('Fitness Partitions')

    # Bottom-left: Control signal
    ax2.plot(u, 'g-', linewidth=1.5, label='Control Signal (u)')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Control')
    ax2.legend()
    ax2.grid(True)

    # Bottom-right: Bar chart
    bar_labels = [name.replace('_', '\n').title() for name in partition_names]
    bars = ax_bar.bar(bar_labels, partition_values, color=colors, edgecolor='black')
    ax_bar.set_ylabel('Fitness Value')
    ax_bar.set_title(f'Total: {total_fitness:.2f}')
    ax_bar.grid(True, axis='y')
    for bar, val in zip(bars, partition_values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    return fig


def build_evolution_fig(evolutions: list[list[float]], root: int = 0):
    """Build the evolution curves figure. Returns the figure."""
    fig, (ax_log, ax_clip) = plt.subplots(1, 2, figsize=(16, 6))

    for i, evo in enumerate(evolutions):
        label = f"Island {i}" + (" (root)" if i == root else "")
        ax_log.plot(evo, alpha=0.7, label=label)
        ax_clip.plot(evo, alpha=0.7, label=label)

    ax_log.set_yscale('log')
    ax_log.set_xlabel('Generation')
    ax_log.set_ylabel('Best Fitness (log)')
    ax_log.set_title('Evolution Curves - Log Scale')
    ax_log.legend()
    ax_log.grid(True)

    all_vals = [v for evo in evolutions for v in evo]
    if all_vals:
        p95 = np.percentile(all_vals, 95)
        min_val = min(all_vals)
        margin = (p95 - min_val) * 0.1
        ax_clip.set_ylim(max(0, min_val - margin), p95 + margin)
    ax_clip.set_xlabel('Generation')
    ax_clip.set_ylabel('Best Fitness')
    ax_clip.set_title('Evolution Curves - Clipped')
    ax_clip.legend()
    ax_clip.grid(True)

    plt.tight_layout()
    return fig
