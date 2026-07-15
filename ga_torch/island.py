"""Island-model Genetic Algorithm using PyTorch tensors.

All populations and GA operations stay on device (CPU/CUDA).
"""

import os
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from . import operators as ops
from . import reproduction as repro
from .display import build_result_fig, build_evolution_fig


class IslandGA:
    """PyTorch-native Island GA with TensorBoard logging.

    Parameters
    ----------
    n_islands : int
        Number of islands.
    pop_size : int
        Population size per island.
    space : Tensor (2, n_genes)
        Gene bounds on device. space[0] = lower, space[1] = upper.
    evaluate_fn : callable
        Batched evaluation: (pop_size, n_genes) tensor -> (pop_size,) fitness tensor.
    evaluate_detail_fn : callable
        Single-individual detailed evaluation: (n_genes,) tensor -> dict with
        keys: fitness (float), partitions (dict of floats), e, y, u (numpy arrays).
    num_generations : int
        Total epochs to run.
    migration_interval : int
        Migrate every N epochs.
    reset_interval : int
        Reset non-root islands every N epochs (0 = never).
    topology : str or dict
        "tree" for auto-built balanced binary tree, or {child: parent} dict.
    device : torch.device or str
        Device for computation.
    log_dir : str
        Directory for TensorBoard logs and saved individuals.
    fitness_fn : str
        Human-readable fitness function description for HParams logging.
    hparams : dict
        Extra hyperparameters to log to TensorBoard (e.g. model architecture).
    """

    def __init__(
        self,
        n_islands: int,
        pop_size: int,
        space: Tensor,
        evaluate_fn,
        evaluate_detail_fn=None,
        num_generations: int = 400,
        migration_interval: int = 50,
        reset_interval: int = 100,
        topology="tree",
        device=None,
        log_dir: str = "results_islands_torch/",
        fitness_fn: str = "",
        hparams: dict = None,
        plan=None,
    ):
        self.n_islands = n_islands
        self.pop_size = pop_size
        self.device = torch.device(device) if device else space.device
        self.space = space.to(self.device)
        self.lower = self.space[0]
        self.upper = self.space[1]
        self.n_genes = space.shape[1]
        self.delta = self.upper / 100
        self.plan = plan if plan is not None else repro.default_plan()
        self.ctx = repro.Context(lower=self.lower, upper=self.upper, delta=self.delta)
        self.evaluate_fn = evaluate_fn
        self.evaluate_detail_fn = evaluate_detail_fn
        self.num_generations = num_generations
        self.migration_interval = migration_interval
        self.reset_interval = reset_interval
        self.log_dir = log_dir
        self.fitness_fn = fitness_fn
        self.extra_hparams = hparams or {}

        # Topology
        if topology == "tree":
            self.parent_of = self._build_tree(n_islands)
        else:
            self.parent_of = dict(topology)

        all_ids = set(range(n_islands))
        children = set(self.parent_of.keys())
        self.root = min(all_ids - children)

        self.children_of = {i: [] for i in range(n_islands)}
        for child, parent in self.parent_of.items():
            self.children_of[parent].append(child)

        # Initialize populations on device
        self.populations = [
            ops.genrpop(pop_size, self.lower, self.upper)
            for _ in range(n_islands)
        ]
        self.fitnesses = [
            torch.full((pop_size,), float('inf'), device=self.device)
            for _ in range(n_islands)
        ]
        self.best_fitness = [float('inf')] * n_islands
        self.best_individual = [None] * n_islands
        self.evolutions: list[list[float]] = [[] for _ in range(n_islands)]

        # Global tracking
        self.global_best_fitness = float('inf')
        self.global_best_individual = None
        self.global_best_island = None

        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------ #
    #  Topology
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_tree(n_islands: int) -> dict:
        return {i: (i - 1) // 2 for i in range(1, n_islands)}

    def print_topology(self):
        def _print_node(node, prefix="", is_last=True):
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            line = f"{prefix}{connector}Island {node}"
            if node == self.root:
                line += " (root, no reset)"
            print(line)
            children = self.children_of[node]
            for i, child in enumerate(children):
                ext = "    " if is_last else "\u2502   "
                _print_node(child, prefix + ext, i == len(children) - 1)

        print("Island topology (migration flows upward):")
        _print_node(self.root, "", True)

    # ------------------------------------------------------------------ #
    #  GA step
    # ------------------------------------------------------------------ #

    def _ga_step(self, pop: Tensor, fitness: Tensor) -> Tensor:
        """One generation, built from the configurable reproduction plan."""
        return repro.step(pop, fitness, self.plan, self.pop_size, self.ctx)

    # ------------------------------------------------------------------ #
    #  Migration & Reset
    # ------------------------------------------------------------------ #

    def _migrate(self):
        for child, parent in self.parent_of.items():
            best_idx = torch.argmin(self.fitnesses[child])
            best_ind = self.populations[child][best_idx].clone()
            replace_idx = torch.randint(6, self.pop_size, (1,)).item()
            self.populations[parent][replace_idx] = best_ind

    def _reset_islands(self):
        for i in range(self.n_islands):
            if i == self.root:
                continue
            self.populations[i] = ops.genrpop(self.pop_size, self.lower, self.upper)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Run the full island GA."""
        self.print_topology()
        print(f"\nIslands: {self.n_islands} | Pop/island: {self.pop_size} | "
              f"Genes: {self.n_genes} | Generations: {self.num_generations}")
        print(f"Migration every {self.migration_interval} epochs | "
              f"Reset every {self.reset_interval} epochs "
              f"({'disabled' if self.reset_interval == 0 else 'non-root only'})")
        print(f"Device: {self.device}")
        if self.fitness_fn:
            print(f"Fitness: {self.fitness_fn}")
        print()

        for epoch in range(1, self.num_generations + 1):

            # --- Evaluate ALL islands in one batched call ---
            all_genes = torch.cat(self.populations, dim=0)
            all_fitness = self.evaluate_fn(all_genes)
            for i in range(self.n_islands):
                start = i * self.pop_size
                self.fitnesses[i] = all_fitness[start:start + self.pop_size]

            # --- Track per-island and global best ---
            island_summary = []
            for i in range(self.n_islands):
                best_idx = torch.argmin(self.fitnesses[i]).item()
                best_fit = self.fitnesses[i][best_idx].item()
                self.evolutions[i].append(best_fit)

                if best_fit < self.best_fitness[i]:
                    self.best_fitness[i] = best_fit
                    self.best_individual[i] = self.populations[i][best_idx].clone()

                if best_fit < self.global_best_fitness:
                    self.global_best_fitness = best_fit
                    self.global_best_individual = self.populations[i][best_idx].clone()
                    self.global_best_island = i

                    torch.save(self.global_best_individual,
                               os.path.join(self.log_dir, "best_individual_global.pt"))

                    print(f"[Epoch {epoch}] New global best: {best_fit:.4f} (island {i})")

                    # Log result image to TensorBoard
                    if self.evaluate_detail_fn is not None:
                        result = self.evaluate_detail_fn(self.global_best_individual)
                        fig = build_result_fig(result, epoch, best_fit)
                        self.writer.add_figure("best_result", fig, epoch)
                        plt.close(fig)

                island_summary.append(f"I{i}={best_fit:.1f}")

            # --- TensorBoard scalars ---
            for i in range(self.n_islands):
                self.writer.add_scalar(f"fitness/island_{i}", self.evolutions[i][-1], epoch)
            self.writer.add_scalar("fitness/global_best", self.global_best_fitness, epoch)

            # --- GA step ---
            for i in range(self.n_islands):
                self.populations[i] = self._ga_step(
                    self.populations[i], self.fitnesses[i])

            # --- Migration ---
            if epoch % self.migration_interval == 0:
                self._migrate()
                print(f"[Epoch {epoch}] Migration performed")

            # --- Reset ---
            if self.reset_interval > 0 and epoch % self.reset_interval == 0:
                self._reset_islands()
                print(f"[Epoch {epoch}] Non-root islands reset")

            # --- Periodic status ---
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.num_generations}: "
                      + " | ".join(island_summary)
                      + f" | Global best: {self.global_best_fitness:.4f}")

        # --- Final summary ---
        print(f"\n{'='*60}")
        print(f"Island GA completed: {self.num_generations} generations")
        print(f"Global best fitness: {self.global_best_fitness:.4f} "
              f"(island {self.global_best_island})")
        print(f"{'='*60}\n")

        for i in range(self.n_islands):
            label = "(root)" if i == self.root else ""
            print(f"  Island {i} {label}: best = {self.best_fitness[i]:.4f}")

        # --- TensorBoard: final evolution plot + result plot + hparams ---
        evo_fig = build_evolution_fig(self.evolutions, self.root)
        self.writer.add_figure("evolution", evo_fig, self.num_generations)
        plt.close(evo_fig)

        if self.evaluate_detail_fn is not None and self.global_best_individual is not None:
            result = self.evaluate_detail_fn(self.global_best_individual)
            fig = build_result_fig(result, self.num_generations, self.global_best_fitness)
            self.writer.add_figure("final_result", fig, self.num_generations)
            plt.close(fig)

        hparams = {
            'n_islands': self.n_islands,
            'pop_size': self.pop_size,
            'n_genes': self.n_genes,
            'generations': self.num_generations,
            'migration_interval': self.migration_interval,
            'reset_interval': self.reset_interval,
            'fitness_fn': self.fitness_fn,
            **self.extra_hparams,
        }
        self.writer.add_hparams(
            hparams,
            {'hparam/global_best_fitness': self.global_best_fitness},
            run_name=".",
        )
        self.writer.close()

        return {
            'global_best_fitness': self.global_best_fitness,
            'global_best_individual': self.global_best_individual,
            'per_island_best': self.best_fitness,
            'evolutions': self.evolutions,
        }
