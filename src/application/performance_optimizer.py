from typing import List, Tuple, Optional
import numpy as np
from src.constraints.cost_constraint import CostConstraint
from src.constraints.accuracy_constraint import AccuracyConstraint
from src.application.function import Function
from src.application.application_graph import ApplicationGraph


class PerformanceOptimizer:
    def __init__(self, app: ApplicationGraph) -> None:
        self.app: ApplicationGraph = app
        # For each function, set it to use the minimum accuracy model (if available) and minimal memory.
        for f in list(self.app.functions.values()):
            min_acc_model: Optional[str] = (
                f.min_accuracy_model
            )  # This may be None for non-inference functions.
            min_mem: int = f.get_min_memory(min_acc_model)
            f.set_config(memory_mb=min_mem, model_name=min_acc_model)
            # Removed update_metrics() since it doesn't exist.

    def get_overall_rt(self) -> float:
        """Calculate overall response time as the sum of each function's execution time."""
        return sum(f.execution_time for f in self.app.functions.values())

    def get_overall_cost(self) -> float:
        """Calculate overall cost as the sum of each function's cost."""
        return sum(f.cost for f in self.app.functions.values())

    def optimize_BPBC(
        self,
        budget: float,
        cost_constraint: Optional[CostConstraint] = None,
        accuracy_constraint: Optional[AccuracyConstraint] = None,
        mem_sample_points: int = 5,
    ) -> None:
        improvement: bool = True
        iteration: int = 0

        while improvement:
            iteration += 1
            improvement = False

            current_rt: float = self.get_overall_rt()
            current_cost: float = self.get_overall_cost()

            # Check cost constraint if provided.
            if cost_constraint and not cost_constraint.is_satisfied(
                list(self.app.functions.values())
            ):
                print(
                    f"Cost constraint violated. Current cost: {current_cost}, Budget: {budget}"
                )
                break

            # Retrieve the critical path from the application graph.
            cp: Optional[List[str]] = self.app.find_critical_path()
            if cp is None:
                print("No critical path found.")
                break

            best_candidate: Optional[
                Tuple[Function, float, float, Tuple[int, Optional[str]]]
            ] = None
            # best_candidate structure: (function, rt_improvement, cost_increase, new_config)

            for func_name in cp:
                f: Function = self.app.functions[func_name]
                current_mem: int = f.config.memory_mb
                current_acc: Optional[float] = f.normalized_accuracy
                current_model: Optional[str] = f.config.model_name

                # If the function is inference-based (has available models and normalized accuracies)
                if (
                    f.available_models is not None
                    and f.available_normalized_accuracy is not None
                ):
                    for cand_model, cand_acc in zip(
                        f.available_models, f.available_normalized_accuracy
                    ):
                        # Only consider candidate models that do not reduce accuracy.
                        if current_acc is not None and cand_acc < current_acc:
                            continue

                        cand_min_mem: int = max(
                            current_mem, f.get_min_memory(cand_model)
                        )
                        cand_max_mem: int = f.get_max_memory(cand_model)
                        if cand_min_mem >= cand_max_mem:
                            continue

                        candidate_mems: np.ndarray = np.linspace(
                            cand_min_mem, cand_max_mem, mem_sample_points, dtype=int
                        )
                        for mem in candidate_mems:
                            original_config: Tuple[
                                int, Optional[float], Optional[str]
                            ] = (
                                f.config.memory_mb,
                                f.normalized_accuracy,
                                f.config.model_name,
                            )
                            # Try candidate configuration.
                            f.set_config(memory_mb=mem, model_name=cand_model)
                            # No update_metrics() call here.

                            new_rt: float = self.get_overall_rt()
                            new_cost: float = self.get_overall_cost()

                            if new_cost > budget:
                                # Revert the change.
                                f.set_config(
                                    memory_mb=original_config[0],
                                    model_name=original_config[2],
                                )
                                continue

                            if (
                                accuracy_constraint is not None
                                and not accuracy_constraint.is_satisfied(
                                    list(self.app.functions.values())
                                )
                            ):
                                f.set_config(
                                    memory_mb=original_config[0],
                                    model_name=original_config[2],
                                )
                                continue

                            rt_improvement: float = current_rt - new_rt
                            if rt_improvement > 0:
                                cost_increase: float = new_cost - current_cost
                                if (
                                    best_candidate is None
                                    or rt_improvement > best_candidate[1]
                                ):
                                    best_candidate = (
                                        f,
                                        rt_improvement,
                                        cost_increase,
                                        (mem, cand_model),
                                    )
                            # Revert configuration before checking next candidate.
                            f.set_config(
                                memory_mb=original_config[0],
                                model_name=original_config[2],
                            )
                else:
                    # For non-inference based functions, candidate accuracy remains fixed.
                    cand_acc: Optional[float] = current_acc
                    cand_model: Optional[str] = current_model
                    cand_min_mem: int = max(current_mem, f.get_min_memory(cand_model))
                    cand_max_mem: int = f.get_max_memory(cand_model)
                    if cand_min_mem < cand_max_mem:
                        candidate_mems: np.ndarray = np.linspace(
                            cand_min_mem, cand_max_mem, mem_sample_points, dtype=int
                        )
                        for mem in candidate_mems:
                            original_config: Tuple[
                                int, Optional[float], Optional[str]
                            ] = (
                                f.config.memory_mb,
                                f.normalized_accuracy,
                                f.config.model_name,
                            )
                            f.set_config(memory_mb=mem, model_name=cand_model)

                            new_rt: float = self.get_overall_rt()
                            new_cost: float = self.get_overall_cost()

                            if new_cost > budget:
                                f.set_config(
                                    memory_mb=original_config[0],
                                    model_name=original_config[2],
                                )
                                continue

                            if (
                                accuracy_constraint is not None
                                and not accuracy_constraint.is_satisfied(
                                    list(self.app.functions.values())
                                )
                            ):
                                f.set_config(
                                    memory_mb=original_config[0],
                                    model_name=original_config[2],
                                )
                                continue

                            rt_improvement: float = current_rt - new_rt
                            if rt_improvement > 0:
                                cost_increase: float = new_cost - current_cost
                                if (
                                    best_candidate is None
                                    or rt_improvement > best_candidate[1]
                                ):
                                    best_candidate = (
                                        f,
                                        rt_improvement,
                                        cost_increase,
                                        (mem, cand_model),
                                    )
                            f.set_config(
                                memory_mb=original_config[0],
                                model_name=original_config[2],
                            )

            if best_candidate is not None:
                best_func, rt_improve_val, cost_inc, new_config = best_candidate
                print(
                    f"Iteration {iteration}: Updating function '{best_func.name}' from {best_func.config.to_string()} to memory_mb: {new_config[0]} model_name: {new_config[1]}."
                )
                best_func.set_config(*new_config)
                new_overall_rt: float = self.get_overall_rt()
                new_overall_cost: float = self.get_overall_cost()
                print(
                    f"  New overall RT: {new_overall_rt:.2f}, New overall cost: {new_overall_cost:.2f}"
                )
                improvement = True
            else:
                print("No further improvement found.")
                break

        print("Optimization complete.")
