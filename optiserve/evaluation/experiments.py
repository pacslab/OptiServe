"""Experiment drivers for the application optimizer.

These generate the artifacts used to evaluate the optimizer and were previously
methods on ``ApplicationOptimizer``; they live here so the optimizer stays a
library component and the experiment/benchmark harness is separate. Behavior is
byte-for-byte identical (locked by tests/golden/test_evaluation_golden.py).
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from optiserve.logging import get_logger

logger = get_logger(__name__)


def generate_perf_cost_table(optimizer, file, start_iterations=1, end_iterations=None):
    """Exhaustively enumerate every (memory x variant) configuration and write a
    ground-truth performance/cost table to ``file`` as CSV.

    ``start_iterations`` / ``end_iterations`` slice the configuration space for
    multiprocessing sharding.
    """
    rows = []
    optimizer.App.update_ne()
    node_list = [item for item in optimizer.App.workflow_graph.nodes if item not in ['Start', 'End']]

    all_available_mem_list = [
        sorted(optimizer.App.workflow_graph.nodes[node]['perf_profile'][0].keys())
        for node in node_list
    ]
    all_available_model_list = [
        sorted(list(range(len(optimizer.App.workflow_graph.nodes[node]['perf_profile']))))
        for node in node_list
    ]

    if end_iterations is not None:
        task_size = end_iterations - start_iterations + 1
    else:
        task_size = (
            np.prod([len(item) for item in all_available_mem_list])
            * np.prod([len(item) for item in all_available_model_list])
            - start_iterations + 1
        )

    model_configurations = list(itertools.product(*all_available_model_list))
    mem_configurations = list(itertools.product(*all_available_mem_list))
    total_configurations = itertools.product(model_configurations, mem_configurations)

    iterations_count = start_iterations - 1
    logger.debug('Get Performance Cost Table - Task Size: {}'.format(task_size))

    for _ in range(start_iterations - 1):
        next(total_configurations)

    with tqdm(total=task_size) as pbar:
        for model_config, mem_config in total_configurations:
            iterations_count += 1
            current_model_config = dict(zip(node_list, model_config))
            current_mem_config = dict(zip(node_list, mem_config))

            optimizer.update_App_workflow_mem_rt(optimizer.App, current_mem_config, current_model_config)
            current_cost = optimizer.App.get_avg_cost()

            optimizer.App.get_simple_dag()
            current_rt = optimizer.App.get_avg_rt()

            mem_config = {f'f{str(node)}_mem': current_mem_config[node] for node in current_mem_config}
            raw_model_config = {f'f{str(node)}_acc': current_model_config[node] for node in current_model_config}
            model_config = {f'f{str(node)}_acc_value': optimizer.model_accuracy_list[node][current_model_config[node]] for node in current_model_config}

            row = mem_config.copy()
            row.update(raw_model_config)
            row.update(model_config)
            row['Cost'] = current_cost
            row['RT'] = current_rt
            row['ID'] = iterations_count
            rows.append(row)
            pbar.update()
            if end_iterations is not None and iterations_count >= end_iterations:
                break

    data = pd.DataFrame(rows).set_index('ID')
    data.to_csv(file, index=True)


def run_opt_curve(
    optimizer,
    filenameprefix,
    budget_list,
    performance_constraint_list,
    accuracy_constraint_list,
    accuracy_formula,
    BCRthreshold=0.2,
):
    """Sweep the three strategies over constraint grids and write
    ``{prefix}_BPBC.csv`` / ``_BCPC.csv`` / ``_BAPB.csv``.

    NOTE: the BAPB sweep invokes BAPB with BCR enabled (BCRtype='ERT/C') but the
    columns are labeled ``BCR_disabled_*``. This mislabeling is preserved to keep
    the existing App*_BAPB_accuracy artifacts reproducible; confirm the intended
    labeling before relying on it.
    """
    budget_list_copy = budget_list
    accuracy_constraint_list_copy = accuracy_constraint_list
    performance_constraint_list_copy = performance_constraint_list

    # ----- BPBC -----
    bpba_rows = []
    for budget, accuracy_constraint in list(itertools.product(budget_list_copy, accuracy_constraint_list_copy)):
        aRow = {'Budget': budget, 'Accuracy_Constraint': accuracy_constraint, 'BCR_threshold': BCRthreshold}
        for label, kwargs in (
            ('BCR_disabled', dict(BCR=False)),
            ('BCR_RT/M', dict(BCR=True, BCRtype='RT/M', BCRthreshold=BCRthreshold)),
            ('BCR_ERT/C', dict(BCR=True, BCRtype='ERT/C', BCRthreshold=BCRthreshold)),
            ('BCR_MAX', dict(BCR=True, BCRtype='MAX')),
        ):
            rt, cost, acc_score, mem_config, model_config, iterations = optimizer.BPBA(
                budget, accuracy_constraint, accuracy_formula, **kwargs
            )
            aRow[f'{label}_RT'] = rt
            aRow[f'{label}_Cost'] = cost
            aRow[f'{label}_Config'] = mem_config
            aRow[f'{label}_Acc_Config'] = model_config
            aRow[f'{label}_Iterations'] = iterations
            aRow[f'{label}_Acc_Score'] = acc_score
        bpba_rows.append(aRow)

    BPBC_data = pd.DataFrame(bpba_rows)[
        ['Budget', 'Accuracy_Constraint', 'BCR_disabled_RT', 'BCR_RT/M_RT', 'BCR_ERT/C_RT', 'BCR_MAX_RT',
         'BCR_disabled_Cost', 'BCR_RT/M_Cost', 'BCR_ERT/C_Cost', 'BCR_MAX_Cost',
         'BCR_disabled_Config', 'BCR_RT/M_Config', 'BCR_ERT/C_Config', 'BCR_MAX_Config',
         'BCR_disabled_Acc_Config', 'BCR_RT/M_Acc_Config', 'BCR_ERT/C_Acc_Config', 'BCR_MAX_Acc_Config',
         'BCR_disabled_Acc_Score', 'BCR_RT/M_Acc_Score', 'BCR_ERT/C_Acc_Score', 'BCR_MAX_Acc_Score',
         'BCR_disabled_Iterations', 'BCR_RT/M_Iterations', 'BCR_ERT/C_Iterations',
         'BCR_MAX_Iterations', 'BCR_threshold']
    ]
    BPBC_data.to_csv(filenameprefix + '_BPBC.csv', index=False)

    # ----- BCPC -----
    bcpa_rows = []
    for perf_constraint, accuracy_constraint in list(itertools.product(performance_constraint_list_copy, accuracy_constraint_list_copy)):
        aRow = {'Performance_Constraint': perf_constraint, 'Accuracy_Constraint': accuracy_constraint, 'BCR_threshold': BCRthreshold}
        for label, kwargs in (
            ('BCR_disabled', dict(BCR=False)),
            ('BCR_M/RT', dict(BCR=True, BCRtype='RT/M', BCRthreshold=BCRthreshold)),
            ('BCR_C/ERT', dict(BCR=True, BCRtype='ERT/C', BCRthreshold=BCRthreshold)),
            ('BCR_MAX', dict(BCR=True, BCRtype='MAX')),
        ):
            rt, cost, acc_score, mem_config, model_config, iterations = optimizer.BCPA(
                perf_constraint, accuracy_constraint, accuracy_formula, **kwargs
            )
            aRow[f'{label}_RT'] = rt
            aRow[f'{label}_Cost'] = cost
            aRow[f'{label}_Config'] = mem_config
            aRow[f'{label}_Acc_Config'] = model_config
            aRow[f'{label}_Iterations'] = iterations
            aRow[f'{label}_Acc_Score'] = acc_score
        bcpa_rows.append(aRow)

    BCPC_data = pd.DataFrame(bcpa_rows)[
        ['Performance_Constraint', 'Accuracy_Constraint', 'BCR_disabled_RT', 'BCR_M/RT_RT', 'BCR_C/ERT_RT', 'BCR_MAX_RT',
         'BCR_disabled_Cost', 'BCR_M/RT_Cost', 'BCR_C/ERT_Cost', 'BCR_MAX_Cost',
         'BCR_disabled_Config', 'BCR_M/RT_Config', 'BCR_C/ERT_Config', 'BCR_MAX_Config',
         'BCR_disabled_Acc_Config', 'BCR_M/RT_Acc_Config', 'BCR_C/ERT_Acc_Config', 'BCR_MAX_Acc_Config',
         'BCR_disabled_Acc_Score', 'BCR_M/RT_Acc_Score', 'BCR_C/ERT_Acc_Score', 'BCR_MAX_Acc_Score',
         'BCR_disabled_Iterations', 'BCR_M/RT_Iterations', 'BCR_C/ERT_Iterations',
         'BCR_MAX_Iterations', 'BCR_threshold']
    ]
    BCPC_data.to_csv(filenameprefix + '_BCPC.csv', index=False)

    # ----- BAPB -----
    bapb_rows = []
    for performance_constraint, budget in list(itertools.product(performance_constraint_list_copy, budget_list_copy)):
        aRow = {'Performance_Constraint': performance_constraint, 'Budget': budget, 'BCR_threshold': BCRthreshold}
        rt, cost, acc_score, mem_config, model_config, iterations = optimizer.BAPB(
            performance_constraint, budget, accuracy_formula, BCR=True, BCRtype='ERT/C', BCRthreshold=BCRthreshold
        )
        aRow['BCR_disabled_RT'] = rt
        aRow['BCR_disabled_Cost'] = cost
        aRow['BCR_disabled_Config'] = mem_config
        aRow['BCR_disabled_Acc_Config'] = model_config
        aRow['BCR_disabled_Iterations'] = iterations
        aRow['BCR_disabled_Acc_Score'] = acc_score
        bapb_rows.append(aRow)

    BAPB_data = pd.DataFrame(bapb_rows)[
        ['Performance_Constraint', 'Budget', 'BCR_disabled_RT', 'BCR_disabled_Cost', 'BCR_disabled_Config',
         'BCR_disabled_Acc_Config', 'BCR_disabled_Acc_Score', 'BCR_disabled_Iterations', 'BCR_threshold']
    ]
    BAPB_data.to_csv(filenameprefix + '_BAPB.csv', index=False)
