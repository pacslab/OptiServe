import copy
import itertools
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from optiserve.logging import get_logger
from optiserve.modeling.application_model import ApplicationPerformanceModeling
from optiserve.optimization.accuracy import AccuracyModel
from optiserve.optimization.result import OptimizationResult

logger = get_logger(__name__)


class ApplicationOptimizer:
    """Greedy Probability-Refined-Critical-Path (PRCP) optimizer over a workflow.

    Three strategies jointly choose a per-function memory size and ML-model
    variant under two constraints while optimizing the third objective:

    - ``BPBA`` / :pyattr:`BPBC` — minimize response time under budget + accuracy.
    - ``BCPA`` / :pyattr:`BCPC` — minimize cost under response-time + accuracy.
    - ``BAPB``                  — maximize accuracy under response-time + budget.

    Each returns an :class:`~optiserve.optimization.result.OptimizationResult`
    (a tuple-compatible NamedTuple). Per-variant accuracy comes from
    :class:`~optiserve.optimization.accuracy.AccuracyModel` (measured values when
    a node carries them, else a normalized rank).

    Note: the greedy candidate-selection blocks in the three strategies are
    intentionally left un-merged. Their lexicographic tie-breaking — including
    the exact first-vs-last behavior when two configurations produce identical
    metric tuples — is load-bearing for reproducibility of the published
    experiments, so they are preserved verbatim rather than factored into a
    single helper that could subtly change tie resolution.
    """

    def __init__(self,
                 Appworkflow: ApplicationPerformanceModeling,
                 mem_list: Dict[str, List] = None,
                 model_list: Dict[str, List] = None):

        self.App = Appworkflow

        # Retained for backward compatibility; not used by the optimizer.
        self.mem_list = mem_list or {}
        self.model_list = model_list or {}

        # Per-node, per-variant accuracy: measured values when a node provides an
        # 'accuracy_list', otherwise a normalized rank i/N. `model_accuracy_list`
        # keeps the old [node][variant_index] shape used throughout this class.
        node_list = [n for n in self.App.workflow_graph.nodes
                     if n not in ('Start', 'End')]
        self.accuracy_model = AccuracyModel.from_graph(
            self.App.workflow_graph, node_list
        )
        self.model_accuracy_list = {
            node: self.accuracy_model.variant_accuracies(node) for node in node_list
        }

        self.minimal_mem_configuration, \
            self.maximal_mem_configuration, \
                self.minimal_model_configuration, \
                    self.maximal_model_configuration, \
                        self.maximal_cost, \
                            self.minimal_avg_rt, \
                                self.minimal_cost, \
                                    self.maximal_avg_rt = self.get_optimization_boundary()

        self.update_BCR()

        self.all_simple_paths = [path for path in
                                 nx.all_simple_paths(self.App.delooped_graph, self.App.start_point, self.App.end_point)]

        self.simple_paths_num = len(self.all_simple_paths)

        self.CPcounter = 0


    # Update mem, model and rt attributes of each node in the workflow
    def update_mem_rt(self, G: ApplicationPerformanceModeling, mem_dict, model_dict):
        for node in mem_dict:
            G.nodes[node]['mem'] = mem_dict[node]
            G.nodes[node]['rt'] = G.nodes[node]['perf_profile'][model_dict[node]][mem_dict[node]]


    # Update mem and rt attributes of each node in the workflow
    def update_App_workflow_mem_rt(self,
                                   App: ApplicationPerformanceModeling,
                                   mem_dict,
                                   model_dict):
        self.update_mem_rt(App.workflow_graph, mem_dict, model_dict)
        App.update_rt()


    def get_optimization_boundary(self):
        node_list = [node for node in self.App.workflow_graph.nodes if node not in ['Start', 'End']]
        
        minimal_mem_configuration = {}
        maximal_mem_configuration = {}
        
        minimal_model_configuration = {}
        maximal_model_configuration = {}
        
        for node in node_list:
            model_variants = self.App.workflow_graph.nodes[node]['perf_profile']
            
            minimal_mem_configuration[node] = np.inf
            maximal_mem_configuration[node] = -np.inf
            
            minimal_model_configuration[node] = 0
            maximal_model_configuration[node] = len(model_variants) - 1

            for var in model_variants:
                minimal_mem_configuration[node] = min(minimal_mem_configuration[node],
                                                    min(list(var.keys())))
                maximal_mem_configuration[node] = max(maximal_mem_configuration[node],
                                                    max(list(var.keys())))

        self.App.update_ne()

        # Calculating the maximal possible cost
        self.update_App_workflow_mem_rt(self.App, maximal_mem_configuration, maximal_model_configuration)
        maximal_cost = self.App.get_avg_cost()

        # Calculating the minimal possible average response time
        self.update_App_workflow_mem_rt(self.App, maximal_mem_configuration, minimal_model_configuration)
        self.App.get_simple_dag()
        minimal_avg_rt = self.App.get_avg_rt()

        # Calculating the minimal possible cost
        self.update_App_workflow_mem_rt(self.App, minimal_mem_configuration, minimal_model_configuration)
        minimal_cost = self.App.get_avg_cost()

        # Calculating the maximal possible average response time
        self.update_App_workflow_mem_rt(self.App, minimal_mem_configuration, maximal_model_configuration)
        self.App.get_simple_dag()
        maximal_avg_rt = self.App.get_avg_rt()

        logger.debug('Minimal Memory Configuration: {}'.format(minimal_mem_configuration))
        logger.debug('Maximal Memory Configuration: {}'.format(maximal_mem_configuration))
        logger.debug('Maximal Model Configuration: {}'.format(maximal_model_configuration))
        logger.debug('Minimal Model Configuration: {}'.format(minimal_model_configuration))
        logger.debug('Maximal Cost: {}'.format(maximal_cost))
        logger.debug('Minimal Average Response Time: {}'.format(minimal_avg_rt))
        logger.debug('Minimal Cost: {}'.format(minimal_cost))
        logger.debug('Maximal Average Response Time: {}'.format(maximal_avg_rt))
        logger.debug('Optimization Boundary Calculation Completed.')
        
        return (minimal_mem_configuration, maximal_mem_configuration, minimal_model_configuration, maximal_model_configuration, maximal_cost, minimal_avg_rt, minimal_cost,
                maximal_avg_rt)


    # Get the Benefit Cost Ratio (absolute value) of each function
    def update_BCR(self):
        node_list = [item for item in self.App.workflow_graph.nodes]
        for node in node_list:
            self.App.workflow_graph.nodes[node]['BCR'] = {}
            if node in ['Start', 'End']:
                continue
            for model_i, _ in enumerate(self.model_accuracy_list[node]):
                available_mem_list = [item for item in np.sort(list(self.App.workflow_graph.nodes[node]['perf_profile'][model_i].keys()))]
                available_rt_list = [self.App.workflow_graph.nodes[node]['perf_profile'][model_i][item] for item in available_mem_list]
                slope, intercept = np.linalg.lstsq(np.vstack([available_mem_list, np.ones(len(available_mem_list))]).T,
                                                np.array(available_rt_list), rcond=None)[0]
                self.App.workflow_graph.nodes[node]['BCR'][model_i] = np.abs(slope)


    # Find the probability refined critical path in self.App
    def find_PRCP(self, order=0, leastCritical=False):
        self.CPcounter += 1
        tp_list = self.App.get_tp(self.App.delooped_graph, self.all_simple_paths)
        rt_list = self.App.sum_rt_with_ne(self.all_simple_paths, include_start_node=True, include_end_node=True)
        prrt_list = np.multiply(tp_list, rt_list)
        if (leastCritical):
            PRCP = np.argsort(prrt_list)[order]
        else:
            PRCP = np.argsort(prrt_list)[-1 - order]
        return (self.all_simple_paths[PRCP])


    # Update the list of available memory configurations in ascending order
    def update_available_mem_list(self, BCR=False, BCRthreshold=0.1, BCRinverse=False):
        node_list = [item for item in self.App.workflow_graph.nodes]
        for node in node_list:
            self.App.workflow_graph.nodes[node]['available_mem'] = {}
            if node in ['Start', 'End']:
                continue
            for model_i, _ in enumerate(self.model_accuracy_list[node]):
                if (BCR):
                    available_mem_list = [item for item in
                                        np.sort(list(self.App.workflow_graph.nodes[node]['perf_profile'][model_i].keys()))]
                    mem_zip = [item for item in zip(available_mem_list, available_mem_list[1:])]
                    if (BCRinverse):
                        available_mem_list = [item for item in mem_zip if np.abs((item[1] - item[0]) / (
                                self.App.workflow_graph.nodes[node]['perf_profile'][model_i][item[1]] -
                                self.App.workflow_graph.nodes[node]['perf_profile'][model_i][item[0]])) > 1.0 / (
                                                self.App.workflow_graph.nodes[node]['BCR'][model_i]) * BCRthreshold]
                    else:
                        available_mem_list = [item for item in mem_zip if np.abs((self.App.workflow_graph.nodes[node][
                                                                                    'perf_profile'][model_i][item[1]] -
                                                                                self.App.workflow_graph.nodes[node][
                                                                                    'perf_profile'][model_i][item[0]]) / (
                                                                                        item[1] - item[0])) >
                                            self.App.workflow_graph.nodes[node]['BCR'][model_i] * BCRthreshold]
                    available_mem_list = list(np.sort(list(set(itertools.chain(*available_mem_list)))))
                else:
                    available_mem_list = [item for item in
                                        np.sort(list(self.App.workflow_graph.nodes[node]['perf_profile'][model_i].keys()))]
                self.App.workflow_graph.nodes[node]['available_mem'][model_i] = available_mem_list  # Sorted list


    def compute_accuracy(self, model_configuration, accuracy_formula):
        accuracy_values = [self.model_accuracy_list[node][model_configuration[node]] for node in self.App.workflow_graph.nodes if node not in ['Start', 'End']]
        
        return accuracy_formula(*accuracy_values)
    
    
    def accuracy_is_satisfied(self, model_configuration, accuracy_constraint, accuracy_formula):    
        return self.compute_accuracy(model_configuration, accuracy_formula) >= accuracy_constraint
    

    def BPBA(self, budget, accuracy_constraint, accuracy_formula, optimize_model_configuration=True, BCR=False, BCRtype="RT/M", BCRthreshold=0.1):
        '''
        Probability Refined Critical Path Algorithm - Minimal end-to-end response time under a budget constraint
        Best Performance under budget constraint

        Args:
            budget (float): the budge constraint
            BCR (bool): True - use benefit-cost ratio optimization False - not use BCR optimization
            BCRtype (string): 'RT/M' - Benefit is RT, Cost is Mem. Eliminate mem configurations which do not conform to BCR limitations.
                                         The greedy strategy is to select the config with maximal RT reduction.
                              'ERT/C' - Benefit is the reduction on end-to-end response time, Cost is increased cost.
                                             The greedy strategy is to select the config with maximal RT reduction.
                              'MAX' - Benefit is the reduction on end-to-end response time, Cost is increased cost.
                                       The greedy strategy is to select the config with maximal BCR
            BCRthreshold (float): The threshold of BCR cut off
        '''
        if BCRtype == 'rt-mem':
            BCRtype = 'RT/M'
        elif BCRtype == 'e2ert-cost':
            BCRtype = 'ERT/C'
        elif BCRtype == 'max':
            BCRtype = 'MAX'
        if (BCR and BCRtype == "RT/M"):
            self.update_available_mem_list(BCR=True, BCRthreshold=BCRthreshold, BCRinverse=False)
        else:
            self.update_available_mem_list(BCR=False)
        if (BCR):
            cost = self.minimal_cost
                    
        curr_model_configuration = copy.deepcopy(self.minimal_model_configuration)
        curr_mem_configuration = copy.deepcopy(self.minimal_mem_configuration)
        curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
        
        # First phase is finding the model configuration that satisfies the accuracy constraint holding the budget constraint.
        # We start the minimal memory configuration and try to optimize the model configuration.
        
        self.update_App_workflow_mem_rt(self.App, curr_mem_configuration, curr_model_configuration)
        current_cost = self.minimal_cost
        surplus = budget - current_cost
    
        last_e2ert_cost_BCR = 0
        order = 0
        iterations_count = 0
        
        ml_functions = [node for node in self.App.workflow_graph.nodes if node not in ['Start', 'End'] and len(self.model_accuracy_list[node]) > 1]
        mem_list = curr_mem_configuration

        w = 100
        if optimize_model_configuration:
            while not self.accuracy_is_satisfied(curr_model_configuration, accuracy_constraint, accuracy_formula) and (round(surplus, 4) >= 0):
                iterations_count += 1
                cp = self.find_PRCP(order=order, leastCritical=False)
                min_avg_cost_increase_of_each_node = {}
                for node in cp:
                    if node not in ml_functions:
                        continue
                    avg_cost_increase_of_each_model_config = {}
                    node_curr_mem = mem_list[node]
                    model_backup = curr_model_configuration[node]
                    for model_i in list(range(len(self.model_accuracy_list[node]))):
                        if model_i <= curr_model_configuration[node]:
                            continue
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_i})
                        curr_model_configuration[node] = model_i
                        increased_cost = self.App.get_avg_cost() - current_cost
                        acc_after = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                        
                        if (increased_cost <= surplus):
                            acc_gap = acc_after - accuracy_constraint
                            score = -increased_cost + w * min(acc_gap, 0.0)
                            increased_acc = (acc_after - curr_accuracy)
                            avg_cost_increase_of_each_model_config[model_i] = (increased_acc,
                                                                            increased_cost,
                                                                            score)
                            
                        curr_model_configuration[node] = model_backup
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_backup})
                    
                    if len(avg_cost_increase_of_each_model_config) != 0:
                        
                        max_BCR = np.max([item[2] for item in avg_cost_increase_of_each_model_config.values()])
                        min_cost_increase_under_MAX_BCR = np.min([item[1] for item in avg_cost_increase_of_each_model_config.values()
                                                                if item[2] == max_BCR])
                        max_increased_acc_under_MAX_cost_increase_MAX_BCR = np.max(
                            [item[0] for item in avg_cost_increase_of_each_model_config.values()
                            if item[1] == min_cost_increase_under_MAX_BCR and item[2] == max_BCR])
                        
                        reversed_dict = dict(zip(avg_cost_increase_of_each_model_config.values(),
                                                    avg_cost_increase_of_each_model_config.keys()))
                        
                        min_avg_cost_increase_of_each_node[node] = (reversed_dict[(
                                max_increased_acc_under_MAX_cost_increase_MAX_BCR, min_cost_increase_under_MAX_BCR,
                                max_BCR)],
                                                                    max_increased_acc_under_MAX_cost_increase_MAX_BCR,
                                                                    min_cost_increase_under_MAX_BCR,
                                                                    max_BCR)
                            
                if (len(min_avg_cost_increase_of_each_node) == 0):
                    if (order >= self.simple_paths_num - 1):
                        break
                    else:
                        order += 1
                        continue
                
                max_BCR = np.max([item[3] for item in min_avg_cost_increase_of_each_node.values()])
                max_increased_acc_under_MAX_cost_increase_MAX_BCR = np.max(
                    [item[1] for item in min_avg_cost_increase_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in min_avg_cost_increase_of_each_node if
                                min_avg_cost_increase_of_each_node[key][3] == max_BCR and
                                min_avg_cost_increase_of_each_node[key][1] == max_increased_acc_under_MAX_cost_increase_MAX_BCR][0]
                
                target_model = min_avg_cost_increase_of_each_node[target_node][0]
                
                self.update_App_workflow_mem_rt(self.App,
                                                mem_dict={target_node: mem_list[target_node]},
                                                model_dict={target_node: target_model})
                curr_model_configuration[target_node] = target_model
                max_increased_acc_under_MAX_cost_increase_MAX_BCR = min_avg_cost_increase_of_each_node[target_node][1]
                min_cost_increase_under_MAX_BCR = min_avg_cost_increase_of_each_node[target_node][2]
                self.App.get_simple_dag()
                current_avg_rt = self.App.get_avg_rt()
                curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                surplus -= min_cost_increase_under_MAX_BCR
                

        order = 0
        
        cost = self.App.get_avg_cost()
        surplus = budget - cost

        self.App.get_simple_dag()
        current_avg_rt = self.App.get_avg_rt()
        current_cost = cost
                
        while (round(surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(order=order, leastCritical=False)
            max_avg_rt_reduction_of_each_node = {}
            mem_backup = nx.get_node_attributes(self.App.workflow_graph, 'mem')
            for node in cp:
                if node in ['Start', 'End']:
                    continue
                avg_rt_reduction_of_each_mem_config = {}
                for mem in reversed(self.App.workflow_graph.nodes[node]['available_mem'][curr_model_configuration[node]]):
                    if (mem <= mem_backup[node]):
                        break
                    self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem}, model_dict={node: curr_model_configuration[node]})
                    increased_cost = self.App.get_avg_cost() - current_cost
                    if (increased_cost < surplus):
                        self.App.get_simple_dag()
                        rt_reduction = current_avg_rt - self.App.get_avg_rt()
                        if (rt_reduction > 0):
                            avg_rt_reduction_of_each_mem_config[mem] = (rt_reduction, increased_cost)
                self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: curr_model_configuration[node]})
                if (BCR and BCRtype == "ERT/C"):
                    avg_rt_reduction_of_each_mem_config = {item: avg_rt_reduction_of_each_mem_config[item] for item in
                                                           avg_rt_reduction_of_each_mem_config.keys() if
                                                           avg_rt_reduction_of_each_mem_config[item][0] /
                                                           avg_rt_reduction_of_each_mem_config[item][
                                                               1] > last_e2ert_cost_BCR * BCRthreshold}
                if (BCR and BCRtype == "MAX"):
                    avg_rt_reduction_of_each_mem_config = {item: (
                        avg_rt_reduction_of_each_mem_config[item][0], avg_rt_reduction_of_each_mem_config[item][1],
                        avg_rt_reduction_of_each_mem_config[item][0] / avg_rt_reduction_of_each_mem_config[item][1]) for
                        item in avg_rt_reduction_of_each_mem_config.keys()}
                if (len(avg_rt_reduction_of_each_mem_config) != 0):
                    if (BCR and BCRtype == "MAX"):
                        max_BCR = np.max([item[2] for item in avg_rt_reduction_of_each_mem_config.values()])
                        max_rt_reduction_under_MAX_BCR = np.max(
                            [item[0] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[2] == max_BCR])
                        min_increased_cost_under_MAX_rt_reduction_MAX_BCR = np.min(
                            [item[1] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[0] == max_rt_reduction_under_MAX_BCR and item[2] == max_BCR])
                        reversed_dict = dict(zip(avg_rt_reduction_of_each_mem_config.values(),
                                                 avg_rt_reduction_of_each_mem_config.keys()))
                        max_avg_rt_reduction_of_each_node[node] = (reversed_dict[(
                            max_rt_reduction_under_MAX_BCR, min_increased_cost_under_MAX_rt_reduction_MAX_BCR,
                            max_BCR)],
                                                                   max_rt_reduction_under_MAX_BCR,
                                                                   min_increased_cost_under_MAX_rt_reduction_MAX_BCR,
                                                                   max_BCR)
                    else:
                        max_rt_reduction = np.max([item[0] for item in avg_rt_reduction_of_each_mem_config.values()])
                        min_increased_cost_under_MAX_rt_reduction = np.min(
                            [item[1] for item in avg_rt_reduction_of_each_mem_config.values() if
                             item[0] == max_rt_reduction])
                        reversed_dict = dict(zip(avg_rt_reduction_of_each_mem_config.values(),
                                                 avg_rt_reduction_of_each_mem_config.keys()))
                        max_avg_rt_reduction_of_each_node[node] = (
                            reversed_dict[(max_rt_reduction, min_increased_cost_under_MAX_rt_reduction)],
                            max_rt_reduction,
                            min_increased_cost_under_MAX_rt_reduction)

            if (len(max_avg_rt_reduction_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            if (BCR and BCRtype == "MAX"):
                max_BCR = np.max([item[3] for item in max_avg_rt_reduction_of_each_node.values()])
                max_rt_reduction_under_MAX_BCR = np.max(
                    [item[1] for item in max_avg_rt_reduction_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in max_avg_rt_reduction_of_each_node if
                               max_avg_rt_reduction_of_each_node[key][3] == max_BCR and
                               max_avg_rt_reduction_of_each_node[key][1] == max_rt_reduction_under_MAX_BCR][0]
                target_mem = max_avg_rt_reduction_of_each_node[target_node][0]
            else:
                max_rt_reduction = np.max([item[1] for item in max_avg_rt_reduction_of_each_node.values()])
                min_increased_cost_under_MAX_rt_reduction = np.min(
                    [item[2] for item in max_avg_rt_reduction_of_each_node.values() if item[1] == max_rt_reduction])
                target_mem = np.min([item[0] for item in max_avg_rt_reduction_of_each_node.values() if
                                     item[1] == max_rt_reduction and item[
                                         2] == min_increased_cost_under_MAX_rt_reduction])
                target_node = [key for key in max_avg_rt_reduction_of_each_node if
                               max_avg_rt_reduction_of_each_node[key] == (
                                   target_mem, max_rt_reduction, min_increased_cost_under_MAX_rt_reduction)][0]
            self.update_App_workflow_mem_rt(self.App, mem_dict={target_node: target_mem}, model_dict={target_node: curr_model_configuration[target_node]})
            max_rt_reduction = max_avg_rt_reduction_of_each_node[target_node][1]
            min_increased_cost_under_MAX_rt_reduction = max_avg_rt_reduction_of_each_node[target_node][2]
            current_avg_rt = current_avg_rt - max_rt_reduction
            surplus = surplus - min_increased_cost_under_MAX_rt_reduction
            current_cost = self.App.get_avg_cost()
            current_e2ert_cost_BCR = max_rt_reduction / min_increased_cost_under_MAX_rt_reduction
            if (current_e2ert_cost_BCR == float('Inf')):
                last_e2ert_cost_BCR = 0
            else:
                last_e2ert_cost_BCR = current_e2ert_cost_BCR
        current_mem_configuration = nx.get_node_attributes(self.App.workflow_graph, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        logger.debug('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        logger.debug('Average end-to-end response time: {}'.format(current_avg_rt))
        logger.debug('Optimized Accuracy Configuration: {}'.format(curr_model_configuration))
        logger.debug('Average Cost: {}'.format(current_cost))
        logger.debug('BPBC optimization completed.')
        return OptimizationResult(current_avg_rt, current_cost, self.compute_accuracy(curr_model_configuration, accuracy_formula), current_mem_configuration, curr_model_configuration, iterations_count)


    def BCPA(self, rt_constraint, accuracy_constraint, accuracy_formula, optimize_model_configuration=True, BCR=False, BCRtype="RT/M", BCRthreshold=0.2):
        '''
        Probability Refined Critical Path Algorithm - Minimal cost under an end-to-end response time constraint
        Best cost under performance (end-to-end response time) constraint

        Args:
            rt_constraint (float): End-to-end response time constraint
            BCR (bool): True - use benefit-cost ratio optimization False - not use BCR optimization
            BCRtype (string): 'M/RT' - Benefit is Mem, Cost is RT. (inverse) Eliminate mem configurations which do not conform to BCR limitations
                              'C/ERT' - Benefit is the cost reduction, Cost is increased ERT.
                              'MAX' - Benefit is the cost reduction, Cost is increased ERT. The greedy strategy is to select the config with maximal BCR
            BCRthreshold (float): The threshold of BCR cut off
        '''
        if BCRtype == 'rt-mem':
            BCRtype = 'M/RT'
        elif BCRtype == 'e2ert-cost':
            BCRtype = 'C/ERT'
        elif BCRtype == 'max':
            BCRtype = 'MAX'
        if (BCR and BCRtype == "M/RT"):
            self.update_available_mem_list(BCR=True, BCRthreshold=BCRthreshold, BCRinverse=True)
        else:
            self.update_available_mem_list(BCR=False)
        
        order = 0
        iterations_count = 0
        last_e2ert_cost_BCR = 0
        
        curr_model_configuration = copy.deepcopy(self.minimal_model_configuration)
        curr_mem_configuration = copy.deepcopy(self.maximal_mem_configuration)
        curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
        
        # First phase is finding the model configuration that satisfies the accuracy constraint holding the budget constraint.
        # We start the maximal memory configuration and try to optimize the model configuration.

        self.update_App_workflow_mem_rt(self.App, curr_mem_configuration, curr_model_configuration)
        current_avg_rt = self.minimal_avg_rt
        performance_surplus = rt_constraint - current_avg_rt
            
        ml_functions = [node for node in self.App.workflow_graph.nodes if node not in ['Start', 'End'] and len(self.model_accuracy_list[node]) > 1]
        mem_list = nx.get_node_attributes(self.App.workflow_graph, 'mem') 
        
        logger.debug('RT Constraint: {}'.format(rt_constraint))
        logger.debug('Accuracy Constraint: {}'.format(accuracy_constraint))
        logger.debug('Performance Surplus: {}'.format(performance_surplus))
        logger.debug('Current Average Response Time: {}'.format(current_avg_rt))
        w = 100

        if optimize_model_configuration:
            while not self.accuracy_is_satisfied(curr_model_configuration, accuracy_constraint, accuracy_formula) and (round(performance_surplus, 4) >= 0):
                iterations_count += 1
                cp = self.find_PRCP(order=order, leastCritical=False)
                min_avg_rt_increase_of_each_node = {}
                for node in cp:
                    if node not in ml_functions:
                        continue
                    avg_rt_increase_of_each_model_config = {}
                    node_curr_mem = mem_list[node]
                    model_backup = curr_model_configuration[node]
                    for model_i in list(range(len(self.model_accuracy_list[node]))):
                        if model_i <= curr_model_configuration[node]:
                            continue
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_i})
                        curr_model_configuration[node] = model_i
                        self.App.get_simple_dag()
                        increased_rt = self.App.get_avg_rt() - current_avg_rt
                        acc_after = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                        
                        if (increased_rt <= performance_surplus):
                            acc_gap = acc_after - accuracy_constraint
                            score = -increased_rt + w * min(acc_gap, 0.0)
                            increased_acc = (acc_after - curr_accuracy)
                            avg_rt_increase_of_each_model_config[model_i] = (increased_acc,
                                                                            increased_rt,
                                                                            score)
                            
                        curr_model_configuration[node] = model_backup
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: node_curr_mem}, model_dict={node: model_backup})
                    
                    if len(avg_rt_increase_of_each_model_config) != 0:
                        
                        max_BCR = np.max([item[2] for item in avg_rt_increase_of_each_model_config.values()])
                        min_rt_increase_under_MAX_BCR = np.min([item[1] for item in avg_rt_increase_of_each_model_config.values()
                                                                if item[2] == max_BCR])
                        max_increased_acc_under_MAX_rt_increase_MAX_BCR = np.max(
                            [item[0] for item in avg_rt_increase_of_each_model_config.values()
                            if item[1] == min_rt_increase_under_MAX_BCR and item[2] == max_BCR])
                        
                        reversed_dict = dict(zip(avg_rt_increase_of_each_model_config.values(),
                                                    avg_rt_increase_of_each_model_config.keys()))
                        
                        min_avg_rt_increase_of_each_node[node] = (reversed_dict[(
                                max_increased_acc_under_MAX_rt_increase_MAX_BCR, min_rt_increase_under_MAX_BCR,
                                max_BCR)],
                                                                    max_increased_acc_under_MAX_rt_increase_MAX_BCR,
                                                                    min_rt_increase_under_MAX_BCR,
                                                                    max_BCR)
                            
                if (len(min_avg_rt_increase_of_each_node) == 0):
                    if (order >= self.simple_paths_num - 1):
                        break
                    else:
                        order += 1
                        continue
                
                max_BCR = np.max([item[3] for item in min_avg_rt_increase_of_each_node.values()])
                max_increased_acc_under_MAX_rt_increase_MAX_BCR = np.max(
                    [item[1] for item in min_avg_rt_increase_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in min_avg_rt_increase_of_each_node if
                                min_avg_rt_increase_of_each_node[key][3] == max_BCR and
                                min_avg_rt_increase_of_each_node[key][1] == max_increased_acc_under_MAX_rt_increase_MAX_BCR][0]
                
                target_model = min_avg_rt_increase_of_each_node[target_node][0]
                
                self.update_App_workflow_mem_rt(self.App,
                                                mem_dict={target_node: mem_list[target_node]},
                                                model_dict={target_node: target_model})
                curr_model_configuration[target_node] = target_model
                max_increased_acc_under_MAX_rt_increase_MAX_BCR = min_avg_rt_increase_of_each_node[target_node][1]
                min_rt_increase_under_MAX_BCR = min_avg_rt_increase_of_each_node[target_node][2]
                self.App.get_simple_dag()
                current_avg_rt = self.App.get_avg_rt()
                curr_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                performance_surplus = performance_surplus - min_rt_increase_under_MAX_BCR
                

        
        current_cost = self.App.get_avg_cost()
        
        self.App.get_simple_dag()
        current_avg_rt = self.App.get_avg_rt()
        performance_surplus = rt_constraint - current_avg_rt

        while (round(performance_surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(leastCritical=True, order=order)
            max_cost_reduction_of_each_node = {}
            mem_backup = nx.get_node_attributes(self.App.workflow_graph, 'mem')
            for node in cp:
                if node in ['Start', 'End']:
                    continue
                cost_reduction_of_each_mem_config = {}
                for mem in self.App.workflow_graph.nodes[node][
                    'available_mem'][curr_model_configuration[node]]:
                    if (mem >= mem_backup[node]):
                        break
                    self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem}, model_dict={node: curr_model_configuration[node]})
                    self.App.get_simple_dag()
                    temp_avg_rt = self.App.get_avg_rt()
                    increased_rt = temp_avg_rt - current_avg_rt
                    cost_reduction = current_cost - self.App.get_avg_cost()
                    if (increased_rt < performance_surplus and cost_reduction > 0):
                        cost_reduction_of_each_mem_config[mem] = (cost_reduction, increased_rt)
                self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: curr_model_configuration[node]})
                if (BCR and BCRtype == 'C/ERT'):
                    cost_reduction_of_each_mem_config = {item: cost_reduction_of_each_mem_config[item] for item in
                                                         cost_reduction_of_each_mem_config.keys() if
                                                         cost_reduction_of_each_mem_config[item][0] /
                                                         cost_reduction_of_each_mem_config[item][
                                                             1] > last_e2ert_cost_BCR * BCRthreshold}
                elif (BCR and BCRtype == "MAX"):
                    cost_reduction_of_each_mem_config = {item: (
                        cost_reduction_of_each_mem_config[item][0], cost_reduction_of_each_mem_config[item][1],
                        cost_reduction_of_each_mem_config[item][0] / cost_reduction_of_each_mem_config[item][1]) for
                        item in
                        cost_reduction_of_each_mem_config.keys()}
                if (len(cost_reduction_of_each_mem_config) != 0):
                    if (BCR and BCRtype == "MAX"):
                        max_BCR = np.max([item[2] for item in cost_reduction_of_each_mem_config.values()])
                        max_cost_reduction_under_MAX_BCR = np.max(
                            [item[0] for item in cost_reduction_of_each_mem_config.values() if
                             item[2] == max_BCR])
                        min_increased_rt_under_MAX_rt_reduction_MAX_BCR = np.min(
                            [item[1] for item in cost_reduction_of_each_mem_config.values() if
                             item[0] == max_cost_reduction_under_MAX_BCR and item[2] == max_BCR])
                        reversed_dict = dict(zip(cost_reduction_of_each_mem_config.values(),
                                                 cost_reduction_of_each_mem_config.keys()))
                        max_cost_reduction_of_each_node[node] = (reversed_dict[(
                            max_cost_reduction_under_MAX_BCR, min_increased_rt_under_MAX_rt_reduction_MAX_BCR,
                            max_BCR)],
                                                                 max_cost_reduction_under_MAX_BCR,
                                                                 min_increased_rt_under_MAX_rt_reduction_MAX_BCR,
                                                                 max_BCR)
                    else:
                        max_cost_reduction = np.max([item[0] for item in cost_reduction_of_each_mem_config.values()])
                        min_increased_rt_under_MAX_cost_reduction = np.min(
                            [item[1] for item in cost_reduction_of_each_mem_config.values() if
                             item[0] == max_cost_reduction])
                        reversed_dict = dict(
                            zip(cost_reduction_of_each_mem_config.values(), cost_reduction_of_each_mem_config.keys()))
                        max_cost_reduction_of_each_node[node] = (
                            reversed_dict[(max_cost_reduction, min_increased_rt_under_MAX_cost_reduction)],
                            max_cost_reduction,
                            min_increased_rt_under_MAX_cost_reduction)
            if (len(max_cost_reduction_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            if (BCR and BCRtype == "MAX"):
                max_BCR = np.max([item[3] for item in max_cost_reduction_of_each_node.values()])
                max_cost_reduction_under_MAX_BCR = np.max(
                    [item[1] for item in max_cost_reduction_of_each_node.values() if item[3] == max_BCR])
                target_node = [key for key in max_cost_reduction_of_each_node if
                               max_cost_reduction_of_each_node[key][3] == max_BCR and
                               max_cost_reduction_of_each_node[key][1] == max_cost_reduction_under_MAX_BCR][0]
                target_mem = max_cost_reduction_of_each_node[target_node][0]
            else:
                max_cost_reduction = np.max([item[1] for item in max_cost_reduction_of_each_node.values()])
                min_increased_rt_under_MAX_cost_reduction = np.min(
                    [item[2] for item in max_cost_reduction_of_each_node.values() if item[1] == max_cost_reduction])
                target_mem = np.min([item[0] for item in max_cost_reduction_of_each_node.values() if
                                     item[1] == max_cost_reduction and item[
                                         2] == min_increased_rt_under_MAX_cost_reduction])
                target_node = [key for key in max_cost_reduction_of_each_node if
                               max_cost_reduction_of_each_node[key] == (
                                   target_mem, max_cost_reduction, min_increased_rt_under_MAX_cost_reduction)][0]
            self.update_App_workflow_mem_rt(self.App, mem_dict={target_node: target_mem}, model_dict={target_node: curr_model_configuration[target_node]})
            max_cost_reduction = max_cost_reduction_of_each_node[target_node][1]
            min_increased_rt_under_MAX_cost_reduction = max_cost_reduction_of_each_node[target_node][2]
            current_cost = current_cost - max_cost_reduction
            performance_surplus = performance_surplus - min_increased_rt_under_MAX_cost_reduction
            current_avg_rt = current_avg_rt + min_increased_rt_under_MAX_cost_reduction
            current_e2ert_cost_BCR = max_cost_reduction / min_increased_rt_under_MAX_cost_reduction
            if (current_e2ert_cost_BCR == float('Inf')):
                last_e2ert_cost_BCR = 0
            else:
                last_e2ert_cost_BCR = current_e2ert_cost_BCR
        current_mem_configuration = nx.get_node_attributes(self.App.workflow_graph, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        logger.debug('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        logger.debug('Average end-to-end response time: {}'.format(current_avg_rt))
        logger.debug('Optimized Accuracy Configuration: {}'.format(curr_model_configuration))
        logger.debug('Average Cost: {}'.format(current_cost))
        logger.debug('BCPC optimization completed.')
        return OptimizationResult(current_avg_rt, current_cost, self.compute_accuracy(curr_model_configuration, accuracy_formula), current_mem_configuration, curr_model_configuration, iterations_count)

    
    def BAPB(self, rt_constraint, budget, accuracy_formula, BCR=False, BCRtype="RT/M", BCRthreshold=0.1):
        
        delta_rt = lambda new_rt, old_rt: abs(new_rt - old_rt) / (self.maximal_avg_rt - self.minimal_avg_rt) if (self.maximal_avg_rt - self.minimal_avg_rt) != 0 else 0
        delta_cost = lambda new_cost, old_cost: abs(new_cost - old_cost) / (self.maximal_cost - self.minimal_cost) if (self.maximal_cost - self.minimal_cost) != 0 else 0
        
        order = 0
        iterations_count = 0
        
        # First phase is finding the best possible RT under the budget constraint and the lowest possible model configuration.
        current_avg_rt, current_avg_cost, current_accuracy, curr_mem_configuration, curr_model_configuration, _ = self.BCPA(rt_constraint,
                                                                                                                    None, 
                                                                                                                    accuracy_formula, 
                                                                                                                    optimize_model_configuration=False, 
                                                                                                                    BCR=BCR, BCRtype=BCRtype, BCRthreshold=BCRthreshold)
        
        performance_surplus = rt_constraint - current_avg_rt
        budget_surplus = budget - current_avg_cost
        
        while (round(performance_surplus, 4) >= 0) and (round(budget_surplus, 4) >= 0):
            iterations_count += 1
            cp = self.find_PRCP(order=order, leastCritical=False)
            max_acc_increase_of_each_node = {}
            
            mem_backup = copy.deepcopy(curr_mem_configuration)
            model_backup = copy.deepcopy(curr_model_configuration)

            for node in cp:
                if node in ['Start', 'End']:
                    continue
                
                bcr_values_for_each_change = {}

                for model_i, _ in enumerate(self.model_accuracy_list[node]):
                    if model_i <= model_backup[node]:
                        continue
                    
                    for mem in reversed(self.App.workflow_graph.nodes[node]['available_mem'][model_i]):
                        if mem <= mem_backup[node]:
                            break
                        self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem}, model_dict={node: model_i})
                        curr_model_configuration[node] = model_i
                        temp_avg_cost = self.App.get_avg_cost()
                        self.App.get_simple_dag()
                        temp_avg_rt = self.App.get_avg_rt()
                        temp_accuracy = self.compute_accuracy(curr_model_configuration, accuracy_formula)
                        
                        increased_cost = temp_avg_cost - current_avg_cost
                        increased_rt = temp_avg_rt - current_avg_rt
                        increased_accuracy = temp_accuracy - current_accuracy
                        
                        if increased_accuracy < 0:
                            break
                        
                        bcr = (increased_accuracy) / (delta_cost(temp_avg_cost, current_avg_cost) + delta_rt(temp_avg_rt, current_avg_rt))
                        
                        if (increased_cost <= budget_surplus) and (increased_rt <= performance_surplus) and not np.isnan(bcr):
                            
                            bcr_values_for_each_change[(mem, model_i)] = (
                                bcr,
                                increased_accuracy,
                                increased_cost,
                                increased_rt
                            )
                            
                    self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: model_i})
                curr_model_configuration[node] = model_backup[node]
                self.update_App_workflow_mem_rt(self.App, mem_dict={node: mem_backup[node]}, model_dict={node: model_backup[node]})
                        
                if len(bcr_values_for_each_change) != 0:
                    max_BCR = np.max([item[0] for item in bcr_values_for_each_change.values()])
                    max_accuracy_increase_under_MAX_BCR = np.max(
                        [item[1] for item in bcr_values_for_each_change.values() if
                            item[0] == max_BCR])
                    min_increased_cost_under_MAX_BCR = np.min(
                        [item[2] for item in bcr_values_for_each_change.values() if
                            item[1] == max_accuracy_increase_under_MAX_BCR and item[0] == max_BCR])
                    min_increased_rt_under_MAX_BCR = np.min(
                        [item[3] for item in bcr_values_for_each_change.values() if
                            item[2] == min_increased_cost_under_MAX_BCR and item[1] == max_accuracy_increase_under_MAX_BCR and item[0] == max_BCR])
                    reversed_dict = dict(zip(bcr_values_for_each_change.values(),
                                                bcr_values_for_each_change.keys()))
                    max_acc_increase_of_each_node[node] = (reversed_dict[(
                        max_BCR, max_accuracy_increase_under_MAX_BCR, min_increased_cost_under_MAX_BCR,
                        min_increased_rt_under_MAX_BCR)],
                                                                max_accuracy_increase_under_MAX_BCR,
                                                                min_increased_cost_under_MAX_BCR,
                                                                min_increased_rt_under_MAX_BCR,
                                                                max_BCR)
                
            if (len(max_acc_increase_of_each_node) == 0):
                if (order >= self.simple_paths_num - 1):
                    break
                else:
                    order += 1
                    continue
            max_BCR = np.max([item[4] for item in max_acc_increase_of_each_node.values()])
            max_accuracy_increase_under_MAX_BCR = np.max(
                [item[1] for item in max_acc_increase_of_each_node.values() if item[4] == max_BCR])
            target_node = [key for key in max_acc_increase_of_each_node if
                            max_acc_increase_of_each_node[key][4] == max_BCR and
                            max_acc_increase_of_each_node[key][1] == max_accuracy_increase_under_MAX_BCR][0]
            target_mem = max_acc_increase_of_each_node[target_node][0][0]
            target_model = max_acc_increase_of_each_node[target_node][0][1]
            
            curr_mem_configuration[target_node] = target_mem
            curr_model_configuration[target_node] = target_model
            
            logger.debug('Target Node: {}, Target Mem: {}, Target Model: {}'.format(target_node, target_mem, target_model))
            
            self.update_App_workflow_mem_rt(self.App, mem_dict={target_node: target_mem}, model_dict={target_node: target_model})
            max_accuracy_increase = max_acc_increase_of_each_node[target_node][1]
            min_increased_cost_under_MAX_BCR = max_acc_increase_of_each_node[target_node][2]
            min_increased_rt_under_MAX_BCR = max_acc_increase_of_each_node[target_node][3]
            
            current_avg_cost = current_avg_cost + min_increased_cost_under_MAX_BCR
            current_avg_rt = current_avg_rt + min_increased_rt_under_MAX_BCR
            current_accuracy = current_accuracy + max_accuracy_increase
            
            budget_surplus = budget_surplus - min_increased_cost_under_MAX_BCR
            performance_surplus = performance_surplus - min_increased_rt_under_MAX_BCR
        
        current_mem_configuration = nx.get_node_attributes(self.App.workflow_graph, 'mem')
        del current_mem_configuration['Start']
        del current_mem_configuration['End']
        logger.debug('Optimized Memory Configuration: {}'.format(current_mem_configuration))
        logger.debug('Average end-to-end response time: {}'.format(current_avg_rt))
        logger.debug('Optimized Accuracy Configuration: {}'.format(curr_model_configuration))
        logger.debug('Average Cost: {}'.format(current_avg_cost))
        logger.debug('BAPB optimization completed.')
        return OptimizationResult(current_avg_rt, current_avg_cost, self.compute_accuracy(curr_model_configuration, accuracy_formula), current_mem_configuration, curr_model_configuration, iterations_count)
            
            

# Canonical strategy names (matching the paper / artifact suffixes). The method
# definitions above keep their historical names; these aliases are the preferred
# spelling: BPBC = best performance under budget + accuracy, BCPC = best cost
# under performance + accuracy, BAPB = best accuracy under performance + budget.
ApplicationOptimizer.BPBC = ApplicationOptimizer.BPBA
ApplicationOptimizer.BCPC = ApplicationOptimizer.BCPA