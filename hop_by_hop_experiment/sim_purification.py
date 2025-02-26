import gc
import json
import operator
import os
import sys
from collections import Counter
from functools import reduce

import numpy as np
import pydynaa as pd
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.qubits import ketstates as ks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network
from utils import Logging
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
import netsquid.qubits.operators as ops


class PurificationExample(LocalProtocol):
    """
    Protocol for a complete purification example.

    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 max_purify_paris=1,
                 skip_noise=False):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.max_purify_paris = max_purify_paris
        self.skip_noise = skip_noise
        null_logger = Logging.Logger("null", logging_enabled=True)
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExamplePurification")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=True)
        # initialize the protocol for each node
        # Initialize the entangle protocol
        for index, node in enumerate(network_nodes):
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            # Initialize the GenEntanglement protocol and EntanglementHandler protocol
            if index - 1 >= 0:
                # case of we have a previous node
                gen_protocol = GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger
                )
                self.add_subprotocol(gen_protocol)
                eh_handler = EntanglementHandler(node=node,
                                                 name=f"entanglement_handler_{node.name}->{network_nodes[index - 1].name}",
                                                 num_pairs=self.max_entangle_pairs,
                                                 qubit_input_protocol=gen_protocol,
                                                 cc_message_handler=self.subprotocols[
                                                     f"message_handler_{node.name}"],
                                                 entangle_node=network_nodes[index - 1].name,
                                                 memory_depolar_rate=memory_depolar_rate,
                                                 node_distance=node_distance,
                                                 is_top_layer=False,
                                                 logger=self.logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                # add purification
                prue_protocol = Purification(node=node,
                                             name=f"purify_{node.name}->{network_nodes[index - 1].name}",
                                             entangled_node=network_nodes[index - 1].name,
                                             entanglement_handler=eh_handler,
                                             cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                             max_purify_pair=self.max_purify_paris,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=True,
                                             logger=self.logger
                                             )
                self.add_subprotocol(prue_protocol)

            if index + 1 < len(network_nodes):
                # case of we have a next node
                gen_protocol = GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger
                )
                self.add_subprotocol(gen_protocol)
                eh_handler = EntanglementHandler(node=node,
                                                 name=f"entanglement_handler_{node.name}->{network_nodes[index + 1].name}",
                                                 num_pairs=self.max_entangle_pairs,
                                                 qubit_input_protocol=gen_protocol,
                                                 cc_message_handler=self.subprotocols[
                                                     f"message_handler_{node.name}"],
                                                 entangle_node=network_nodes[index + 1].name,
                                                 memory_depolar_rate=memory_depolar_rate,
                                                 node_distance=node_distance,
                                                 is_top_layer=False,
                                                 logger=self.logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                # Initialize the purification protocol
                self.add_subprotocol(Purification(node=node,
                                                  name=f"purify_{node.name}->{network_nodes[index + 1].name}",
                                                  entangled_node=network_nodes[index + 1].name,
                                                  entanglement_handler=eh_handler,
                                                  cc_message_handler=self.subprotocols[
                                                      f"message_handler_{node.name}"],
                                                  max_purify_pair=self.max_purify_paris,
                                                  target_fidelity=target_fidelity,
                                                  is_top_layer=True,
                                                  logger=self.logger
                                                  ))

    def run(self):
        self.start_subprotocols()
        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            pure_protocols = []
            for subprotocol in self.subprotocols.values():
                if "purify" in subprotocol.name:
                    pure_protocols.append(subprotocol)

            wait_signals = [self.await_signal(p, MessageType.PURIFICATION_FINISHED)
                            for p in pure_protocols]

            yield reduce(operator.and_, wait_signals)

            results = [p.get_signal_result(MessageType.PURIFICATION_FINISHED)
                       for p in pure_protocols]
            result_dic = {}
            """
            {"satisfied_pairs": self.satisfied_pairs,
            "purification_count": self.purification_count,
            "purification_success_count": self.purification_success_count,
             "finish_time": sim_time()}
            """
            node_index = 0
            for i in range(0, len(results), 2):
                entangle_node = self.all_nodes[node_index + 1].name
                node = self.all_nodes[node_index].name
                node_pair_res = results[i]["satisfied_pairs"]
                purified_count = results[i]["purification_count"]
                purified_success_count = results[i]["purification_success_count"]
                finish_time = results[i]["finish_time"]
                # entangle_pair_res = results[i + 1]["satisfied_pairs"]
                # print(f"Finish time: {finish_time}")
                # measure the actual fidelity
                actual_fidelities = {}
                theoretical_fidelities = {}
                teleport_success_count = 0
                for mem_pos, theoretical_fidelity in node_pair_res.items():
                    q_a = self.all_nodes[i].subcomponents[f"{entangle_node}_qmemory"].pop(mem_pos,
                                                                                          skip_noise=self.skip_noise)[0]
                    q_b = self.all_nodes[i + 1].subcomponents[f"{node}_qmemory"].pop(mem_pos,
                                                                                     skip_noise=self.skip_noise)[0]
                    q_a_name = str(q_a.name).split("#")[-1].split("-")[0]
                    q_b_name = str(q_b.name).split("#")[-1].split("-")[0]
                    # print(f"Qubit names: {q_a_name}, {q_b_name}")
                    if q_a_name != q_b_name:
                        raise ValueError(f"Qubit names are not the same at {mem_pos}: {q_a_name}, {q_b_name}")
                    # if q_a.qstate != q_b.qstate:
                    #     raise ValueError(f"Qubit states are not the same: {q_a.qstate}, {q_b.qstate}")
                    f = qapi.fidelity([q_a, q_b], ks.b00)
                    if 0 < f < 0.99:
                        raise ValueError(f"Fidelity is not correct: {f}, \n\t{q_a.qstate}\n\t{q_b.qstate}")
                        # self.logger.error(f"Fidelity is not correct: {f}, \n\t{q_a.qstate}\n\t{q_b.qstate}",
                        #                   color="red")
                        # print_red(f"Fidelity is not correct: {f}, \n\t{q_a.qstate}\n\t{q_b.qstate}")
                    # print(f"Actual fidelity at {mem_pos} is {f}, {q_a.qstate}, {q_b.qstate}")
                    # start_teleportation, generate a qubit for teleportation
                    # rotate the qubit to y0 state
                    qubit = qapi.create_qubits(1)[0]
                    qapi.operate(qubit, ops.H)
                    qapi.operate(qubit, ops.S)
                    # teleport the qubit
                    fid = self.test_teleportation(q_a, q_b, qubit)
                    if fid > 0.99:
                        teleport_success_count += 1
                    actual_fidelities[mem_pos] = f
                    theoretical_fidelities[mem_pos] = theoretical_fidelity
                result_dic[f"{node}->{entangle_node}"] = {
                    "actual_fidelities": actual_fidelities,
                    "theoretical_fidelities": theoretical_fidelities,
                    "purified_count": purified_count,
                    "purified_success_count": purified_success_count,
                    "experiment_duration": finish_time - start_time,
                    "satisfied_pairs_count": len(node_pair_res),
                    "teleport_success_count": teleport_success_count}
                node_index += 1

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

    @staticmethod
    def test_teleportation(qubit_a, qubit_b, teleport_qubit):
        """
        Test teleportation with two qubits
        :return:
        """
        # Store the initial state
        initial_state = teleport_qubit.qstate

        # Perform teleportation
        qapi.operate(qubits=[teleport_qubit, qubit_a], operator=ops.CNOT)
        qapi.operate(teleport_qubit, ops.H)
        m1, _ = qapi.measure(teleport_qubit)
        m2, _ = qapi.measure(qubit_a)
        if m1 == 1:
            qapi.operate(qubit_b, ops.Z)
        if m2 == 1:
            qapi.operate(qubit_b, ops.X)

        # Calculate fidelity
        # teleported_state = qapi.reduced_dm(qubit_b)
        # fidelity = qapi.fidelity(teleported_state, initial_state)
        fidelity = qapi.fidelity(qubit_b, ns.y0)

        return fidelity

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


def example_sim_run(nodes, num_runs, memory_depolar_rate, node_distance, max_entangle_pairs, target_fidelity,
                    max_purify_pair,
                    skip_noise=False):
    """Example simulation setup for purification protocols.

    Returns
    -------
    :class:`~netsquid.examples.purify.FilteringExample`
        Example protocol to run.
    :class:`pandas.DataFrame`
        Dataframe of collected data.

    """
    purify_example = PurificationExample(network_nodes=nodes,
                                         num_runs=num_runs,
                                         memory_depolar_rate=memory_depolar_rate,
                                         node_distance=node_distance,
                                         max_entangle_pairs=max_entangle_pairs,
                                         max_purify_paris=max_purify_pair,
                                         target_fidelity=target_fidelity,
                                         skip_noise=skip_noise)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=purify_example,
                                     event_type=Signals.SUCCESS.value))
    return purify_example, dc


def run_single_stack(nodes_count, skip_noise=False):
    # create a network
    nodes_list = [f"Node_{i}" for i in range(nodes_count)]
    network = setup_network(nodes_list, "hop-by-hop-purification",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=20, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    max_pairs = 128
    if os.path.exists(f"./purification_results/purification_results_2_nodes_{max_pairs}_paris_noise_{skip_noise}.json"):
        with open(f"./purification_results/purification_results_2_nodes_{max_pairs}_paris_noise_{skip_noise}.json",
                  "r") as f:
            experiment_result = json.load(f)
    else:
        experiment_result = {}
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=(max_pairs - 4) // 2)
        for entangle_pairs in range(4, max_pairs + 1, 2):
            if str(entangle_pairs) in experiment_result:
                print(f"Skipping entangle pairs: {entangle_pairs}, loaded from file")
                progress.update(task, advance=1)
                continue
            filt_example, dc = example_sim_run(sample_nodes, num_runs=1000, memory_depolar_rate=100,
                                               node_distance=20,
                                               max_entangle_pairs=entangle_pairs, target_fidelity=0.995,
                                               skip_noise=skip_noise)
            filt_example.start()
            ns.sim_run()
            collected_data = dc.dataframe
            all_node_actual_fidelity = []
            all_node_estimated_fidelity = []
            all_node_purified_count = []
            all_node_purified_success_count = []
            all_satisfied_pairs_count = []
            all_experiment_duration = []
            all_teleport_success_count = []
            all_raw_fidelity_count = []
            all_raw_estimated_fidelity = []
            all_raw_teleport_success_count = []
            all_raw_purified_count = []
            all_raw_purified_success_count = []
            all_raw_satisfied_pairs_count = []
            # pandas.set_option('display.precision', 10)
            for column in dc.dataframe.columns:
                # Flatten the lists in the column
                # we have dictionary in the column
                # {'actual_fidelities': {3: 1.0, 9: 1.0, 4: 1.0,},
                # 'theoretical_fidelities': {3: 1.0, 9: 1.0, 4: 1.0,},
                # 'purified_count': 0,
                # 'purified_success_count': 0}

                flattened_actual_fidelities = []
                flattened_theoretical_fidelities = []
                flattened_purified_count = []
                flattened_purified_success_count = []
                flattened_experiment_duration = []
                flattened_satisfied_pairs_count = []
                flattened_teleport_success_count = []
                for result_data in dc.dataframe[column]:
                    if isinstance(result_data, dict):
                        for key, value in result_data.items():
                            if "actual_fidelities" in key:
                                flattened_actual_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                                all_raw_fidelity_count.append(Counter(value.values()))
                            elif "theoretical_fidelities" in key:
                                flattened_theoretical_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                                all_raw_estimated_fidelity.append(Counter(value.values()))
                            elif "purified_count" in key:
                                flattened_purified_count.append(value)
                                all_raw_purified_count.append(value)
                            elif "purified_success_count" in key:
                                flattened_purified_success_count.append(value)
                                all_raw_purified_success_count.append(value)
                            elif "experiment_duration" in key:
                                flattened_experiment_duration.append(value)
                            elif "satisfied_pairs_count" in key:
                                flattened_satisfied_pairs_count.append(value)
                                all_raw_satisfied_pairs_count.append(value)
                            elif "teleport_success_count" in key:
                                flattened_teleport_success_count.append(value)
                                all_raw_teleport_success_count.append(value)
                # calculate the average of the flattened values
                # actual fidelities
                actual_fidelities = np.mean(flattened_actual_fidelities, dtype=np.float64)
                all_node_actual_fidelity.append(actual_fidelities)
                # theoretical fidelities
                estimated_fidelities = np.mean(flattened_theoretical_fidelities, dtype=np.float64)
                all_node_estimated_fidelity.append(estimated_fidelities)
                # purified count
                purified_count = np.mean(flattened_purified_count, dtype=np.float64)
                all_node_purified_count.append(purified_count)

                # purified success count
                purified_success_count = np.mean(flattened_purified_success_count, dtype=np.float64)
                all_node_purified_success_count.append(purified_success_count)
                # experiment duration
                experiment_duration = np.mean(flattened_experiment_duration, dtype=np.float64)
                all_experiment_duration.append(experiment_duration)
                # satisfied pairs count
                satisfied_pairs_count = np.mean(flattened_satisfied_pairs_count, dtype=np.float64)
                all_satisfied_pairs_count.append(satisfied_pairs_count)
                # teleport success count
                teleport_success_count = np.mean(flattened_teleport_success_count, dtype=np.float64)
                all_teleport_success_count.append(teleport_success_count)
            filt_example.stop()
            del filt_example
            gc.collect()
            # check the time condition
            if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
                # case we have time overflow, reset the simulation and run again with different RNG
                ns.sim_reset()
                new_rng = np.random.RandomState()
                if new_rng == ns.get_random_state():
                    raise ValueError("Random state is not resetting")
                ns.set_random_state(rng=new_rng)

            final_fidelity = np.mean(all_node_actual_fidelity, dtype=np.float64)
            final_estimated_fidelity = np.mean(all_node_estimated_fidelity, dtype=np.float64)
            final_purified_count = np.mean(all_node_purified_count, dtype=np.float64)
            final_purified_success_count = np.mean(all_node_purified_success_count, dtype=np.float64)
            final_experiment_duration = np.mean(all_experiment_duration, dtype=np.float64)
            final_satisfied_pairs_count = np.mean(all_satisfied_pairs_count, dtype=np.float64)
            final_teleport_success_count = np.mean(all_teleport_success_count, dtype=np.float64)
            print(f"-*-" * 50)
            print(f"Skip noise: {skip_noise}")
            print(f"Entangle pairs: {entangle_pairs}")
            print(f"Final Satisfied pairs count: {final_satisfied_pairs_count}")
            print(f"Final experiment duration: {final_experiment_duration}")
            print(f"Final estimated fidelity: {final_estimated_fidelity}")
            print(f"Final fidelity: {final_fidelity}")
            print(f"Final purified count: {final_purified_count}")
            print(f"Final purified success count: {final_purified_success_count}")
            print(f"Final teleport success count: {final_teleport_success_count}")
            experiment_result[entangle_pairs] = {"actual_fidelity": final_fidelity,
                                                 "estimated_fidelity": final_estimated_fidelity,
                                                 "purified_count": final_purified_count,
                                                 "purified_success_count": final_purified_success_count,
                                                 "experiment_duration": final_experiment_duration / 1e9,
                                                 "satisfied_pairs_count": final_satisfied_pairs_count,
                                                 "teleport_success_count": final_teleport_success_count,
                                                 "raw_data": {
                                                     "actual_fidelity": all_raw_fidelity_count,
                                                     "estimated_fidelity": all_raw_estimated_fidelity,
                                                     "purified_count": Counter(all_raw_purified_count),
                                                     "purified_success_count": Counter(all_raw_purified_success_count),
                                                     "experiment_duration": all_experiment_duration,
                                                     "satisfied_pairs_count": Counter(all_raw_satisfied_pairs_count),
                                                     "teleport_success_count": Counter(all_raw_teleport_success_count)}
                                                 }
            with open(f"./purification_results/purification_results_2_nodes_{max_pairs}_paris_noise_{skip_noise}.json",
                      "w") as f:
                json.dump(experiment_result, f, indent=4)
            progress.update(task, advance=1)


def simulate_purification_increase_node(max_node=2):
    nodes_list = [f"Node_{i}" for i in range(max_node)]
    network = setup_network(nodes_list, "hop-by-hop-purification",
                            memory_capacity=1001, memory_depolar_rate=24583,
                            node_distance=3, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    if os.path.exists(f"./purification_results/purification_results_{max_node}_nodes_1_paris_noise.json"):
        with open(f"./purification_results/purification_results_{max_node}_nodes_1_paris_noise.json",
                  "r") as f:
            experiment_result = json.load(f)
    else:
        experiment_result = {}
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=max_node)
        for node_count in range(2, max_node + 1):
            if str(node_count) in experiment_result:
                print(f"Skipping node: {node_count}, loaded from file")
                progress.update(task, advance=1)
                continue
            filt_example, dc = example_sim_run(sample_nodes[:node_count], num_runs=1, memory_depolar_rate=24583,
                                               node_distance=3,
                                               max_entangle_pairs=1000,
                                               max_purify_pair=1,
                                               target_fidelity=0.93,
                                               skip_noise=True)
            filt_example.start()
            ns.sim_run()
            collected_data = dc.dataframe
            all_node_actual_fidelity = []
            all_node_estimated_fidelity = []
            all_node_purified_count = []
            all_node_purified_success_count = []
            all_satisfied_pairs_count = []
            all_experiment_duration = []
            all_teleport_success_count = []
            all_raw_fidelity_count = []
            all_raw_estimated_fidelity = []
            all_raw_teleport_success_count = []
            all_raw_purified_count = []
            all_raw_purified_success_count = []
            all_raw_satisfied_pairs_count = []
            # pandas.set_option('display.precision', 10)
            for column in dc.dataframe.columns:
                # Flatten the lists in the column
                # we have dictionary in the column
                # {'actual_fidelities': {3: 1.0, 9: 1.0, 4: 1.0,},
                # 'theoretical_fidelities': {3: 1.0, 9: 1.0, 4: 1.0,},
                # 'purified_count': 0,
                # 'purified_success_count': 0}

                flattened_actual_fidelities = []
                flattened_theoretical_fidelities = []
                flattened_purified_count = []
                flattened_purified_success_count = []
                flattened_experiment_duration = []
                flattened_satisfied_pairs_count = []
                flattened_teleport_success_count = []
                for result_data in dc.dataframe[column]:
                    if isinstance(result_data, dict):
                        for key, value in result_data.items():
                            if "actual_fidelities" in key:
                                flattened_actual_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                                all_raw_fidelity_count.append(Counter(value.values()))
                            elif "theoretical_fidelities" in key:
                                flattened_theoretical_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                                all_raw_estimated_fidelity.append(Counter(value.values()))
                            elif "purified_count" in key:
                                flattened_purified_count.append(value)
                                all_raw_purified_count.append(value)
                            elif "purified_success_count" in key:
                                flattened_purified_success_count.append(value)
                                all_raw_purified_success_count.append(value)
                            elif "experiment_duration" in key:
                                flattened_experiment_duration.append(value)
                            elif "satisfied_pairs_count" in key:
                                flattened_satisfied_pairs_count.append(value)
                                all_raw_satisfied_pairs_count.append(value)
                            elif "teleport_success_count" in key:
                                flattened_teleport_success_count.append(value)
                                all_raw_teleport_success_count.append(value)
                # calculate the average of the flattened values
                # actual fidelities
                actual_fidelities = np.mean(flattened_actual_fidelities, dtype=np.float64)
                all_node_actual_fidelity.append(actual_fidelities)
                # theoretical fidelities
                estimated_fidelities = np.mean(flattened_theoretical_fidelities, dtype=np.float64)
                all_node_estimated_fidelity.append(estimated_fidelities)
                # purified count
                purified_count = np.mean(flattened_purified_count, dtype=np.float64)
                all_node_purified_count.append(purified_count)

                # purified success count
                purified_success_count = np.mean(flattened_purified_success_count, dtype=np.float64)
                all_node_purified_success_count.append(purified_success_count)
                # experiment duration
                experiment_duration = np.mean(flattened_experiment_duration, dtype=np.float64)
                all_experiment_duration.append(experiment_duration)
                # satisfied pairs count
                satisfied_pairs_count = np.mean(flattened_satisfied_pairs_count, dtype=np.float64)
                all_satisfied_pairs_count.append(satisfied_pairs_count)
                # teleport success count
                teleport_success_count = np.mean(flattened_teleport_success_count, dtype=np.float64)
                all_teleport_success_count.append(teleport_success_count)
            filt_example.stop()
            del filt_example
            gc.collect()
            # check the time condition
            if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
                # case we have time overflow, reset the simulation and run again with different RNG
                ns.sim_reset()
                new_rng = np.random.RandomState()
                if new_rng == ns.get_random_state():
                    raise ValueError("Random state is not resetting")
                ns.set_random_state(rng=new_rng)

            final_fidelity = np.mean(all_node_actual_fidelity, dtype=np.float64)
            final_estimated_fidelity = np.mean(all_node_estimated_fidelity, dtype=np.float64)
            final_purified_count = np.mean(all_node_purified_count, dtype=np.float64)
            final_purified_success_count = np.mean(all_node_purified_success_count, dtype=np.float64)
            final_experiment_duration = np.mean(all_experiment_duration, dtype=np.float64)
            final_satisfied_pairs_count = np.mean(all_satisfied_pairs_count, dtype=np.float64)
            final_teleport_success_count = np.mean(all_teleport_success_count, dtype=np.float64)
            print(f"-*-" * 50)
            print(f"Skip noise: True")
            print(f"Node Count: {node_count}")
            print(f"Final Satisfied pairs count: {final_satisfied_pairs_count}")
            print(f"Final experiment duration: {final_experiment_duration}")
            print(f"Final estimated fidelity: {final_estimated_fidelity}")
            print(f"Final fidelity: {final_fidelity}")
            print(f"Final purified count: {final_purified_count}")
            print(f"Final purified success count: {final_purified_success_count}")
            print(f"Final teleport success count: {final_teleport_success_count}")
            experiment_result[node_count] = {"actual_fidelity": final_fidelity,
                                             "estimated_fidelity": final_estimated_fidelity,
                                             "purified_count": final_purified_count,
                                             "purified_success_count": final_purified_success_count,
                                             "experiment_duration": final_experiment_duration / 1e9,
                                             "satisfied_pairs_count": final_satisfied_pairs_count,
                                             "teleport_success_count": final_teleport_success_count,
                                             "raw_data": {
                                                 "actual_fidelity": all_raw_fidelity_count,
                                                 "estimated_fidelity": all_raw_estimated_fidelity,
                                                 "purified_count": Counter(all_raw_purified_count),
                                                 "purified_success_count": Counter(all_raw_purified_success_count),
                                                 "experiment_duration": all_experiment_duration,
                                                 "satisfied_pairs_count": Counter(all_raw_satisfied_pairs_count),
                                                 "teleport_success_count": Counter(all_raw_teleport_success_count)}
                                             }
            with open(f"./purification_results/purification_results_{max_node}_nodes_paris.json",
                      "w") as f:
                json.dump(experiment_result, f, indent=4)
            progress.update(task, advance=1)


def simulate_purification_increase_distance(max_dis=2.0, preload=False):
    target_fid_dic = {
        "500": [0.989, 0.99],
        "1000": [0.97, 0.98],
        "1500": [0.96, 0.97],
        "2000": [0.94, 0.95],
        "2500": [0.93, 0.94],
        "3000": [0.91, 0.92],
        "3500": [0.90, 0.91],
        "4000": [0.89, 0.90],
        "4500": [0.87, 0.88],
        "5000": [0.86, 0.87],
    }
    if preload:
        if os.path.exists(f"./purification_results/purification_results_2_nodes_1_paris_{max_dis}km.json"):
            with open(f"./purification_results/purification_results_2_nodes_1_paris_{max_dis}km.json",
                      "r") as f:
                experiment_result = json.load(f)
        else:
            experiment_result = {}
    else:
        experiment_result = {}
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=int(max_dis) * 4)
        dis_m = int(max_dis * 1000)
        for node_dis in range(500,  dis_m + 1, 500):
            if str(node_dis) in experiment_result:
                print(f"Skipping distance: {node_dis}, loaded from file")
                progress.update(task, advance=1)
                continue
            experiment_result[str(node_dis)] = {}
            for target_fid in target_fid_dic[str(node_dis)]:
                experiment_result[str(node_dis)][target_fid] = run_purify_sim(node_dis/1000, target_fid)

            with open(f"./purification_results/purification_results_2_nodes_1_paris_{max_dis}km.json",
                      "w") as f:
                json.dump(experiment_result, f, indent=4)
            progress.update(task, advance=1)


def run_purify_sim(distance, target_fid) -> dict:
    nodes_list = [f"Node_{i}" for i in range(2)]
    network = setup_network(nodes_list, "hop-by-hop-purification",
                            memory_capacity=11, memory_depolar_rate=24583,
                            node_distance=distance, source_delay=1)
    sample_nodes = [node for node in network.nodes.values()]
    filt_example, dc = example_sim_run(sample_nodes, num_runs=1, memory_depolar_rate=24583,
                                       node_distance=distance,
                                       max_entangle_pairs=11,
                                       max_purify_pair=9,
                                       target_fidelity=target_fid,
                                       skip_noise=True)
    filt_example.start()
    ns.sim_run()
    collected_data = dc.dataframe
    all_node_actual_fidelity = []
    all_node_estimated_fidelity = []
    all_node_purified_count = []
    all_node_purified_success_count = []
    all_satisfied_pairs_count = []
    all_experiment_duration = []
    all_teleport_success_count = []
    all_raw_fidelity_count = []
    all_raw_estimated_fidelity = []
    all_raw_teleport_success_count = []
    all_raw_purified_count = []
    all_raw_purified_success_count = []
    all_raw_satisfied_pairs_count = []
    # pandas.set_option('display.precision', 10)
    for column in dc.dataframe.columns:
        # Flatten the lists in the column
        # we have dictionary in the column
        # {'actual_fidelities': {3: 1.0, 9: 1.0, 4: 1.0,},
        # 'theoretical_fidelities': {3: 1.0, 9: 1.0, 4: 1.0,},
        # 'purified_count': 0,
        # 'purified_success_count': 0}

        flattened_actual_fidelities = []
        flattened_theoretical_fidelities = []
        flattened_purified_count = []
        flattened_purified_success_count = []
        flattened_experiment_duration = []
        flattened_satisfied_pairs_count = []
        flattened_teleport_success_count = []
        for result_data in dc.dataframe[column]:
            if isinstance(result_data, dict):
                for key, value in result_data.items():
                    if "actual_fidelities" in key:
                        flattened_actual_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                        all_raw_fidelity_count.append(Counter(value.values()))
                    elif "theoretical_fidelities" in key:
                        flattened_theoretical_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                        all_raw_estimated_fidelity.append(Counter(value.values()))
                    elif "purified_count" in key:
                        flattened_purified_count.append(value)
                        all_raw_purified_count.append(value)
                    elif "purified_success_count" in key:
                        flattened_purified_success_count.append(value)
                        all_raw_purified_success_count.append(value)
                    elif "experiment_duration" in key:
                        flattened_experiment_duration.append(value)
                    elif "satisfied_pairs_count" in key:
                        flattened_satisfied_pairs_count.append(value)
                        all_raw_satisfied_pairs_count.append(value)
                    elif "teleport_success_count" in key:
                        flattened_teleport_success_count.append(value)
                        all_raw_teleport_success_count.append(value)
        # calculate the average of the flattened values
        # actual fidelities
        actual_fidelities = np.mean(flattened_actual_fidelities, dtype=np.float64)
        all_node_actual_fidelity.append(actual_fidelities)
        # theoretical fidelities
        estimated_fidelities = np.mean(flattened_theoretical_fidelities, dtype=np.float64)
        all_node_estimated_fidelity.append(estimated_fidelities)
        # purified count
        purified_count = np.mean(flattened_purified_count, dtype=np.float64)
        all_node_purified_count.append(purified_count)

        # purified success count
        purified_success_count = np.mean(flattened_purified_success_count, dtype=np.float64)
        all_node_purified_success_count.append(purified_success_count)
        # experiment duration
        experiment_duration = np.mean(flattened_experiment_duration, dtype=np.float64)
        all_experiment_duration.append(experiment_duration)
        # satisfied pairs count
        satisfied_pairs_count = np.mean(flattened_satisfied_pairs_count, dtype=np.float64)
        all_satisfied_pairs_count.append(satisfied_pairs_count)
        # teleport success count
        teleport_success_count = np.mean(flattened_teleport_success_count, dtype=np.float64)
        all_teleport_success_count.append(teleport_success_count)
    filt_example.stop()
    del filt_example
    gc.collect()
    # check the time condition
    if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
        # case we have time overflow, reset the simulation and run again with different RNG
        ns.sim_reset()
        new_rng = np.random.RandomState()
        if new_rng == ns.get_random_state():
            raise ValueError("Random state is not resetting")
        ns.set_random_state(rng=new_rng)

    final_fidelity = np.mean(all_node_actual_fidelity, dtype=np.float64)
    final_estimated_fidelity = np.mean(all_node_estimated_fidelity, dtype=np.float64)
    final_purified_count = np.mean(all_node_purified_count, dtype=np.float64)
    final_purified_success_count = np.mean(all_node_purified_success_count, dtype=np.float64)
    final_experiment_duration = np.mean(all_experiment_duration, dtype=np.float64)
    final_satisfied_pairs_count = np.mean(all_satisfied_pairs_count, dtype=np.float64)
    final_teleport_success_count = np.mean(all_teleport_success_count, dtype=np.float64)
    print(f"-*-" * 50)
    print(f"Skip noise: True")
    print(f"Node Distance: {distance}")
    print(f"Target Fidelity: {target_fid}")
    print(f"Final Satisfied pairs count: {final_satisfied_pairs_count}")
    print(f"Final experiment duration: {final_experiment_duration}")
    print(f"Final estimated fidelity: {final_estimated_fidelity}")
    print(f"Final fidelity: {final_fidelity}")
    print(f"Final purified count: {final_purified_count}")
    print(f"Final purified success count: {final_purified_success_count}")
    print(f"Final teleport success count: {final_teleport_success_count}")
    run_res = {"actual_fidelity": final_fidelity,
               "estimated_fidelity": final_estimated_fidelity,
               "purified_count": final_purified_count,
               "purified_success_count": final_purified_success_count,
               "experiment_duration": final_experiment_duration,
               "satisfied_pairs_count": final_satisfied_pairs_count,
               "teleport_success_count": final_teleport_success_count,
               "raw_data": {
                   "actual_fidelity": all_raw_fidelity_count,
                   "estimated_fidelity": all_raw_estimated_fidelity,
                   "purified_count": Counter(all_raw_purified_count),
                   "purified_success_count": Counter(all_raw_purified_success_count),
                   "experiment_duration": all_experiment_duration,
                   "satisfied_pairs_count": Counter(all_raw_satisfied_pairs_count),
                   "teleport_success_count": Counter(all_raw_teleport_success_count)}
               }
    return run_res


if __name__ == "__main__":
    # experiment_with_increasing_node(3, "purification_results")
    # run_test(2)
    # if len(sys.argv) < 2:
    #     print("Please provide an argument to skip noise")
    #     exit(0)
    # if sys.argv[1] == "true":
    #     pop_noise = True
    # elif sys.argv[1] == "false":
    #     pop_noise = False
    # else:
    #     print("Invalid argument. Please use 'true' or 'false'")
    #     exit(0)
    # run_single_stack(2, pop_noise)
    # simulate_purification_increase_node(2)
    simulate_purification_increase_distance(0.5)
