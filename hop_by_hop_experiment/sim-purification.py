import json
import operator
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

from utils.NetworkSetup import setup_network
from utils import Logging
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import PurifyEntangle


class PurificationExample(LocalProtocol):
    """
    Protocol for a complete purification example.

    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExamplePurification")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        # initialize the protocol for each node
        # Initialize the entangle protocol
        for index, node in enumerate(network_nodes):
            qubit_input_signals = []
            # dictionary to store the entangle node and the corresponding protocol name
            entangle_nodes = {}
            if index - 1 >= 0:
                # case of we have a previous node
                self.add_subprotocol(GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=self.logger
                ))
                qubit_input_signals.append(self.subprotocols[f"entangle_{node.name}->{network_nodes[index - 1].name}"])
                entangle_nodes[network_nodes[index - 1].name] = f"entangle_{node.name}->{network_nodes[index - 1].name}"
            if index + 1 < len(network_nodes):
                # case of we have a next node
                self.add_subprotocol(GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=self.logger
                ))
                qubit_input_signals.append(self.subprotocols[f"entangle_{node.name}->{network_nodes[index + 1].name}"])
                entangle_nodes[network_nodes[index + 1].name] = f"entangle_{node.name}->{network_nodes[index + 1].name}"
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            # Initialize the entanglement handler protocol
            self.add_subprotocol(EntanglementHandler(node=node,
                                                     name=f"entanglement_handler_{node.name}",
                                                     num_pairs=self.max_entangle_pairs,
                                                     qubit_input_signals=qubit_input_signals,
                                                     cc_message_handler=self.subprotocols[
                                                         f"message_handler_{node.name}"],
                                                     entangle_nodes=entangle_nodes,
                                                     memory_depolar_rate=memory_depolar_rate,
                                                     node_distance=node_distance,
                                                     is_top_layer=False,
                                                     logger=self.logger
                                                     ))
            # Initialize the purification protocol
            self.add_subprotocol(PurifyEntangle(node=node,
                                                name=f"purify_{node.name}",
                                                entangled_nodes=list(entangle_nodes.keys()),
                                                entanglement_handler=self.subprotocols[
                                                    f"entanglement_handler_{node.name}"],
                                                cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                                max_entangled_pair=self.max_entangle_pairs,
                                                target_fidelity=target_fidelity,
                                                is_top_layer=True,
                                                logger=self.logger
                                                ))
            # Add re-entangle protocol
            for entangle_protocols in qubit_input_signals:
                entangle_protocols.entanglement_handler = self.subprotocols[f"entanglement_handler_{node.name}"]
                # no need to add new signal as the entanglement handler protocol will handle during initialization
                # self.subprotocols[f"entanglement_handler_{node.name}"].add_new_signal(entangle_protocols.name)

    def run(self):
        self.start_subprotocols()
        for index in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            wait_signals = [self.await_signal(self.subprotocols[f"purify_{node.name}"], MessageType.PROTOCOL_FINISHED)
                            for node in self.all_nodes]

            yield reduce(operator.and_, wait_signals)

            results = [self.subprotocols[f"purify_{node.name}"].get_signal_result(MessageType.PROTOCOL_FINISHED)
                       for node in self.all_nodes]
            result_dic = {}
            """
            {"satisfied_pairs": self.satisfied_pairs,
            "purification_count": self.purification_count,
            "purification_success_count": self.purification_success_count,
             "finish_time": sim_time()}
            """
            for i in range(0, len(results) - 1):
                entangle_node = self.all_nodes[i + 1].name
                node = self.all_nodes[i].name
                node_pair_res = results[i]["satisfied_pairs"][entangle_node]
                purified_count = results[i]["purification_count"][entangle_node]
                purified_success_count = results[i]["purification_success_count"][entangle_node]
                finish_time = results[i]["finish_time"]
                entangle_pair_res = results[i + 1]["satisfied_pairs"][node]
                print(f"Finish time: {finish_time}")
                # measure the actual fidelity
                actual_fidelities = {}
                theoretical_fidelities = {}
                for mem_pos, theoretical_fidelity in node_pair_res.items():
                    q_a = self.all_nodes[i].subcomponents[f"{entangle_node}_qmemory"].peek(mem_pos)[0]
                    q_b = self.all_nodes[i + 1].subcomponents[f"{node}_qmemory"].peek(mem_pos)[0]
                    q_a_name = str(q_a.name).split("#")[-1].split("-")[0]
                    q_b_name = str(q_b.name).split("#")[-1].split("-")[0]
                    # print(f"Qubit names: {q_a_name}, {q_b_name}")
                    if q_a_name != q_b_name:
                        raise ValueError(f"Qubit names are not the same at {mem_pos}: {q_a_name}, {q_b_name}")
                    # if q_a.qstate != q_b.qstate:
                    #     raise ValueError(f"Qubit states are not the same: {q_a.qstate}, {q_b.qstate}")
                    f = qapi.fidelity([q_a, q_b], ks.b00)
                    if 0 < f < 0.99:
                        raise ValueError(f"Fidelity is not correct: {f}")
                    # print(f"Actual fidelity at {mem_pos} is {f}, {q_a.qstate}, {q_b.qstate}")
                    actual_fidelities[mem_pos] = f
                    theoretical_fidelities[mem_pos] = theoretical_fidelity
                result_dic[f"{node}->{entangle_node}"] = {
                    "actual_fidelities": actual_fidelities,
                    "theoretical_fidelities": theoretical_fidelities,
                    "purified_count": purified_count,
                    "purified_success_count": purified_success_count,
                    "experiment_duration": finish_time - start_time,
                    "satisfied_pairs_count": len(node_pair_res)}

            print(result_dic)
            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            # TODO: This is to gracefully reset the protocol.
            for node in self.all_nodes:
                eh_protocol = self.subprotocols[f"entanglement_handler_{node.name}"]
                if eh_protocol.is_running:
                    self.await_signal(eh_protocol, MessageType.PROTOCOL_FINISHED)
            # wait_signals = [self.await_signal(self.subprotocols[f"entanglement_handler_{node.name}"],
            #                                   MessageType.PROTOCOL_FINISHED)
            #                 for node in self.all_nodes]

            # yield reduce(operator.and_, wait_signals)
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()
            # self.reset()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


def example_sim_run(nodes, num_runs, memory_depolar_rate, node_distance, max_entangle_pairs, target_fidelity):
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
                                         target_fidelity=target_fidelity)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Purification Run {result['run_index']} completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=purify_example,
                                     event_type=Signals.SUCCESS.value))
    return purify_example, dc


def run_test(max_node):
    # create a network
    nodes_list = [f"Node_{i}" for i in range(max_node)]
    network = setup_network(nodes_list, "hop-by-hop-purification",
                            memory_capacity=512, memory_depolar_rate=100,
                            node_distance=20, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    experiment_result = {}
    max_pairs = 257
    for max_entangle_pair in range(5, max_pairs + 1, 2):
        # process the collected data
        # compute average for each column
        all_node_actual_fidelity = []
        all_node_estimated_fidelity = []
        all_node_purified_count = []
        all_node_purified_success_count = []
        all_satisfied_pairs_count = []
        all_experiment_duration = []
        for _ in range(1000):
            filt_example, dc = example_sim_run(sample_nodes[:max_node], num_runs=1, memory_depolar_rate=100,
                                               node_distance=20,
                                               max_entangle_pairs=max_entangle_pair, target_fidelity=0.995)
            filt_example.start()
            ns.sim_run()
            collected_data = dc.dataframe
            print(collected_data)

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
                for result_data in dc.dataframe[column]:
                    if isinstance(result_data, dict):
                        for key, value in result_data.items():
                            if "actual_fidelities" in key:
                                flattened_actual_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                            elif "theoretical_fidelities" in key:
                                flattened_theoretical_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                            elif "purified_count" in key:
                                flattened_purified_count.append(value)
                            elif "purified_success_count" in key:
                                flattened_purified_success_count.append(value)
                            elif "experiment_duration" in key:
                                flattened_experiment_duration.append(value)
                            elif "satisfied_pairs_count" in key:
                                flattened_satisfied_pairs_count.append(value)

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

            filt_example.stop()

        # calculate the final fidelity
        # final_fidelity = 1
        # for fidelity in all_node_actual_fidelity:
        #     final_fidelity *= fidelity
        # final_estimated_fidelity = 1
        # for fidelity in all_node_estimated_fidelity:
        #     final_estimated_fidelity *= fidelity
        final_fidelity = np.mean(all_node_actual_fidelity, dtype=np.float64)
        final_estimated_fidelity = np.mean(all_node_estimated_fidelity, dtype=np.float64)
        final_purified_count = np.mean(all_node_purified_count, dtype=np.float64)
        final_purified_success_count = np.mean(all_node_purified_success_count, dtype=np.float64)
        final_experiment_duration = np.mean(all_experiment_duration, dtype=np.float64)
        final_satisfied_pairs_count = np.mean(all_satisfied_pairs_count, dtype=np.float64)
        print(f"-*-"*10)
        print(f"Max entangle pair: {max_entangle_pair}")
        print(f"Final Satisfied pairs count: {final_satisfied_pairs_count}")
        print(f"Final experiment duration: {final_experiment_duration}")
        print(f"Final estimated fidelity: {final_estimated_fidelity}")
        print(f"Final fidelity: {final_fidelity}")
        print(f"Final purified count: {final_purified_count}")
        print(f"Final purified success count: {final_purified_success_count}")
        experiment_result[max_entangle_pair] = {"actual_fidelity": final_fidelity,
                                                "estimated_fidelity": final_estimated_fidelity,
                                                "purified_count": final_purified_count,
                                                "purified_success_count": final_purified_success_count,
                                                "experiment_duration": final_experiment_duration/1e9,
                                                "satisfied_pairs_count": final_satisfied_pairs_count}
        with open(f"./purification_results/purification_result_{max_node}_node_{max_pairs}_pairs.json", "w") as f:
            json.dump(experiment_result, f)


def experiment_with_increasing_node(max_node, save_dir):
    """
    Run the purification protocol with increasing number of nodes.

    Parameters
    ----------
    max_node : int
        Maximum number of nodes to run the experiment.
    save_dir : str
        Directory to save the results.

    """
    # create a network
    nodes_list = [f"Node_{i}" for i in range(max_node)]
    network = setup_network(nodes_list, "hop-by-hop-purification",
                            memory_capacity=10, memory_depolar_rate=100,
                            node_distance=20, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    experiment_result = {}
    # run the protocol
    for i in range(2, max_node + 1):

        # ns.sim_run()
        round_data = {}
        for _ in range(10):
            filt_example, dc = example_sim_run(sample_nodes[:i], num_runs=1, memory_depolar_rate=100, node_distance=20,
                                               max_entangle_pairs=10, target_fidelity=0.995)
            filt_example.start()
            ns.sim_run()
            collected_data = dc.dataframe
            print(collected_data)
            # process the collected data
            # compute average for each column
            all_node_actual_fidelity = []
            all_node_estimated_fidelity = []
            all_node_purified_count = []
            all_node_purified_success_count = []
            all_experiment_duration = []
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

                for result_data in dc.dataframe[column]:
                    if isinstance(result_data, dict):
                        for key, value in result_data.items():
                            if "actual_fidelities" in key:
                                flattened_actual_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                            elif "theoretical_fidelities" in key:
                                flattened_theoretical_fidelities.append(np.mean(list(value.values()), dtype=np.float64))
                            elif "purified_count" in key:
                                flattened_purified_count.append(value)
                            elif "purified_success_count" in key:
                                flattened_purified_success_count.append(value)
                            elif "experiment_duration" in key:
                                flattened_experiment_duration.append(value)
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
            # calculate the final fidelity
            final_fidelity = 1
            for fidelity in all_node_actual_fidelity:
                final_fidelity *= fidelity
            final_estimated_fidelity = 1
            for fidelity in all_node_estimated_fidelity:
                final_estimated_fidelity *= fidelity
            final_purified_count = np.mean(all_node_purified_count, dtype=np.float64)
            final_purified_success_count = np.mean(all_node_purified_success_count, dtype=np.float64)
            final_experiment_duration = np.mean(all_experiment_duration, dtype=np.float64)
            print(f"Final experiment duration: {final_experiment_duration}")
            print(f"Final estimated fidelity: {final_estimated_fidelity}")
            print(f"Final fidelity: {final_fidelity}")
            print(f"Final purified count: {final_purified_count}")
            print(f"Final purified success count: {final_purified_success_count}")
            experiment_result[i] = {"actual_fidelity": final_fidelity,
                                    "estimated_fidelity": final_estimated_fidelity,
                                    "purified_count": final_purified_count,
                                    "purified_success_count": final_purified_success_count,
                                    "experiment_duration": final_experiment_duration}
            filt_example.stop()
    with open(f"{save_dir}/purification_result_{max_node}_node.json", "w") as f:
        json.dump(experiment_result, f)
    return experiment_result


if __name__ == "__main__":
    # experiment_with_increasing_node(3, "purification_results")
    run_test(2)
