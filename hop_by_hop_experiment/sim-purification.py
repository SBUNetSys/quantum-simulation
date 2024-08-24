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

from hop_by_hop_experiment.sim_entanglement import print_red
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
                prue_protocol = PurifyEntangle(node=node,
                                               name=f"purify_{node.name}->{network_nodes[index - 1].name}",
                                               entangled_node=network_nodes[index-1].name,
                                               entanglement_handler=eh_handler,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               max_entangled_pair=self.max_entangle_pairs,
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
                self.add_subprotocol(PurifyEntangle(node=node,
                                                    name=f"purify_{node.name}->{network_nodes[index + 1].name}",
                                                    entangled_node=network_nodes[index + 1].name,
                                                    entanglement_handler=eh_handler,
                                                    cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                                    max_entangled_pair=self.max_entangle_pairs,
                                                    target_fidelity=target_fidelity,
                                                    is_top_layer=True,
                                                    logger=self.logger
                                                    ))

    def run(self):
        self.start_subprotocols()
        for subprotoco, val in self.subprotocols.items():
            print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            pure_protocols = []
            for subprotocol in self.subprotocols.values():
                if "purify" in subprotocol.name:
                    pure_protocols.append(subprotocol)

            wait_signals = [self.await_signal(p, MessageType.PROTOCOL_FINISHED)
                            for p in pure_protocols]

            yield reduce(operator.and_, wait_signals)

            results = [p.get_signal_result(MessageType.PROTOCOL_FINISHED)
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
                        # raise ValueError(f"Fidelity is not correct: {f}, \n\t{q_a.qstate}\n\t{q_b.qstate}")
                        print_red(f"Fidelity is not correct: {f}, \n\t{q_a.qstate}\n\t{q_b.qstate}")
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
                node_index += 1

            print(result_dic)
            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            # TODO: This is to gracefully reset the protocol.
            for subprotocol in self.subprotocols.values():
                if "entanglement_handler" in subprotocol.name and subprotocol.is_running:
                    self.await_signal(subprotocol, MessageType.PROTOCOL_FINISHED)
            # for node in self.all_nodes:
            #     eh_protocol = self.subprotocols[f"entanglement_handler_{node.name}"]
            #     if eh_protocol.is_running:
            #         self.await_signal(eh_protocol, MessageType.PROTOCOL_FINISHED)
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




def run_single_stack(nodes_count):
    # create a network
    nodes_list = [f"Node_{i}" for i in range(nodes_count)]
    network = setup_network(nodes_list, "hop-by-hop-purification",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=20, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    experiment_result = {}
    filt_example, dc = example_sim_run(sample_nodes, num_runs=10, memory_depolar_rate=100,
                                       node_distance=20,
                                       max_entangle_pairs=128, target_fidelity=0.995)
    filt_example.start()
    ns.sim_run()
    collected_data = dc.dataframe
    print(collected_data)
    all_node_actual_fidelity = []
    all_node_estimated_fidelity = []
    all_node_purified_count = []
    all_node_purified_success_count = []
    all_satisfied_pairs_count = []
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

    final_fidelity = np.mean(all_node_actual_fidelity, dtype=np.float64)
    final_estimated_fidelity = np.mean(all_node_estimated_fidelity, dtype=np.float64)
    final_purified_count = np.mean(all_node_purified_count, dtype=np.float64)
    final_purified_success_count = np.mean(all_node_purified_success_count, dtype=np.float64)
    final_experiment_duration = np.mean(all_experiment_duration, dtype=np.float64)
    final_satisfied_pairs_count = np.mean(all_satisfied_pairs_count, dtype=np.float64)
    print(f"-*-" * 10)
    print(f"Final Satisfied pairs count: {final_satisfied_pairs_count}")
    print(f"Final experiment duration: {final_experiment_duration}")
    print(f"Final estimated fidelity: {final_estimated_fidelity}")
    print(f"Final fidelity: {final_fidelity}")
    print(f"Final purified count: {final_purified_count}")
    print(f"Final purified success count: {final_purified_success_count}")
    # experiment_result[max_entangle_pair] = {"actual_fidelity": final_fidelity,
    #                                         "estimated_fidelity": final_estimated_fidelity,
    #                                         "purified_count": final_purified_count,
    #                                         "purified_success_count": final_purified_success_count,
    #                                         "experiment_duration": final_experiment_duration / 1e9,
    #                                         "satisfied_pairs_count": final_satisfied_pairs_count}
if __name__ == "__main__":
    # experiment_with_increasing_node(3, "purification_results")
    # run_test(2)
    run_single_stack(2)