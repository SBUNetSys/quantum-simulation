import json
import operator
import os.path
import sys
from functools import reduce

import numpy as np
import pandas
import pydynaa as pd
import matplotlib.pyplot as plt
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops, ketstates
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
import netsquid as ns
import netsquid.qubits.ketstates as ks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network_setup import setup_network
from entangle import GenEntanglement
from messagehandler import MessageHandler
from manage_entangle import ManageEntanglement


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_red(text):
    print(f"\033[91m{text}\033[0m")


class ExampleEntanglement(LocalProtocol):
    """
    Protocol to create entanglement between two nodes.
    """

    def __init__(self, network_nodes, num_runs=1, max_entangle_pairs=2, memory_depolar_rate=1, node_distance=20):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleEntanglement")

        # initialize the protocol for each node
        # Initialize the entangle protocol
        for index, node in enumerate(network_nodes):
            qubit_input_signals = []
            entangle_nodes = []
            if index - 1 >= 0:
                # case of we have a previous node
                self.add_subprotocol(GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                ))
                qubit_input_signals.append(self.subprotocols[f"entangle_{node.name}->{network_nodes[index - 1].name}"])
                entangle_nodes.append(network_nodes[index - 1].name)
            if index + 1 < len(network_nodes):
                # case of we have a next node
                self.add_subprotocol(GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                ))
                qubit_input_signals.append(self.subprotocols[f"entangle_{node.name}->{network_nodes[index + 1].name}"])
                entangle_nodes.append(network_nodes[index + 1].name)
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            # Initialize the swap protocol
            self.add_subprotocol(ManageEntanglement(node=node,
                                                    name=f"manage_entangle_{node.name}",
                                                    num_pairs=self.max_entangle_pairs,
                                                    qubit_input_signals=qubit_input_signals,
                                                    cc_message_handler=self.subprotocols[
                                                        f"message_handler_{node.name}"],
                                                    entangle_nodes=entangle_nodes,
                                                    memory_depolar_rate=memory_depolar_rate,
                                                    node_distance=node_distance
                                                    ))
            # Add re-entangle protocol
            for entangle_protocols in qubit_input_signals:
                entangle_protocols.re_entangle_sender = self.subprotocols[f"manage_entangle_{node.name}"]
                self.subprotocols[f"manage_entangle_{node.name}"].add_new_signal(entangle_protocols.name)

    def run(self):
        self.start_subprotocols()
        for _ in range(self.num_runs):
            start_time = sim_time()
            # set yield expression to wait for end of experiment
            await_signals = [self.await_signal(self.subprotocols[f"manage_entangle_{node.name}"], Signals.SUCCESS)
                             for node in self.all_nodes]
            yield reduce(operator.and_, await_signals)
            end_time = sim_time()
            print_green(f"Entanglement time: {end_time - start_time}")
            # get all the entangled qubits and calculate the fidelity
            results = [self.subprotocols[f"manage_entangle_{node.name}"].get_signal_result(Signals.SUCCESS, self)
                       for node in self.all_nodes]
            result_dic = {}
            for i in range(0, len(results) - 1):
                entangle_node = self.all_nodes[i + 1].name
                node = self.all_nodes[i].name
                node_res = results[i][entangle_node]
                entangle_res = results[i + 1][node]

                fidelity = []
                estimated_fidelity = []
                for index in node_res.keys():
                    qubit1, = self.nodes[node].subcomponents[f"{entangle_node}_qmemory"].peek(index)
                    qubit2, = self.nodes[entangle_node].subcomponents[f"{node}_qmemory"].peek(index)
                    estimated_fidelity.append(node_res[index])
                    f = qapi.fidelity([qubit1, qubit2], ks.b00)
                    if 0 < f < 0.99:
                        print_red(f"Fidelity is not expected: {f}")
                    fidelity.append(f)

                result_dic[f"{node}->{entangle_node}"] = fidelity
                print_green(f"Actual fidelity: {sum(fidelity) / len(fidelity)}")
                result_dic[f"{node}->{entangle_node} Estimated"] = estimated_fidelity
                print_green(f"Estimated fidelity: {sum(estimated_fidelity) / len(estimated_fidelity)}")
            self.send_signal(Signals.SUCCESS, {"results": result_dic})
            # reset the manage entangle protocol first
            # for node in self.all_nodes:
            #     self.subprotocols[f"manage_entangle_{node.name}"].reset()
            # # now reset entangle
            # for name, proctocol in self.subprotocols.items():
            #     if "entangle" in name:
            #         proctocol.reset()
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()
            # self.await_timer(1e9)

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


def example_sim_run(nodes, num_runs, memory_depolar_rate, node_distance, max_entangle_pairs):
    entangle_example = ExampleEntanglement(nodes, num_runs=num_runs,
                                           max_entangle_pairs=max_entangle_pairs,
                                           memory_depolar_rate=memory_depolar_rate,
                                           node_distance=node_distance)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=entangle_example, event_type=Signals.SUCCESS.value))
    return entangle_example, dc


def experiment_with_increasing_nodes(max_node, save_dir):
    # create a network
    nodes_list = [f"Node_{i}" for i in range(max_node)]
    network = setup_network(nodes_list, "hop-by-hop",
                            memory_capacity=2, memory_depolar_rate=100,
                            node_distance=20, source_delay=1e5)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    data = {}
    for i in range(2, max_node + 1):

        entangle_protocol, dc = example_sim_run(sample_nodes[:i], num_runs=1000,
                                                memory_depolar_rate=100,
                                                node_distance=20,
                                                max_entangle_pairs=2)
        entangle_protocol.start()
        # run the protocol
        ns.sim_run()

        # compute average for each column
        all_node_actual_fidelity = []
        all_node_estimated_fidelity = []
        # pandas.set_option('display.precision', 10)

        for column in dc.dataframe.columns:
            # Flatten the lists in the column
            flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
            if "Estimated" in column:
                all_node_estimated_fidelity.append(sum(flattened_values) / len(flattened_values))
            else:
                all_node_actual_fidelity.append(sum(flattened_values) / len(flattened_values))
        final_fidelity = 1
        for fidelity in all_node_actual_fidelity:
            final_fidelity *= fidelity
        final_estimated_fidelity = 1
        for fidelity in all_node_estimated_fidelity:
            final_estimated_fidelity *= fidelity
        print(f"Final estimated fidelity: {final_estimated_fidelity}")
        print(f"Final fidelity: {final_fidelity}")
        data[i] = {"actual_fidelity": final_fidelity, "estimated_fidelity": final_estimated_fidelity}

    with open(os.path.join(save_dir, f"entanglement_results_{max_node}_node.json"), "w") as f:
        json.dump(data, f)


def main():
    experiment_with_increasing_nodes(50, "./")


if __name__ == '__main__':
    main()
