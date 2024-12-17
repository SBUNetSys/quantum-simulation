import json
import os.path
import sys
import matplotlib.pyplot as plt

import pydynaa as pd
import netsquid as ns
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops, ketstates
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from wheel.cli.convert import egg2wheel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network
from utils import Logging, GenSwappingTree
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
from protocols.EndToEnd import EndToEndProtocol
import netsquid.qubits.operators as ops

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


class EndToEndExample(LocalProtocol):
    """
    A simple example of a swapping protocol.
    """

    def __init__(self, network_nodes: list,
                 num_runs=1,
                 node_path=None,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99):
        if node_path is None:
            raise ValueError("node_path must be provided")
        # generate the swapping tree and levels
        swapping_nodes, _, _ = GenSwappingTree.generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.final_entanglement = (node_path[0], node_path[-1])
        self.all_nodes = network_nodes
        self.max_entangle_pairs = max_entangle_pairs
        self.logger = Logging.Logger("EndToEnd", logging_enabled=True)
        null_logger = Logging.Logger("null", logging_enabled=False)

        super().__init__(nodes={node.name: node for node in network_nodes}, name="EndToEndExample")
        self.num_runs = num_runs
        # Initialize the entangle protocol
        for index, node in enumerate(network_nodes):
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            lower_protocols = []
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
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                # add purification
                pure_protocol = Purification(node=node,
                                             name=f"purify_{node.name}->{network_nodes[index - 1].name}",
                                             entangled_node=network_nodes[index - 1].name,
                                             entanglement_handler=eh_handler,
                                             cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                             max_entangled_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=self.logger
                                             )
                self.add_subprotocol(pure_protocol)
                lower_protocols.append(pure_protocol)

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
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                pure_protocol = Purification(node=node,
                                             name=f"purify_{node.name}->{network_nodes[index + 1].name}",
                                             entangled_node=network_nodes[index + 1].name,
                                             entanglement_handler=eh_handler,
                                             cc_message_handler=self.subprotocols[
                                                 f"message_handler_{node.name}"],
                                             max_entangled_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=self.logger
                                             )
                # Initialize the purification protocol
                self.add_subprotocol(pure_protocol)
                lower_protocols.append(pure_protocol)
            # Add end to end protocol to each node
            end_to_end = EndToEndProtocol(node=node,
                                          name=f"e2e_{node.name}",
                                          swapping_nodes=self.swap_nodes,
                                          final_entanglement=self.final_entanglement,
                                          cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                          qubit_ready_protocols=lower_protocols,
                                          max_pairs=self.max_entangle_pairs - 1,
                                          logger=self.logger,
                                          is_top_layer=True)
            self.add_subprotocol(end_to_end)


    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            yield (self.await_signal(self.subprotocols[f"e2e_{self.final_entanglement[0]}"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols[f"e2e_{self.final_entanglement[1]}"], Signals.SUCCESS))
            end_time = sim_time()

            print(f"Swapping completed in {(end_time - start_time) / 1e9} seconds.")
            result_a = self.subprotocols[f"e2e_{self.final_entanglement[0]}"].get_signal_result(Signals.SUCCESS, self)
            result_b = self.subprotocols[f"e2e_{self.final_entanglement[1]}"].get_signal_result(Signals.SUCCESS, self)
            print(f"Swapping result: {result_a}, {result_b}")
            mem_pos_a = list(result_a[self.final_entanglement[1]].keys())
            mem_pos_b = list(result_b[self.final_entanglement[0]].keys())
            # check the final entanglement's fidelity
            qubit_a = self.nodes[self.final_entanglement[0]].qmemory.peek(mem_pos_a[0])[0]
            qubit_b = self.nodes[self.final_entanglement[1]].qmemory.peek(mem_pos_b[0])[0]
            # rd = qapi.reduced_dm([qubit_a, qubit_b])
            # fidelity_result = qapi.fidelity(rd, ks.b00)
            fidelity_result = qapi.fidelity([qubit_a, qubit_b], ns.b00)
            print(f"Fidelity of the final entanglement: {fidelity_result}")

            # generate a qubit for teleportation
            # qubit = qapi.create_qubits(1)[0]
            #
            # qapi.operate(qubit, ops.H)
            # qapi.operate(qubit, ops.S)
            # og_state = qubit.qstate
            # og_bstate = qubit_b.qstate
            # # teleport the qubit
            # qapi.operate(qubits=[qubit, qubit_a], operator=ops.CNOT)
            # qapi.operate(qubit, ops.H)
            # m1, _ = qapi.measure(qubit)
            # m2, _ = qapi.measure(qubit_a)
            # if m1 == 1:
            #     qapi.operate(qubit_b, ops.Z)
            # if m2 == 1:
            #     qapi.operate(qubit_b, ops.X)
            # # check if the teleportation was successful
            # new_state = qubit_b.qstate

            # if og_state == new_state:
            #     print_green("Teleportation successful")

            # rd = qapi.reduced_dm([qubit_a, qubit_b])
            # fidelity_result = qapi.fidelity(rd, ks.b00)
            # fidelity_result = qapi.fidelity([qubit_a, qubit_b], ks.b00)
            # print_red(f"Fidelity of the final entanglement after {10e9 / 1e9} seconds: {fidelity_result}")
            self.send_signal(Signals.SUCCESS, {"fidelity": fidelity_result})
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


def example_sim_run(nodes, num_runs, memory_depolar_rate,
                    node_distance, max_entangle_pairs, target_fidelity):
    e2e_example = EndToEndExample(network_nodes=nodes,
                                       num_runs=num_runs,
                                       node_path=[node.name for node in nodes],
                                       max_entangle_pairs=max_entangle_pairs,
                                       memory_depolar_rate=memory_depolar_rate,
                                       node_distance=node_distance,
                                       target_fidelity=target_fidelity)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run completed: {result}")
        return result

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=e2e_example, event_type=Signals.SUCCESS.value))
    return e2e_example, dc


def plot_scatter_data(fidelity_data):
    for key, value in fidelity_data.items():
        plt.scatter(key, sum(value) / len(value))

    plt.show()


def plot_line(xs, ys, title, x_label, y_label, data_legends, xlim=None, save=True, save_dir="./", num_bins=20):
    fig, ax = plt.subplots(figsize=(12, 6))
    for x, y, legend in zip(xs, ys, data_legends):
        ax.plot(x, y, label=f'{legend}')
    ax.set_title(f'{title}')
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend(loc="best")  # loc="upper left"bbox_to_anchor=(1.05, 1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # rotate x labels
    plt.xticks(rotation=45)
    ax.set_xlim(left=0)
    ax.set_ylim(top=1)
    if xlim:
        ax.set_xlim(xlim)
    # Group x labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Format the labels as integers
    plt.tight_layout()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.show()

def run_e2e_test(distances=3):
    nodes_list = [f"Node_{i}" for i in range(4)]
    network = setup_network(nodes_list, "end-to-end-network",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=distances, source_delay=1)
    sample_nodes = [node for node in network.nodes.values()]
    end_to_end_example, dc = example_sim_run(sample_nodes,
                                             1,
                                             100,
                                             distances,
                                             2,
                                             0.995)
    end_to_end_example.start()
    ns.sim_run()
    collected_data = dc.dataframe
    print(collected_data)

if __name__ == '__main__':
    # node_list = ["node_A", "node_B", "node_C", "node_D", "node_E", "node_F"]
    # network = example_network_setup(nodes_list=node_list, node_distance=20, memory_depolar_rate=100)
    # sample_nodes = [node for node in network.nodes.values()]
    # swapping_example, dc = example_sim_run(sample_nodes, 1000)
    # swapping_example.start()
    # ns.sim_run()
    # collected_data = dc.dataframe
    # # print average fidelity
    # fidelities = collected_data["fidelity"]
    # print(f"Average fidelity: {sum(fidelities) / len(fidelities)}")
    # save_dir = "./swapping_experiment"
    # data = experiment_with_increasing_node(50)
    # data = experiment_with_increase_memory_noise(100000)
    # data = experiment_with_increase_node_distance(1000)
    run_e2e_test(3)
