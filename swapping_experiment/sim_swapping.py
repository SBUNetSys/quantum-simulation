import json
import os.path
import sys
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
from netsquid.nodes.network import Network
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate, INSTR_Z, INSTR_SWAP, INSTR_H
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from pydynaa import EventExpression
from netsquid.qubits.qubitapi import measure
from netsquid.qubits.operators import CNOT, Z
from netsquid.components.instructions import INSTR_MEASURE
from netsquid.nodes import Node
from netsquid.qubits.qubitapi import fidelity
import netsquid as ns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entangle import *
from swapping import *
from gen_swapping_tree import generate_swapping_tree
from network_setup_swapping import example_network_setup

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


class SwappingExample(LocalProtocol):
    """
    A simple example of a swapping protocol.
    """

    def __init__(self, nodes: list, num_runs=1, node_path=None, wait_time=0):
        if node_path is None:
            raise ValueError("node_path must be provided")
        # generate the swapping tree and levels
        swapping_nodes, levels = generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.levels = levels
        self.final_entanglement = (node_path[0], node_path[-1])
        self.all_nodes = nodes
        super().__init__(nodes={node.name: node for node in nodes}, name="SwappingExample")
        self.num_runs = num_runs
        # Initialize the entangle protocol
        for index, node in enumerate(nodes):
            qubit_input_signals = []
            if index - 1 >= 0:
                # case of we have a previous node
                self.add_subprotocol(GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=2,
                    entangle_node=nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{nodes[index - 1].name}",
                    is_source=False,
                ))
                qubit_input_signals.append(self.subprotocols[f"entangle_{node.name}->{nodes[index - 1].name}"])
            if index + 1 < len(nodes):
                # case of we have a next node
                self.add_subprotocol(GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=2,
                    entangle_node=nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{nodes[index + 1].name}",
                    is_source=True,
                ))
                qubit_input_signals.append(self.subprotocols[f"entangle_{node.name}->{nodes[index + 1].name}"])
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            # Initialize the swap protocol
            self.add_subprotocol(SwapProtocol(
                node=node,
                qubit_input_signals=qubit_input_signals,
                cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                swapping_tree=swapping_nodes,
                final_entanglement=self.final_entanglement,
                name=f"swap_{node.name}",
            ))
            # Add re-entangle protocol
            for entangle_protocols in qubit_input_signals:
                entangle_protocols.entanglement_handler = self.subprotocols[f"swap_{node.name}"]
                self.subprotocols[f"swap_{node.name}"].add_new_signal(entangle_protocols.name)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            yield (self.await_signal(self.subprotocols[f"swap_{self.final_entanglement[0]}"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols[f"swap_{self.final_entanglement[1]}"], Signals.SUCCESS))
            end_time = sim_time()

            print(f"Swapping completed in {(end_time - start_time) / 1e9} seconds.")
            result_a = self.subprotocols[f"swap_{self.final_entanglement[0]}"].get_signal_result(Signals.SUCCESS, self)
            result_b = self.subprotocols[f"swap_{self.final_entanglement[1]}"].get_signal_result(Signals.SUCCESS, self)
            print(f"Swapping result: {result_a}, {result_b}")
            # check the final entanglement's fidelity
            qubit_a = self.nodes[self.final_entanglement[0]].qmemory.peek(result_a[self.final_entanglement[1]])[0]
            qubit_b = self.nodes[self.final_entanglement[1]].qmemory.peek(result_b[self.final_entanglement[0]])[0]
            # rd = qapi.reduced_dm([qubit_a, qubit_b])
            # fidelity_result = qapi.fidelity(rd, ks.b00)
            fidelity_result = qapi.fidelity([qubit_a, qubit_b], ns.b00)
            print_red(f"Fidelity of the final entanglement: {fidelity_result}")

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
                yield subprotocol.reset()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


def example_sim_run(nodes, num_runs):
    swapping_example = SwappingExample(nodes=nodes, num_runs=num_runs, node_path=[node.name for node in nodes])

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run completed: {result}")
        return {"fidelity": result["fidelity"]}

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=swapping_example, event_type=Signals.SUCCESS.value))
    return swapping_example, dc


def experiment_with_increasing_node(max_nodes_count, save_dir="./"):
    node_list = [f"node_{i}" for i in range(max_nodes_count)]
    network = example_network_setup(nodes_list=node_list, node_distance=20, memory_depolar_rate=100)
    sample_nodes = [node for node in network.nodes.values()]

    fidelity_data = {}  # key: number of nodes, value: list of fidelities
    for i in range(3, max_nodes_count + 1):
        swapping_example, dc = example_sim_run(sample_nodes[:i], 1000)
        swapping_example.start()
        ns.sim_run()
        collected_data = dc.dataframe
        fidelities = collected_data["fidelity"]
        fidelity_data[i] = list(fidelities)
    with open(os.path.join(save_dir, f"fidelity_data_max_node_{max_nodes_count}.json"), "w") as f:
        json.dump(fidelity_data, f)
    return fidelity_data


def experiment_with_increase_memory_noise(max_depolar_rate, save_dir="./"):
    node_list = [f"node_{i}" for i in range(5)]

    fidelity_data = {}  # key: number of nodes, value: list of fidelities
    for i in range(10, max_depolar_rate + 1, 100):
        network = example_network_setup(nodes_list=node_list, node_distance=20, memory_depolar_rate=i)
        sample_nodes = [node for node in network.nodes.values()]
        swapping_example, dc = example_sim_run(sample_nodes, 1000)
        swapping_example.start()
        ns.sim_run()
        collected_data = dc.dataframe
        fidelities = collected_data["fidelity"]
        fidelity_data[i] = list(fidelities)
    with open(os.path.join(save_dir, f"fidelity_data_max_depolar_rate_{max_depolar_rate}.json"), "w") as f:
        json.dump(fidelity_data, f)
    return fidelity_data


def experiment_with_increase_node_distance(max_distance, save_dir="./"):
    node_list = [f"node_{i}" for i in range(5)]

    fidelity_data = {}  # key: number of nodes, value: list of fidelities
    for i in range(10, max_distance + 1, 10):
        network = example_network_setup(nodes_list=node_list, node_distance=i, memory_depolar_rate=100)
        sample_nodes = [node for node in network.nodes.values()]
        swapping_example, dc = example_sim_run(sample_nodes, 1000)
        swapping_example.start()
        ns.sim_run()
        collected_data = dc.dataframe
        fidelities = collected_data["fidelity"]
        fidelity_data[i] = list(fidelities)
    with open(os.path.join(save_dir, f"fidelity_data_max_distance_{max_distance}.json"), "w") as f:
        json.dump(fidelity_data, f)
    return fidelity_data


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

    with open("./fidelity_data_max_distance_1000.json", "r") as fin:
        data = json.load(fin)
    xs = list(int(x) for x in data.keys())
    ys = list(np.mean(data[key]) for key in data.keys())
    plot_line([xs], [ys], "Fidelity vs Node Distance (5 Node Swap)", "Node Distance (Km)", "Fidelity"
              , ["Fidelity"], save_dir="figures")
    with open("./fidelity_data_max_node_50.json", "r") as fin:
        data = json.load(fin)
    xs = list(int(x) for x in data.keys())
    ys = list(np.mean(data[key]) for key in data.keys())
    plot_line([xs], [ys], "Fidelity vs Number of Nodes (50 Node Swap)", "Number of Nodes", "Fidelity"
              , ["Fidelity"], save_dir="figures")
    with open("./fidelity_data_max_depolar_rate_100000.json", "r") as fin:
        data = json.load(fin)
    xs = list(int(x) for x in data.keys())
    ys = list(np.mean(data[key]) for key in data.keys())
    plot_line([xs], [ys], "Fidelity vs Memory Depolar Rate (5 Node Swap)", "Memory Depolar Rate", "Fidelity"
              , ["Fidelity"], save_dir="figures")
    plot_line([xs], [ys], "Fidelity vs Memory Depolar Rate (5 Node Swap) xlim=[0,3000]", "Memory Depolar Rate",
              "Fidelity"
              , ["Fidelity"], save_dir="figures", xlim=(0, 3000))
