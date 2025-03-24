import json
import operator
import os.path
import sys
from functools import reduce

import numpy as np
import pydynaa as pd
import matplotlib.pyplot as plt

import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector

from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
import netsquid.qubits.operators as ops

import netsquid.qubits.ketstates as ks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network, setup_network_parallel
from utils import Logging
from protocols.GenEntanglementConcurrent import GenEntanglementConcurrent
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14


class ExampleEntanglement(LocalProtocol):
    """
    Protocol to create entanglement between two nodes.
    """

    def __init__(self, network_nodes, num_runs=1, max_entangle_pairs=2, memory_depolar_rate=1,
                 node_distance=20,
                 skip_noise=False):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least 2 nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.skip_noise = skip_noise
        self.logger = Logging.Logger("ExampleEntanglement", logging_enabled=False)
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleEntanglement")

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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=self.logger
                )
                self.add_subprotocol(gen_protocol)
                eh_handler = EntanglementHandlerConcurrent(node=node,
                                                 name=f"entanglement_handler_{node.name}->{network_nodes[index - 1].name}",
                                                 num_pairs=self.max_entangle_pairs,
                                                 qubit_input_protocol=gen_protocol,
                                                 cc_message_handler=self.subprotocols[
                                                     f"message_handler_{node.name}"],
                                                 entangle_node=network_nodes[index - 1].name,
                                                 memory_depolar_rate=memory_depolar_rate,
                                                 node_distance=node_distance,
                                                 is_top_layer=True,
                                                 logger=self.logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler

            if index + 1 < len(network_nodes):
                # case of we have a next node
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=self.logger
                )
                self.add_subprotocol(gen_protocol)
                eh_handler = EntanglementHandlerConcurrent(node=node,
                                                 name=f"entanglement_handler_{node.name}->{network_nodes[index + 1].name}",
                                                 num_pairs=self.max_entangle_pairs,
                                                 qubit_input_protocol=gen_protocol,
                                                 cc_message_handler=self.subprotocols[
                                                     f"message_handler_{node.name}"],
                                                 entangle_node=network_nodes[index + 1].name,
                                                 memory_depolar_rate=memory_depolar_rate,
                                                 node_distance=node_distance,
                                                 is_top_layer=True,
                                                 logger=self.logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler

    def run(self):
        self.start_subprotocols()
        for _ in range(self.num_runs):
            start_time = sim_time()
            # set yield expression to wait for end of experiment
            eh_protocols = []
            for p in self.subprotocols.values():
                if "entanglement_handler" in p.name:
                    eh_protocols.append(p)

            await_signals = [self.await_signal(p, MessageType.ENTANGLEMENT_HANDLER_FINISHED)
                             for p in eh_protocols]
            yield reduce(operator.and_, await_signals)
            end_time = sim_time()
            # print_green(f"Entanglement time: {end_time - start_time}")
            # get all the entangled qubits and calculate the fidelity
            results = [p.get_signal_result(MessageType.ENTANGLEMENT_HANDLER_FINISHED, self)
                       for p in eh_protocols]
            result_dic = {}
            node_index = 0
            for i in range(0, len(results), 2):
                entangle_node = self.all_nodes[node_index + 1].name
                node = self.all_nodes[node_index].name
                node_res = results[i]
                entangle_res = results[i + 1]
                node_index += 1
                fidelity = []
                estimated_fidelity = []
                teleport_success_count = 0
                for index in node_res.keys():
                    qubit1, = self.nodes[node].subcomponents[f"{entangle_node}_qmemory"].pop(index,
                                                                                             skip_noise=self.skip_noise)
                    qubit2, = self.nodes[entangle_node].subcomponents[f"{node}_qmemory"].pop(index,
                                                                                             skip_noise=self.skip_noise)
                    estimated_fidelity.append(node_res[index])
                    f = qapi.fidelity([qubit1, qubit2], ks.b00)
                    # sanity check
                    q_a_name = str(qubit1.name).split("#")[-1].split("-")[0]
                    q_b_name = str(qubit2.name).split("#")[-1].split("-")[0]
                    # print(f"Qubit names: {q_a_name}, {q_b_name}")
                    if q_a_name != q_b_name:
                        raise ValueError(f"Qubit names are not the same at {index}: {q_a_name}, {q_b_name}")
                    if 0 < f < 0.99:
                        raise ValueError(f"Fidelity is not expected: {f}")
                    fidelity.append(f)
                    # start_teleportation, generate a qubit for teleportation
                    # rotate the qubit to y0 state
                    qubit = qapi.create_qubits(1)[0]
                    qapi.operate(qubit, ops.H)
                    qapi.operate(qubit, ops.S)
                    # teleport the qubit
                    fid = self.test_teleportation(qubit1, qubit2, qubit)

                    # print(f"Teleportation fidelity: {fid}")
                    if fid > 0.99:
                        teleport_success_count += 1

                result_dic[f"{node}->{entangle_node}"] = fidelity
                # print_green(f"Actual fidelity: {sum(fidelity) / len(fidelity)}")
                result_dic[f"{node}->{entangle_node} Estimated"] = estimated_fidelity
                result_dic[f"{node}->{entangle_node} Duration"] = end_time - start_time
                result_dic[f"{node}->{entangle_node} Teleportation Success"] = teleport_success_count
                # print_green(f"Estimated fidelity: {sum(estimated_fidelity) / len(estimated_fidelity)}")
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


def example_sim_run(nodes, num_runs, memory_depolar_rate, node_distance, max_entangle_pairs, skip_noise=False):
    entangle_example = ExampleEntanglement(nodes, num_runs=num_runs,
                                           max_entangle_pairs=max_entangle_pairs,
                                           memory_depolar_rate=memory_depolar_rate,
                                           node_distance=node_distance,
                                           skip_noise=skip_noise)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # print(f"Run completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=entangle_example, event_type=Signals.SUCCESS.value))
    return entangle_example, dc


def plot_lines(xs, ys, title, x_label, y_label, data_legends, xlim=None, save=True, save_dir="./", num_bins=20):
    fig, ax = plt.subplots(figsize=(12, 6))
    for x, y, legend in zip(xs, ys, data_legends):
        ax.plot(x, y, label=f'{legend}', lw=2)
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


def experiment_with_increasing_nodes(max_node, save_dir):
    # create a network
    nodes_list = [f"Node_{i}" for i in range(max_node)]
    network = setup_network(nodes_list, "hop-by-hop-entangle",
                            memory_capacity=128, memory_depolar_rate=24583,
                            node_distance=1, source_delay=1e5)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    data = {}
    from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Nodes...", total=max_node)
        for node_count in range(2, max_node + 1):
            entangle_protocol, dc = example_sim_run(sample_nodes[:node_count], num_runs=1000,
                                                    memory_depolar_rate=24583,
                                                    node_distance=1,
                                                    max_entangle_pairs=100,
                                                    skip_noise=True
                                                    )
            entangle_protocol.start()
            # run the protocol
            ns.sim_run()

            # compute average for each column
            all_node_actual_fidelity = []
            all_node_estimated_fidelity = []
            all_node_duration = []
            all_node_teleportation_success = []
            # print(dc.dataframe)
            for column in dc.dataframe.columns:
                # Flatten the lists in the column
                if "Estimated" in column:
                    flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
                    all_node_estimated_fidelity.append(sum(flattened_values) / len(flattened_values))
                elif "Duration" in column:
                    all_node_duration.append(sum(dc.dataframe[column]) / len(dc.dataframe[column]))
                elif "Teleportation" in column:
                    all_node_teleportation_success.append(sum(dc.dataframe[column]) / len(dc.dataframe[column]))
                else:
                    flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
                    all_node_actual_fidelity.append(sum(flattened_values) / len(flattened_values))
            final_fidelity = 1
            for fidelity in all_node_actual_fidelity:
                final_fidelity *= fidelity
            final_estimated_fidelity = 1
            for fidelity in all_node_estimated_fidelity:
                final_estimated_fidelity *= fidelity
            average_duration = sum(all_node_duration) / len(all_node_duration)
            average_teleportation_success = sum(all_node_teleportation_success) / len(all_node_teleportation_success)
            print("*" * 50)
            print(f"Skip noise: {True}")
            print(f"Current Node Count: {node_count}")
            print(f"Teleportation success count: {average_teleportation_success}")
            print(f"Final estimated fidelity: {final_estimated_fidelity}")
            print(f"Final fidelity: {final_fidelity}")
            print("Average duration: ", average_duration)
            data[node_count] = {"actual_fidelity": final_fidelity,
                                    "estimated_fidelity": final_estimated_fidelity,
                                    "average_duration": average_duration,
                                    "average_teleportation_success": average_teleportation_success}
            entangle_protocol.stop()
            # check the time condition
            if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
                # case we have time overflow, reset the simulation and run again with different RNG
                ns.sim_reset()
                new_rng = np.random.RandomState()
                if new_rng == ns.get_random_state():
                    raise ValueError("Random state is not resetting")
                ns.set_random_state(rng=new_rng)
            with open(os.path.join(save_dir, f"entanglement_results_{max_node}_node.json"),
                      "w") as f:
                json.dump(data, f, indent=4)
            progress.update(task, advance=1)

def experiment_with_increasing_distance(max_dis, save_dir):
    # create a network

    data = {}
    from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Nodes...", total=max_dis)
        for node_dis in range(500, int(max_dis*1000) + 1, 500):
            nodes_list = [f"Node_{i}" for i in range(2)]
            network = setup_network_parallel(nodes_list, "hop-by-hop-entangle",
                                    memory_capacity=24001, memory_depolar_rate=24583,
                                    node_distance=node_dis/1000)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            entangle_protocol, dc = example_sim_run(sample_nodes, num_runs=1,
                                                    memory_depolar_rate=24583,
                                                    node_distance=node_dis/1000,
                                                    max_entangle_pairs=24000,
                                                    skip_noise=True
                                                    )
            entangle_protocol.start()
            # run the protocol
            ns.sim_run()

            # compute average for each column
            all_node_actual_fidelity = []
            all_node_estimated_fidelity = []
            all_node_duration = []
            all_node_teleportation_success = []
            # print(dc.dataframe)
            for column in dc.dataframe.columns:
                # Flatten the lists in the column
                if "Estimated" in column:
                    flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
                    all_node_estimated_fidelity.append(sum(flattened_values) / len(flattened_values))
                elif "Duration" in column:
                    all_node_duration.append(sum(dc.dataframe[column]) / len(dc.dataframe[column]))
                elif "Teleportation" in column:
                    all_node_teleportation_success.append(sum(dc.dataframe[column]) / len(dc.dataframe[column]))
                else:
                    flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
                    all_node_actual_fidelity.append(sum(flattened_values) / len(flattened_values))
            final_fidelity = 1
            for fidelity in all_node_actual_fidelity:
                final_fidelity *= fidelity
            final_estimated_fidelity = 1
            for fidelity in all_node_estimated_fidelity:
                final_estimated_fidelity *= fidelity
            average_duration = sum(all_node_duration) / len(all_node_duration)
            average_teleportation_success = sum(all_node_teleportation_success) / len(all_node_teleportation_success)
            print("*" * 50)
            print(f"Skip noise: {True}")
            print(f"Current Node Count: {node_dis}")
            print(f"Teleportation success count: {average_teleportation_success}")
            print(f"Final estimated fidelity: {final_estimated_fidelity}")
            print(f"Final fidelity: {final_fidelity}")
            print("Average duration: ", average_duration)
            data[node_dis] = {"actual_fidelity": final_fidelity,
                                    "estimated_fidelity": final_estimated_fidelity,
                                    "average_duration": average_duration,
                                    "average_teleportation_success": average_teleportation_success}
            entangle_protocol.stop()
            # check the time condition
            if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
                # case we have time overflow, reset the simulation and run again with different RNG
                ns.sim_reset()
                new_rng = np.random.RandomState()
                if new_rng == ns.get_random_state():
                    raise ValueError("Random state is not resetting")
                ns.set_random_state(rng=new_rng)
            with open(os.path.join(save_dir, f"concurrent_entanglement_results_2nodes_{max_dis}km.json"),
                      "w") as f:
                json.dump(data, f, indent=4)
            progress.update(task, advance=1)


def experiment_with_increasing_pairs(max_node, save_dir, skip_noise=False):
    # create a network
    nodes_list = [f"Node_{i}" for i in range(max_node)]
    network = setup_network(nodes_list, "hop-by-hop-entangle",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=20, source_delay=1e5)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    data = {}
    max_pairs = 128
    from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=max_pairs)
        for entangle_pairs in range(2, max_pairs + 1):
            entangle_protocol, dc = example_sim_run(sample_nodes, num_runs=1000,
                                                    memory_depolar_rate=100,
                                                    node_distance=20,
                                                    max_entangle_pairs=entangle_pairs,
                                                    skip_noise=skip_noise
                                                    )
            entangle_protocol.start()
            # run the protocol
            ns.sim_run()

            # compute average for each column
            all_node_actual_fidelity = []
            all_node_estimated_fidelity = []
            all_node_duration = []
            all_node_teleportation_success = []
            # print(dc.dataframe)
            for column in dc.dataframe.columns:
                # Flatten the lists in the column
                if "Estimated" in column:
                    flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
                    all_node_estimated_fidelity.append(sum(flattened_values) / len(flattened_values))
                elif "Duration" in column:
                    all_node_duration.append(sum(dc.dataframe[column]) / len(dc.dataframe[column]))
                elif "Teleportation" in column:
                    all_node_teleportation_success.append(sum(dc.dataframe[column]) / len(dc.dataframe[column]))
                else:
                    flattened_values = [item for sublist in dc.dataframe[column] for item in sublist]
                    all_node_actual_fidelity.append(sum(flattened_values) / len(flattened_values))
            final_fidelity = 1
            for fidelity in all_node_actual_fidelity:
                final_fidelity *= fidelity
            final_estimated_fidelity = 1
            for fidelity in all_node_estimated_fidelity:
                final_estimated_fidelity *= fidelity
            average_duration = sum(all_node_duration) / len(all_node_duration)
            average_teleportation_success = sum(all_node_teleportation_success) / len(all_node_teleportation_success)
            print("*" * 50)
            print(f"Skip noise: {skip_noise}")
            print(f"Entangle pairs: {entangle_pairs}")
            print(f"Teleportation success count: {average_teleportation_success}")
            print(f"Final estimated fidelity: {final_estimated_fidelity}")
            print(f"Final fidelity: {final_fidelity}")
            print("Average duration: ", average_duration)
            data[entangle_pairs] = {"actual_fidelity": final_fidelity,
                                    "estimated_fidelity": final_estimated_fidelity,
                                    "average_duration": average_duration,
                                    "average_teleportation_success": average_teleportation_success}
            entangle_protocol.stop()
            # check the time condition
            if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
                # case we have time overflow, reset the simulation and run again with different RNG
                ns.sim_reset()
                new_rng = np.random.RandomState()
                if new_rng == ns.get_random_state():
                    raise ValueError("Random state is not resetting")
                ns.set_random_state(rng=new_rng)
            with open(os.path.join(save_dir, f"entanglement_results_2_node_{max_pairs}_pairs_noise_{skip_noise}.json"),
                      "w") as f:
                json.dump(data, f, indent=4)
            progress.update(task, advance=1)


def main():
    # exit(0)
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
    # experiment_with_increasing_pairs(2, "./entanglement_results", skip_noise=pop_noise)
    # experiment_with_increasing_nodes(5, "./entanglement_results")
    experiment_with_increasing_distance(0.5, "./entanglement_results")



if __name__ == '__main__':
    main()
