import json
import os.path
import sys

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

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

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network
from utils import Logging, GenSwappingTree
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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
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
                                          max_pairs=1,
                                          logger=self.logger,
                                          is_top_layer=True)
            self.add_subprotocol(end_to_end)

    def run(self):

        end_time = None
        self.start_subprotocols()
        start_time = sim_time()
        for i in range(self.num_runs):
            # repeat experiment we start from here
            if end_time is not None:
                start_time = sim_time()
            yield (self.await_signal(self.subprotocols[f"e2e_{self.final_entanglement[0]}"],
                                     MessageType.SWAP_FINISHED) &
                   self.await_signal(self.subprotocols[f"e2e_{self.final_entanglement[1]}"], MessageType.SWAP_FINISHED))
            end_time = sim_time()

            # print(f"Swapping completed in {(end_time - start_time) / 1e9} seconds.")
            result_a = self.subprotocols[f"e2e_{self.final_entanglement[0]}"].get_signal_result(
                MessageType.SWAP_FINISHED, self)
            result_b = self.subprotocols[f"e2e_{self.final_entanglement[1]}"].get_signal_result(
                MessageType.SWAP_FINISHED, self)
            print(f"Swapping result: {result_a}, {result_b}")
            mem_pos_a = list(result_a[self.final_entanglement[1]].keys())
            mem_pos_b = list(result_b[self.final_entanglement[0]].keys())
            # check the final entanglement's fidelity
            qubit_a = self.nodes[self.final_entanglement[0]].qmemory.peek(mem_pos_a[0])[0]
            qubit_b = self.nodes[self.final_entanglement[1]].qmemory.peek(mem_pos_b[0])[0]

            # print(f"Qubit names: {q_a_name}, {q_b_name}")
            print(f"Final QState {qubit_a.qstate}, {qubit_b.qstate}")
            if str(qubit_a.qstate) != str(qubit_a.qstate):
                raise ValueError(f"Qubit states are not the same: {qubit_a.qstate}, {qubit_b.qstate}")

            # rd = qapi.reduced_dm([qubit_a, qubit_b])
            # fidelity_result = qapi.fidelity(rd, ks.b00)
            # fidelity_result = qapi.fidelity([qubit_a, qubit_b], ns.b00)
            # print(f"Fidelity of the final entanglement: {fidelity_result}")

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
            # start_teleportation, generate a qubit for teleportation
            # rotate the qubit to y0 state
            qubit = qapi.create_qubits(1)[0]
            qapi.operate(qubit, ops.H)
            qapi.operate(qubit, ops.S)
            teleport_res = self.test_teleportation(qubit_a, qubit_b, qubit)
            self.send_signal(Signals.SUCCESS, {"fidelity": teleport_res,
                                               "duration": end_time - start_time,
                                               "teleportation_success": 1 if teleport_res else 0, })
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

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


class EndToEndExampleWithDelay(LocalProtocol):
    """
    A simple example of a swapping protocol.
    """

    def __init__(self, network_nodes: list,
                 num_runs=1,
                 node_path=None,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 delay_time=0):
        if node_path is None:
            raise ValueError("node_path must be provided")
        # generate the swapping tree and levels
        swapping_nodes, _, _ = GenSwappingTree.generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.final_entanglement = (node_path[0], node_path[-1])
        self.all_nodes = network_nodes
        self.max_entangle_pairs = max_entangle_pairs
        self.logger = Logging.Logger("EndToEnd", logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.delay_time = delay_time

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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
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
                                          max_pairs=1,
                                          logger=self.logger,
                                          is_top_layer=True,
                                          delay_time=self.delay_time,)
            self.add_subprotocol(end_to_end)

    def run(self):

        end_time = None
        self.start_subprotocols()
        start_time = sim_time()
        for i in range(self.num_runs):
            # repeat experiment we start from here
            if end_time is not None:
                start_time = sim_time()
            yield (self.await_signal(self.subprotocols[f"e2e_{self.final_entanglement[0]}"],
                                     MessageType.SWAP_FINISHED) &
                   self.await_signal(self.subprotocols[f"e2e_{self.final_entanglement[1]}"], MessageType.SWAP_FINISHED))
            end_time = sim_time()

            # print(f"Swapping completed in {(end_time - start_time) / 1e9} seconds.")
            # print(f"Swapping completed at {end_time}")
            result_a = self.subprotocols[f"e2e_{self.final_entanglement[0]}"].get_signal_result(
                MessageType.SWAP_FINISHED, self)
            result_b = self.subprotocols[f"e2e_{self.final_entanglement[1]}"].get_signal_result(
                MessageType.SWAP_FINISHED, self)
            # if self.delay_time > 0:
            #     yield self.await_timer(self.delay_time)
            # print(f"Swapping result: {result_a}, {result_b}")
            mem_pos_a = list(result_a[self.final_entanglement[1]].keys())
            mem_pos_b = list(result_b[self.final_entanglement[0]].keys())
            # check the final entanglement's fidelity
            q_mem_a = self.all_nodes[0].subcomponents[f"{self.all_nodes[1].name}_qmemory"]
            q_mem_b = self.all_nodes[-1].subcomponents[f"{self.all_nodes[-2].name}_qmemory"]
            # qubit_a = self.nodes[self.final_entanglement[0]].qmemory.peek(mem_pos_a[0])[0]
            # qubit_b = self.nodes[self.final_entanglement[1]].qmemory.peek(mem_pos_b[0])[0]
            qubit_a = q_mem_a.pop(mem_pos_a[0], skip_noise=False)[0]
            qubit_b = q_mem_b.pop(mem_pos_b[0], skip_noise=False)[0]

            # print(f"Qubit names: {q_a_name}, {q_b_name}")
            # print(f"Final QState {qubit_a.qstate}, {qubit_b.qstate}")
            if str(qubit_a.qstate) != str(qubit_a.qstate):
                raise ValueError(f"Qubit states are not the same: {qubit_a.qstate}, {qubit_b.qstate}")
            # rd = qapi.reduced_dm([qubit_a, qubit_b])
            # fidelity_result = qapi.fidelity(rd, ks.b00)
            # fidelity_result = qapi.fidelity([qubit_a, qubit_b], ns.b00)
            # print(f"Fidelity of the final entanglement: {fidelity_result}")

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
            # start_teleportation, generate a qubit for teleportation
            # rotate the qubit to y0 state
            # qubit = qapi.create_qubits(1)[0]
            # qapi.operate(qubit, ops.H)
            # qapi.operate(qubit, ops.S)
            # teleport_res = self.test_teleportation(qubit_a, qubit_b, qubit)
            fid = qapi.fidelity([qubit_a, qubit_b], ns.b00)
            self.send_signal(Signals.SUCCESS, {"fidelity": fid,
                                               "delay_time": self.delay_time, })
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

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


def example_sim_run_delay(nodes, num_runs, memory_depolar_rate,
                          node_distance, max_entangle_pairs, target_fidelity, delay_time):
    e2e_example = EndToEndExampleWithDelay(network_nodes=nodes,
                                  num_runs=num_runs,
                                  node_path=[node.name for node in nodes],
                                  max_entangle_pairs=max_entangle_pairs,
                                  memory_depolar_rate=memory_depolar_rate,
                                  node_distance=node_distance,
                                  target_fidelity=target_fidelity,
                                  delay_time=delay_time)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run completed: {result}")
        return result

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=e2e_example, event_type=Signals.SUCCESS.value))
    return e2e_example, dc


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
        # print(f"Run completed: {result}")
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
    # ax.set_ylim(top=1)
    if xlim:
        ax.set_xlim(xlim)
    # Group x labels
    # ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=20))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))  # Format the labels as integers
    plt.tight_layout()
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    plt.show()


def run_e2e_test(distances=1):
    nodes_list = [f"Node_{i}" for i in range(5)]
    network = setup_network(nodes_list, "end-to-end-network",
                            memory_capacity=5, memory_depolar_rate=63109,
                            node_distance=distances, source_delay=1)
    sample_nodes = [node for node in network.nodes.values()]
    end_to_end_example, dc = example_sim_run(sample_nodes,
                                             1,
                                             63109,
                                             distances,
                                             5,
                                             0.98)
    end_to_end_example.start()
    ns.sim_run()
    collected_data = dc.dataframe
    print(collected_data)


def run_e2e_multi_node(max_node, distances=3):
    os.makedirs("./multi-node", exist_ok=True)
    final_data = {}
    for node_count in range(3, max_node + 1):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "end-to-end-network",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=distances, source_delay=1)
        sample_nodes = [node for node in network.nodes.values()]
        end_to_end_example, dc = example_sim_run(sample_nodes,
                                                 1000,
                                                 100,
                                                 distances,
                                                 2,
                                                 0.995)
        end_to_end_example.start()
        ns.sim_run()
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_count}->{c}: {collected_data[c].mean()}")
        final_data[node_count] = node_data
        end_to_end_example.stop()
        ns.sim_reset()
    with open(f"./multi-node/max_{max_node}_nodes.json", "w") as f:
        json.dump(final_data, f)


def plot_multi_node(max_node):
    with open(f"./multi-node/max_{max_node}_nodes.json", "r") as f:
        final_data = json.load(f)
    # plot the fidelity
    x = [int(i) for i in final_data.keys()]
    y_fid = [final_data[str(i)]["fidelity"] for i in x]
    y_time = [final_data[str(i)]["duration"] for i in x]
    y_tel = [final_data[str(i)]["teleportation_success"] for i in x]

    plot_line([x, x], [y_fid, y_tel],
              "Multiple Node End to End Fidelity and Teleportation Success Comparison",
              "Node Count",
              "Success/Fidelity",
              ["Fidelity", "Teleportation Success Rate"],
              xlim=[3, max_node],
              save_dir="./multi-node/figures")
    plot_line([x], [y_time],
              "Multiple Node End to End Entanglement Time Comparison",
              "Node Count",
              "Time (ns)",
              ["Time"],
              xlim=[3, max_node],
              save_dir="./multi-node/figures")


def run_e2e_distance(max_distance):
    os.makedirs("./distance", exist_ok=True)
    final_data = {}
    node_count = 5
    for distances in range(1, max_distance):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "end-to-end-network",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=distances, source_delay=1)
        sample_nodes = [node for node in network.nodes.values()]
        end_to_end_example, dc = example_sim_run(sample_nodes,
                                                 1000,
                                                 100,
                                                 distances,
                                                 2,
                                                 0.995)
        end_to_end_example.start()
        ns.sim_run()
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{distances}->{c}: {collected_data[c].mean()}")
        final_data[distances] = node_data
        end_to_end_example.stop()
        ns.sim_reset()
    with open(f"./distance/max_{max_distance}_km.json", "w") as f:
        json.dump(final_data, f)


def plot_distance(max_distance):
    with open(f"./distance/max_{max_distance}_km.json", "r") as f:
        final_data = json.load(f)
    # plot the fidelity
    x = [int(i) for i in final_data.keys()]
    y_fid = [final_data[str(i)]["fidelity"] for i in x]
    y_time = [final_data[str(i)]["duration"] for i in x]
    y_tel = [final_data[str(i)]["teleportation_success"] for i in x]

    plot_line([x, x], [y_fid, y_tel],
              "Multi Distance (5 Nodes) End to End Fidelity and Teleportation Success Comparison",
              "Node Distance (Km)",
              "Success/Fidelity",
              ["Fidelity", "Teleportation Success Rate"],
              # xlim=[1, max_distance],
              save_dir="./distance/figures")
    plot_line([x], [y_time],
              "Multi Distance (5 Nodes) End to End Entanglement Time Comparison",
              "Node Distance (Km)",
              "Time (ns)",
              ["Time"],
              # xlim=[3, max_distance],
              save_dir="./distance/figures")

def run_e2e_delay(max_delay_time=2000):
    os.makedirs("./delay_result", exist_ok=True)
    final_data = {}
    node_count = 3
    for delays in range(0, max_delay_time + 1, 100):
        print(f"Running Delay: {delays}ns")
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "end-to-end-network",
                                memory_capacity=128, memory_depolar_rate=63109,
                                node_distance=1, source_delay=1)
        sample_nodes = [node for node in network.nodes.values()]
        end_to_end_example, dc = example_sim_run_delay(sample_nodes,
                                                 1000,
                                                 63109,
                                                 1,
                                                 128,
                                                 0.98,
                                                       delay_time=delays)
        end_to_end_example.start()
        ns.sim_run()
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{delays}->{c}: {collected_data[c].mean()}")
        final_data[delays] = node_data
        with open(f"./delay_result/e2e_{node_count}nodes_max_delay{max_delay_time}ns_fid.json", "w") as f:
            json.dump(final_data, f, indent=4)
        end_to_end_example.stop()
        # ns.set_random_state(rng=np.random.RandomState())
        ns.sim_reset()

if __name__ == '__main__':
    # seed = np.random.randint(0, 10000)
    # seed = 524
    # np.random.seed(seed)
    # print(f'seed {seed}')

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
    # run_e2e_test(1)
    run_e2e_delay(100000)
    # run_e2e_multi_node(11, distances=3)
    # matplotlib.use('TkAgg')
    # import matplotlib
    # print(matplotlib.get_backend())
    # matplotlib.use('QtAgg')
    # matplotlib.use('module://backend_interagg')
    # plot_multi_node(11)
    # run_e2e_distance(10)
    # plot_distance(10)
