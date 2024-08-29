import json
import operator
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

from utils.NetworkSetup import setup_network
from utils import Logging
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
from protocols.Verification import Verification
import netsquid.qubits.operators as ops


class VerifyExample(LocalProtocol):
    """
    Protocol for a complete verification example.
    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 m_size=3,
                 batch_size=10):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleVerification")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=True)
        null_logger = Logging.Logger("null", logging_enabled=False)
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
                verify_protocol = Verification(node=node,
                                               name=f"verify_{node.name}->{network_nodes[index - 1].name}",
                                               entangled_node=network_nodes[index - 1].name,
                                               purification_protocol=pure_protocol,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               m_size=m_size,
                                               batch_size=batch_size,
                                               logger=self.logger,
                                               is_top_layer=True,
                                               max_entangled_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)

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
                # Initialize the purification protocol
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
                self.add_subprotocol(pure_protocol)
                verify_protocol = Verification(node=node,
                                               name=f"verify_{node.name}->{network_nodes[index + 1].name}",
                                               entangled_node=network_nodes[index + 1].name,
                                               purification_protocol=pure_protocol,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               m_size=m_size,
                                               batch_size=batch_size,
                                               logger=self.logger,
                                               is_top_layer=True,
                                               max_entangled_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)



    def run(self):
        self.start_subprotocols()
        for subprotoco, val in self.subprotocols.items():
            print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            verify_protocols = []
            for subprotocol in self.subprotocols.values():
                if "verify" in subprotocol.name:
                    verify_protocols.append(subprotocol)

            wait_signals = [self.await_signal(p, MessageType.PROTOCOL_FINISHED)
                            for p in verify_protocols]

            yield reduce(operator.and_, wait_signals)

            results = [p.get_signal_result(MessageType.PROTOCOL_FINISHED)
                       for p in verify_protocols]
            """
            result = {batch_id: [mem_pos1, mem_pos2, ...]}
            """
            result_dic = {}
            node_index = 0
            for i in range(0, len(results), 2):
                entangle_node = self.all_nodes[node_index + 1].name
                node = self.all_nodes[node_index].name
                node_pair_res = []
                for res in results[i].values():
                    node_pair_res += res
                # measure the actual fidelity
                actual_fidelities = {}
                total_batch = len(node_pair_res)
                teleport_success_count = 0
                for mem_pos in node_pair_res:
                    qubit_a = self.all_nodes[i].subcomponents[f"{entangle_node}_qmemory"].pop(mem_pos)[0]
                    qubit_b = self.all_nodes[i + 1].subcomponents[f"{node}_qmemory"].pop(mem_pos)[0]
                    q_a_name = str(qubit_a.name).split("#")[-1].split("-")[0]
                    q_b_name = str(qubit_b.name).split("#")[-1].split("-")[0]
                    # print(f"Qubit names: {q_a_name}, {q_b_name}")
                    if q_a_name != q_b_name:
                        raise ValueError(f"Qubit names are not the same at {mem_pos}: {q_a_name}, {q_b_name}")
                    # if q_a.qstate != q_b.qstate:
                    #     raise ValueError(f"Qubit states are not the same: {q_a.qstate}, {q_b.qstate}")
                    f = qapi.fidelity([qubit_a, qubit_b], ks.b00)
                    if 0 < f < 0.99:
                        raise ValueError(f"Fidelity is not correct: {f}, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}")
                    actual_fidelities[mem_pos] = f
                    # start_teleportation, generate a qubit for teleportation
                    # rotate the qubit to y0 state
                    qubit = qapi.create_qubits(1)[0]
                    qapi.operate(qubit, ops.H)
                    qapi.operate(qubit, ops.S)
                    # teleport the qubit
                    fid = self.test_teleportation(qubit_a, qubit_b, qubit)
                    if fid > 0.99:
                        teleport_success_count += 1
                result_dic[f"{node}->{entangle_node}"] = {
                    "total_batch": total_batch,
                    "actual_fidelities": actual_fidelities,
                    "teleport_success_count": teleport_success_count
                }
            print(result_dic)
            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})

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

def example_sim_run(nodes, num_runs, memory_depolar_rate,
                    node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size):
    """
    Run the example verification protocol
    :param nodes: list of nodes
    :param num_runs: number of runs
    :param memory_depolar_rate: memory depolar rate
    :param node_distance: node distance
    :param max_entangle_pairs: maximum entangle pairs
    :param target_fidelity: target fidelity
    :param m_size: m size
    :param batch_size: batch size
    :return:
    """
    # Create the protocol
    verify_example = VerifyExample(network_nodes=nodes,
                             num_runs=num_runs,
                             max_entangle_pairs=max_entangle_pairs,
                             memory_depolar_rate=memory_depolar_rate,
                             node_distance=node_distance,
                             target_fidelity=target_fidelity,
                             m_size=m_size,
                             batch_size=batch_size)
    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Purification Run {result['run_index']} completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=verify_example,
                                     event_type=Signals.SUCCESS.value))
    return verify_example, dc

def run_experiment(nodes_count):
    nodes_list = [f"Node_{i}" for i in range(nodes_count)]
    network = setup_network(nodes_list, "hop-by-hop-verification",
                            memory_capacity=16, memory_depolar_rate=100,
                            node_distance=20, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    verify_example, dc = example_sim_run(sample_nodes, num_runs=1, memory_depolar_rate=100,
                                           node_distance=20,
                                           max_entangle_pairs=16, target_fidelity=0.995, m_size=3, batch_size=10)
    # Run the simulation
    verify_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.data
    print(results)
if __name__ == '__main__':
    run_experiment(2)