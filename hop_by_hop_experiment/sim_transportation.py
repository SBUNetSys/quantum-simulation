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
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
from protocols.Verification import Verification
from protocols.Transport import Transportation
import netsquid.qubits.operators as ops

class TransportWithVerificationExample(LocalProtocol):
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
                 batch_size=10,
                 qubits_to_transport=1,
                 skip_noise=False):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.m_size = m_size
        self.batch_size = batch_size
        self.qubits_to_transport = qubits_to_transport
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleTransportation")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=True)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.skip_noise = skip_noise

        # Initialize the controlled unitary matrix and measurement operators
        CU_matrix = controlled_unitary(batch_size)
        measurement_m0, measurement_m1 = measure_operator()
        CU_gate = ops.Operator("CU_Gate", CU_matrix)
        CCU_gate = CU_gate.conj

        # initialize the protocol for each node
        for index, node in enumerate(network_nodes):
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            qubit_input_protocols = []
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
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                verify_protocol = Verification(node=node,
                                               name=f"verify_{node.name}->{network_nodes[index - 1].name}",
                                               entangled_node=network_nodes[index - 1].name,
                                               purification_protocol=pure_protocol,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               m_size=m_size,
                                               batch_size=batch_size,
                                               CU_Gate=CU_gate,
                                               CCU_Gate=CCU_gate,
                                               measurement_m0=measurement_m0,
                                               measurement_m1=measurement_m1,
                                               logger=null_logger,
                                               is_top_layer=False,
                                               max_entangled_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)
                qubit_input_protocols.append(verify_protocol)

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
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                verify_protocol = Verification(node=node,
                                               name=f"verify_{node.name}->{network_nodes[index + 1].name}",
                                               entangled_node=network_nodes[index + 1].name,
                                               purification_protocol=pure_protocol,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               m_size=m_size,
                                               batch_size=batch_size,
                                               CU_Gate=CU_gate,
                                               CCU_Gate=CCU_gate,
                                               measurement_m0=measurement_m0,
                                               measurement_m1=measurement_m1,
                                               logger=null_logger,
                                               is_top_layer=False,
                                               max_entangled_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)
                qubit_input_protocols.append(verify_protocol)
            # add transport protocol
            entangle_name = ""
            if index + 1 < len(network_nodes):
                entangle_name = network_nodes[index + 1].name
            else:
                entangle_name = network_nodes[index - 1].name
            transport = Transportation(node=node,
                                       name=f"transport_{node.name}",
                                       qubit_ready_protocols=qubit_input_protocols,
                                       entangled_node=entangle_name,
                                       source=network_nodes[0].name,
                                       destination=network_nodes[-1].name,
                                       cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                       transmitting_qubit_size=qubits_to_transport,
                                       logger=self.logger,
                                       is_top_layer=True,
                                       )
            self.add_subprotocol(transport)

    def run(self):
        self.start_subprotocols()
        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()

            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"], Signals.SUCCESS)
            end_time = sim_time()
            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(Signals.SUCCESS, self)
            """
            result = {entangle_node: name, mem_poses:[]}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "duration": end_time - start_time,}
            for mem_pos in results["mem_poses"]:
                result_dic["total_count"] += 1
                # get the qubit
                qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                    mem_pos, skip_noise=True)[0]
                # measure the state
                fidelity = qapi.fidelity(qubit, ns.y0)
                if fidelity > 0.99:
                    result_dic["teleport_success_count"] += 1
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()
        # remove any gates after finish running
        for subprotocol in self.subprotocols.values():
            if "verify" in subprotocol.name:
                subprotocol.clean_gates()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()
class TransportWithPurificationExample(LocalProtocol):
    """
    Protocol for a complete verification example.
    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 qubits_to_transport=1,
                 skip_noise=False):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.qubits_to_transport = qubits_to_transport
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleTransportation")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=True)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.skip_noise = skip_noise


        # initialize the protocol for each node
        for index, node in enumerate(network_nodes):
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            qubit_input_protocols = []
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
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                qubit_input_protocols.append(pure_protocol)

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
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                qubit_input_protocols.append(pure_protocol)
            # add transport protocol
            entangle_name = network_nodes[index + 1].name if index + 1 < len(network_nodes) \
                else network_nodes[index - 1].name
            transport = Transportation(node=node,
                                       name=f"transport_{node.name}",
                                       qubit_ready_protocols=qubit_input_protocols,
                                       entangled_node=entangle_name,
                                       source=network_nodes[0].name,
                                       destination=network_nodes[-1].name,
                                       cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                       transmitting_qubit_size=qubits_to_transport,
                                       logger=self.logger,
                                       is_top_layer=True,
                                       )
            self.add_subprotocol(transport)

    def run(self):
        self.start_subprotocols()
        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()

            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"], Signals.SUCCESS)
            end_time = sim_time()
            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(Signals.SUCCESS, self)
            """
            result = {entangle_node: name, mem_poses:[]}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "duration": end_time - start_time,}
            for mem_pos in results["mem_poses"]:
                result_dic["total_count"] += 1
                # get the qubit
                qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                    mem_pos, skip_noise=True)[0]
                # measure the state
                fidelity = qapi.fidelity(qubit, ns.y0)
                if fidelity > 0.99:
                    result_dic["teleport_success_count"] += 1
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]

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

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()


def example_sim_run_with_verification(nodes, num_runs, memory_depolar_rate,
                    node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size,
                                      qubit_to_transport,
                    skip_noise=True):
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
    :param qubit_to_transport: number of qubits to transmit
    :param skip_noise: skip noise when popping qubits
    :return:
    """
    # Create the protocol
    transport_example = TransportWithVerificationExample(network_nodes=nodes,
                                   num_runs=num_runs,
                                   max_entangle_pairs=max_entangle_pairs,
                                   memory_depolar_rate=memory_depolar_rate,
                                   node_distance=node_distance,
                                   target_fidelity=target_fidelity,
                                   m_size=m_size,
                                   batch_size=batch_size,
                                   skip_noise=skip_noise,
                                   qubits_to_transport=qubit_to_transport)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # print(f"Purification Run {result['run_index']} completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc

def example_sim_run_with_purification(nodes, num_runs, memory_depolar_rate,
                    node_distance, max_entangle_pairs, target_fidelity,
                                      qubit_to_transport,
                    skip_noise=True):
    """
    Run the example verification protocol
    :param nodes: list of nodes
    :param num_runs: number of runs
    :param memory_depolar_rate: memory depolar rate
    :param node_distance: node distance
    :param max_entangle_pairs: maximum entangle pairs
    :param target_fidelity: target fidelity
    :param qubit_to_transport: number of qubits to transmit
    :param skip_noise: skip noise when popping qubits
    :return:
    """
    # Create the protocol
    transport_example = TransportWithPurificationExample(network_nodes=nodes,
                                   num_runs=num_runs,
                                   max_entangle_pairs=max_entangle_pairs,
                                   memory_depolar_rate=memory_depolar_rate,
                                   node_distance=node_distance,
                                   target_fidelity=target_fidelity,
                                   skip_noise=skip_noise,
                                   qubits_to_transport=qubit_to_transport)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # print(f"Purification Run {result['run_index']} completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc

def run_test_example_with_verification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(4)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=3, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=1, memory_depolar_rate=100,
                                         node_distance=3,
                                         max_entangle_pairs=10, target_fidelity=0.995, m_size=3, batch_size=4,
                                         skip_noise=True,qubit_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results.columns)
    print(results)

def run_test_example_with_purification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(4)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=3, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1, memory_depolar_rate=100,
                                         node_distance=3,
                                         max_entangle_pairs=10, target_fidelity=0.995,
                                         skip_noise=True,qubit_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results.columns)
    print(results)

if __name__ == '__main__':
    run_test_example_with_purification()
    # run_test_example_with_verification()