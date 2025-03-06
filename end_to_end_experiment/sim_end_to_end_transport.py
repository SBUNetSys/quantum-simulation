import gc
import json
import os
import sys
import traceback

import numpy as np
import pydynaa as pd
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.operators as ops
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network
from utils import Logging, GenSwappingTree
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
from protocols.EndToEnd import EndToEndProtocol
from protocols.Verification import Verification
from protocols.Transport import Transportation
from utils.SignalMessages import ProtocolFinishedSignalMessage


class EndToEndTransportWithPurificationExample(LocalProtocol):
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
                 qubits_to_transport=1):
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
                                          max_pairs=self.max_entangle_pairs - 1,
                                          logger=self.logger,
                                          is_top_layer=False)
            self.add_subprotocol(end_to_end)
            if index == 0:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[-1].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)
            if index == len(network_nodes) - 1:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[0].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)

    def run(self):

        end_time = None
        self.start_subprotocols()
        start_time = sim_time()
        for index in range(self.num_runs):
            if end_time is not None:
                start_time = sim_time()
            yield self.await_signal(self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_FINISHED, self)
            """
            result = {entangle_node: name, mem_poses:[]}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "teleport_fids": [],
                          "duration": end_time - start_time, }
            for mem_pos, fid in results["results"].items():
                result_dic["total_count"] += 1
                # get the qubit
                # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                #     mem_pos, skip_noise=True)[0]
                # # measure the state
                # fidelity = qapi.fidelity(qubit, ns.y0)
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
                result_dic['teleport_fids'].append(fid)
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            # return
            for subprotocol_name, subprotocol in self.subprotocols.items():
                if "purify" in subprotocol_name:
                    subprotocol.cc_message_handler.send_signal(MessageType.VERIFICATION_FINISHED,
                                                               ProtocolFinishedSignalMessage(
                                                                   from_protocol=subprotocol,
                                                                   from_node=subprotocol.node.name,
                                                                   entangle_node=subprotocol.entangled_node
                                                               ))
            p_done = False
            p_start = sim_time()
            while not p_done:
                yield self.await_timer(1000)
                all_done = True
                for subprotocol_name, subprotocol in self.subprotocols.items():
                    if "purify" in subprotocol_name:
                        if subprotocol.is_running:
                            all_done = False
                            # print(f"Subprotocol {subprotocol.name} still running.")
                if all_done:
                    p_done = True
                if sim_time() - p_start > 100000:
                    break
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


class EndToEndTransportWithPurificationThroughput(LocalProtocol):
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
                 qubits_to_transport=1):
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
                                          max_pairs=self.max_entangle_pairs - 1,
                                          logger=self.logger,
                                          is_top_layer=False)
            self.add_subprotocol(end_to_end)
            if index == 0:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[-1].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)
            if index == len(network_nodes) - 1:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[0].name,
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
        while True:
            yield self.await_signal(self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_SUCCESS)
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_SUCCESS, self)
            self.send_signal(Signals.SUCCESS, results)

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


class EndToEndTransportWithVerificationExample(LocalProtocol):
    def __init__(self, network_nodes,
                 node_path,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 m_size=3,
                 batch_size=10,
                 qubits_to_transport=1,
                 skip_noise=False,
                 CU_gate=None,
                 CCU_gate=None,):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        swapping_nodes, _, _ = GenSwappingTree.generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.final_entanglement = (node_path[0], node_path[-1])
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.max_swap_qubit = qubits_to_transport
        self.m_size = m_size
        self.batch_size = batch_size
        self.qubits_to_transport = qubits_to_transport
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleTransportation")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.skip_noise = skip_noise

        # Initialize the controlled unitary matrix and measurement operators
        # CU_matrix = controlled_unitary(batch_size)
        measurement_m0, measurement_m1 = measure_operator()
        # CU_gate = ops.Operator("CU_Gate", CU_matrix)
        # CCU_gate = CU_gate.conj
        CU_gate = CU_gate
        CCU_gate = CCU_gate

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
                                             max_purify_pair=self.max_entangle_pairs,
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
                                               max_verify_pairs=self.max_entangle_pairs,
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
                                             max_purify_pair=self.max_entangle_pairs,
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
                                               max_verify_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)
                qubit_input_protocols.append(verify_protocol)

            # Add end to end protocol to each node
            end_to_end = EndToEndProtocol(node=node,
                                          name=f"e2e_{node.name}",
                                          swapping_nodes=self.swap_nodes,
                                          final_entanglement=self.final_entanglement,
                                          cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                          qubit_ready_protocols=qubit_input_protocols,
                                          max_pairs=self.max_entangle_pairs - 1,
                                          logger=self.logger,
                                          is_top_layer=False)
            self.add_subprotocol(end_to_end)
            # add transport protocol
            if index == 0:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[-1].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)
            if index == len(network_nodes) - 1:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[0].name,
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
        start_time = sim_time()
        for index in range(self.num_runs):
            yield self.await_signal(self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_FINISHED, self)
            """
            result = {entangle_node: name, mem_poses:[]}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "teleport_fids": [],
                          "duration": end_time - start_time, }
            for mem_pos, fid in results["results"].items():
                result_dic["total_count"] += 1
                # get the qubit
                # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                #     mem_pos, skip_noise=True)[0]
                # # measure the state
                # fidelity = qapi.fidelity(qubit, ns.y0)
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
                result_dic['teleport_fids'].append(fid)
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
            # for subprotocol_name, subprotocol in self.subprotocols.items():
            #     if "purify" in subprotocol_name:
            #         subprotocol.cc_message_handler.send_signal(MessageType.VERIFICATION_FINISHED,
            #                                                    ProtocolFinishedSignalMessage(
            #                                                        from_protocol=subprotocol,
            #                                                        from_node=subprotocol.node.name,
            #                                                        entangle_node=subprotocol.entangled_node
            #                                                    ))

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            break
            # print(f"Start Stop Purification of run index {index}")
            p_done = False
            while not p_done:
                yield self.await_timer(1000)
                all_done = True
                for subprotocol_name, subprotocol in self.subprotocols.items():
                    if "purify" in subprotocol_name:
                        if subprotocol.is_running:
                            all_done = False
                if all_done:
                    p_done = True
            # print(f"Finished Stop Purification of run index {index}")
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


class EndToEndTransportWithVerificationThroughput(LocalProtocol):
    def __init__(self, network_nodes,
                 node_path,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 m_size=3,
                 batch_size=10,
                 qubits_to_transport=1,
                 skip_noise=False,
                 CU_gate=None,
                 CCU_gate=None,):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        swapping_nodes, _, _ = GenSwappingTree.generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.final_entanglement = (node_path[0], node_path[-1])
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.max_swap_qubit = qubits_to_transport
        self.m_size = m_size
        self.batch_size = batch_size
        self.qubits_to_transport = qubits_to_transport
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleTransportation")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.skip_noise = skip_noise

        # Initialize the controlled unitary matrix and measurement operators
        # CU_matrix = controlled_unitary(batch_size)
        measurement_m0, measurement_m1 = measure_operator()
        # CU_gate = ops.Operator("CU_Gate", CU_matrix)
        # CCU_gate = CU_gate.conj
        CU_gate = CU_gate
        CCU_gate = CCU_gate

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
                                             max_purify_pair=self.max_entangle_pairs,
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
                                               max_verify_pairs=self.max_entangle_pairs,
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
                                             max_purify_pair=self.max_entangle_pairs,
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
                                               max_verify_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)
                qubit_input_protocols.append(verify_protocol)

            # Add end to end protocol to each node
            end_to_end = EndToEndProtocol(node=node,
                                          name=f"e2e_{node.name}",
                                          swapping_nodes=self.swap_nodes,
                                          final_entanglement=self.final_entanglement,
                                          cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                          qubit_ready_protocols=qubit_input_protocols,
                                          max_pairs=self.max_entangle_pairs - 1,
                                          logger=self.logger,
                                          is_top_layer=False)
            self.add_subprotocol(end_to_end)
            # add transport protocol
            if index == 0:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[-1].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)
            if index == len(network_nodes) - 1:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[0].name,
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
        while True:
            yield self.await_signal(self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_SUCCESS)
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_SUCCESS, self)
            self.send_signal(Signals.SUCCESS, results)

        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")
        # start_time = sim_time()
        # for index in range(self.num_runs):
        #     yield self.await_signal(self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"],
        #                             MessageType.TRANSPORT_FINISHED)
        #     end_time = sim_time()
        #     results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
        #         MessageType.TRANSPORT_FINISHED, self)
        #     """
        #     result = {entangle_node: name, mem_poses:[]}
        #     """
        #     result_dic = {"teleport_success_count": 0,
        #                   "total_count": 0,
        #                   "teleport_success_rate": 0,
        #                   "teleport_fids": [],
        #                   "duration": end_time - start_time, }
        #     for mem_pos, fid in results["results"].items():
        #         result_dic["total_count"] += 1
        #         # get the qubit
        #         # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
        #         #     mem_pos, skip_noise=True)[0]
        #         # # measure the state
        #         # fidelity = qapi.fidelity(qubit, ns.y0)
        #         if fid > 0.99:
        #             result_dic["teleport_success_count"] += 1
        #         result_dic['teleport_fids'].append(fid)
        #     # final success rate
        #     result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
        #     # for subprotocol_name, subprotocol in self.subprotocols.items():
        #     #     if "purify" in subprotocol_name:
        #     #         subprotocol.cc_message_handler.send_signal(MessageType.VERIFICATION_FINISHED,
        #     #                                                    ProtocolFinishedSignalMessage(
        #     #                                                        from_protocol=subprotocol,
        #     #                                                        from_node=subprotocol.node.name,
        #     #                                                        entangle_node=subprotocol.entangled_node
        #     #                                                    ))
        #
        #     self.send_signal(Signals.SUCCESS, {"results": result_dic,
        #                                        "run_index": index})
        #     break
        #     # print(f"Start Stop Purification of run index {index}")
        #     p_done = False
        #     while not p_done:
        #         yield self.await_timer(1000)
        #         all_done = True
        #         for subprotocol_name, subprotocol in self.subprotocols.items():
        #             if "purify" in subprotocol_name:
        #                 if subprotocol.is_running:
        #                     all_done = False
        #         if all_done:
        #             p_done = True
        #     # print(f"Finished Stop Purification of run index {index}")
        #     for subprotocol in self.subprotocols.values():
        #         subprotocol.reset()
        # # remove any gates after finish running
        # for subprotocol in self.subprotocols.values():
        #     if "verify" in subprotocol.name:
        #         subprotocol.clean_gates()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()


def example_sim_run_with_purification(nodes,
                                      num_runs,
                                      memory_depolar_rate,
                                      node_distance,
                                      max_entangle_pairs,
                                      target_fidelity,
                                      qubits_to_transport):
    e2e_example = EndToEndTransportWithPurificationExample(network_nodes=nodes,
                                                           num_runs=num_runs,
                                                           node_path=[node.name for node in nodes],
                                                           max_entangle_pairs=max_entangle_pairs,
                                                           memory_depolar_rate=memory_depolar_rate,
                                                           node_distance=node_distance,
                                                           target_fidelity=target_fidelity,
                                                           qubits_to_transport=qubits_to_transport, )

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run completed: {result['run_index']}, fid{result['results']['teleport_fids']}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=e2e_example, event_type=Signals.SUCCESS.value))
    return e2e_example, dc


def example_sim_run_with_purification_throughput(nodes,
                                                 num_runs,
                                                 memory_depolar_rate,
                                                 node_distance,
                                                 max_entangle_pairs,
                                                 target_fidelity,
                                                 qubits_to_transport):
    e2e_example = EndToEndTransportWithPurificationThroughput(network_nodes=nodes,
                                                              num_runs=num_runs,
                                                              node_path=[node.name for node in nodes],
                                                              max_entangle_pairs=max_entangle_pairs,
                                                              memory_depolar_rate=memory_depolar_rate,
                                                              node_distance=node_distance,
                                                              target_fidelity=target_fidelity,
                                                              qubits_to_transport=qubits_to_transport, )

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Transported {len(result['results'])} Qubit")
        return result

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=e2e_example, event_type=Signals.SUCCESS.value))
    return e2e_example, dc


def example_sim_run_with_verification(nodes, num_runs, memory_depolar_rate,
                                      node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size,
                                      qubit_to_transport,
                                      skip_noise=True,
                                      CU_gate=None,
                                      CCU_gate=None,):
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
    transport_example = EndToEndTransportWithVerificationExample(network_nodes=nodes,
                                                                 num_runs=num_runs,
                                                                 max_entangle_pairs=max_entangle_pairs,
                                                                 memory_depolar_rate=memory_depolar_rate,
                                                                 node_distance=node_distance,
                                                                 target_fidelity=target_fidelity,
                                                                 node_path=[node.name for node in nodes],
                                                                 m_size=m_size,
                                                                 batch_size=batch_size,
                                                                 skip_noise=skip_noise,
                                                                 qubits_to_transport=qubit_to_transport,
                                                                 CU_gate=CU_gate,
                                                                 CCU_gate=CCU_gate)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Verification Run {result['run_index']} completed, fid{result['results']['teleport_fids']}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc


def example_sim_run_with_verification_throughput(nodes, num_runs, memory_depolar_rate,
                                                 node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size,
                                                 qubit_to_transport,
                                                 skip_noise=True,
                                                 CU_gate=None, CCU_gate=None):
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
    transport_example = EndToEndTransportWithVerificationThroughput(network_nodes=nodes,
                                                                    num_runs=num_runs,
                                                                    max_entangle_pairs=max_entangle_pairs,
                                                                    memory_depolar_rate=memory_depolar_rate,
                                                                    node_distance=node_distance,
                                                                    target_fidelity=target_fidelity,
                                                                    node_path=[node.name for node in nodes],
                                                                    m_size=m_size,
                                                                    batch_size=batch_size,
                                                                    skip_noise=skip_noise,
                                                                    qubits_to_transport=qubit_to_transport,
                                                                    CCU_gate=CCU_gate,
                                                                    CU_gate=CU_gate)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Verification Transported {len(result['results'])} Qubit")
        return result

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc


def run_test_example_with_purification(qubit_number=2):
    nodes_list = [f"Node_{i}" for i in range(3)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=10, memory_depolar_rate=0.001,
                            node_distance=1, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                              num_runs=1,
                                                              memory_depolar_rate=0.001,
                                                              node_distance=1,
                                                              max_entangle_pairs=9,
                                                              target_fidelity=0.995,
                                                              qubits_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results.columns)
    print(results)


def run_test_example_with_verification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(3)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=3, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=2, memory_depolar_rate=100,
                                                              node_distance=3,
                                                              max_entangle_pairs=10, target_fidelity=0.995, m_size=3,
                                                              batch_size=4,
                                                              skip_noise=True, qubit_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results.columns)
    print(results)


def run_multi_node_purification(max_node, qubit_number=1):
    final_data = {}
    os.makedirs("./transportation_results", exist_ok=True)
    for node_count in range(3, max_node + 1):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "end-to-end-transportation",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=3, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                  num_runs=1000,
                                                                  memory_depolar_rate=100,
                                                                  node_distance=3,
                                                                  max_entangle_pairs=2,
                                                                  target_fidelity=0.995,
                                                                  qubits_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_count}->{c}: {collected_data[c].mean()}")
        final_data[node_count] = node_data
        transport_example.stop()
        ns.sim_reset()
    with open(f"./transportation_results/e2e_transport_{max_node}.json", "w") as f:
        json.dump(final_data, f)


def run_multi_node_purification_distance(max_distance, qubit_number=1):
    final_data = {}
    os.makedirs("./transportation_results", exist_ok=True)
    for node_distance in range(1, max_distance):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "end-to-end-transportation",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=node_distance, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                  num_runs=1000,
                                                                  memory_depolar_rate=100,
                                                                  node_distance=node_distance,
                                                                  max_entangle_pairs=2,
                                                                  target_fidelity=0.995,
                                                                  qubits_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_distance}->{c}: {collected_data[c].mean()}")
        final_data[node_distance] = node_data
        transport_example.stop()
        ns.sim_reset()
    with open(f"./transportation_results/e2e_transport_{max_distance}_km_5_nodes.json", "w") as f:
        json.dump(final_data, f)


def run_5_node_e2e_purification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(5)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=10, memory_depolar_rate=63109,
                            node_distance=1, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                              num_runs=1000,
                                                              memory_depolar_rate=63109,
                                                              node_distance=1,
                                                              max_entangle_pairs=9,
                                                              target_fidelity=0.98,
                                                              qubits_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    collected_data = dc.dataframe
    node_data = {}
    collected_data.to_json(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_raw.json")
    for c in collected_data.columns:
        if c == "teleport_fids":
            s = []
            for t in collected_data[c]:
                s += t
            node_data[c] = np.mean(s)
        else:
            node_data[c] = collected_data[c].mean()
        # if c not in node_data:
        #     node_data[c] = []
        # node_data[c].append(collected_data[c].mean())
        if len(collected_data[c]) < 1000:
            print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
        print(f"5 Node ->{c}: {collected_data[c]}")
    with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification.json", "w") as f:
        json.dump(node_data, f)


def run_5_node_e2e_purification_distance(qubit_number=1, max_dis=10):
    final_data = {}
    final_data_raw = {}
    for d in range(1, max_dis + 1):
        print(f"Running {d} / {max_dis} km")
        final_data[d] = {}
        final_data_raw[d] = {}
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=10, memory_depolar_rate=63109,
                                node_distance=d, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                  num_runs=1000,
                                                                  memory_depolar_rate=63109,
                                                                  node_distance=d,
                                                                  max_entangle_pairs=9,
                                                                  target_fidelity=0.98,
                                                                  qubits_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        node_data = {}
        final_data_raw[d] = collected_data.to_dict()
        for c in collected_data.columns:
            if c == "teleport_fids":
                s = []
                for t in collected_data[c]:
                    s += t
                node_data[c] = np.mean(s)
            else:
                node_data[c] = collected_data[c].mean()
            # if c not in node_data:
            #     node_data[c] = []
            # node_data[c].append(collected_data[c].mean())
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
        final_data[d] = node_data
        print(f"Run {d} km, res {node_data}")
        with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_raw_{max_dis}km.json",
                  "w") as f:
            json.dump(final_data_raw, f)
        with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_{max_dis}km.json", "w") as f:
            json.dump(final_data, f)
        ns.set_random_state(rng=np.random.RandomState())
        ns.sim_stop()
        ns.sim_reset()
        gc.collect()


def run_5_node_e2e_purification_node(qubit_number=1, max_node=10):
    final_data = {}
    final_data_raw = {}
    if os.path.exists(
            f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_raw_{max_node}_node.json"):
        with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_raw_{max_node}_node.json",
                  "r") as f:
            final_data_raw = json.load(f)
    if os.path.exists(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_{max_node}_node.json"):
        with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_{max_node}_node.json",
                  "r") as f:
            final_data = json.load(f)
    for node in range(3, max_node):
        print(f"Running {node} / {max_node} Node")
        if str(node) in final_data_raw:
            print(f"Node {node} already exists, skipping")
            continue
        final_data[node] = {}
        final_data_raw[node] = {}
        node_data = {"teleport_fids": []}
        while len(node_data["teleport_fids"]) < 1000:
            try:
                nodes_list = [f"Node_{j}" for j in range(node)]
                network = setup_network(nodes_list, "hop-by-hop-transportation",
                                        memory_capacity=10, memory_depolar_rate=63109,
                                        node_distance=1, source_delay=1)
                # create a protocol to entangle two nodes
                sample_nodes = [node for node in network.nodes.values()]
                transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                          num_runs=1,
                                                                          memory_depolar_rate=63109,
                                                                          node_distance=1,
                                                                          max_entangle_pairs=9,
                                                                          target_fidelity=0.98,
                                                                          qubits_to_transport=qubit_number)
                # Run the simulation
                transport_example.start()
                ns.sim_run()
                # Collect the data
                collected_data = dc.dataframe
                # final_data_raw[node] = collected_data.to_dict()
                for c in collected_data.columns:
                    if c not in node_data:
                        node_data[c] = []
                    if c == "teleport_fids":
                        s = []
                        for t in collected_data[c]:
                            s += t
                        node_data[c].append(np.mean(s))
                    else:
                        node_data[c].append(collected_data[c].mean())
                    # if c not in node_data:
                    #     node_data[c] = []
                    # node_data[c].append(collected_data[c].mean())
                    # if len(collected_data[c]) < 1000:
                    #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
                print(f"Run {node} node, {len(node_data['teleport_fids'])}/1000")
                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
            except Exception as e:
                print(e)
                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
        # final_data[node] = node_data
        final_data_raw[node] = node_data
        node_data_calculated = {k: np.mean(v) for k, v in node_data.items()}
        final_data[node] = node_data_calculated
        print(f"Run {node} node, res {node_data_calculated}")
        with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_raw_{max_node}_node.json",
                  "w") as f:
            json.dump(final_data_raw, f)
        with open(f"./transportation_results/e2e_5nodes_{qubit_number}_qubit_purification_{max_node}_node.json",
                  "w") as f:
            json.dump(final_data, f)

def run_e2e_purification_target_node(qubit_number=1, target_node=4):
    final_data_raw = {}
    run_count = 0
    if os.path.exists(
            f"./transportation_results/e2e_{target_node}nodes_{qubit_number}_qubit_purification_raw.json"):
        with open(f"./transportation_results/e2e_{target_node}nodes_{qubit_number}_qubit_purification_raw.json",
                  "r") as f:
            final_data_raw = json.load(f)
            run_count = len(final_data_raw['teleport_fids'])
    # if os.path.exists(f"./transportation_results/e2e_{target_node}nodes_{qubit_number}_qubit_purification.json"):
    #     with open(f"./transportation_results/e2e_{target_node}nodes_{qubit_number}_qubit_purification.json",
    #               "r") as f:
    #         final_data = json.load(f)

    while run_count < 1000:
        try:
            nodes_list = [f"Node_{j}" for j in range(target_node)]
            network = setup_network(nodes_list, "hop-by-hop-transportation",
                                    memory_capacity=10, memory_depolar_rate=63109,
                                    node_distance=1, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                      num_runs=1,
                                                                      memory_depolar_rate=63109,
                                                                      node_distance=1,
                                                                      max_entangle_pairs=10,
                                                                      target_fidelity=0.98,
                                                                      qubits_to_transport=qubit_number)
            # Run the simulation
            transport_example.start()
            ns.sim_run()
            # Collect the data
            collected_data = dc.dataframe
            # final_data_raw[node] = collected_data.to_dict()
            for c in collected_data.columns:
                if c not in final_data_raw:
                    final_data_raw[c] = []
                if c == "teleport_fids":
                    s = []
                    for t in collected_data[c]:
                        s += t
                    final_data_raw[c].append(np.mean(s))
                else:
                    final_data_raw[c].append(collected_data[c].mean())
                # if c not in node_data:
                #     node_data[c] = []
                # node_data[c].append(collected_data[c].mean())
                # if len(collected_data[c]) < 1000:
                #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"Run {target_node} node, {run_count}/1000")
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            run_count += 1
            # final_data[node] = node_data
            node_data_calculated = {k: np.mean(v) for k, v in final_data_raw.items()}
            with open(f"./transportation_results/e2e_{target_node}nodes_{qubit_number}_qubit_purification_raw.json",
                      "w") as f:
                json.dump(final_data_raw, f)
            with open(f"./transportation_results/e2e_{target_node}nodes_{qubit_number}_qubit_purification.json",
                      "w") as f:
                json.dump(node_data_calculated, f)
        except Exception as e:
            print(e)
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()



def run_5_node_e2e_purification_throughput(qubit_number=1000):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    if os.path.exists("./transportation_results/e2e_5nodes_throughput_raw.json"):
        with open(f"./transportation_results/e2e_5nodes_throughput_raw.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for key, val in final_data_raw.items():
                print(f"Loading {key}")
                total_count.append(val['total_count'])
                success_count.append(val['teleport_success_count'])
                average_fids.append(val['average_fidelity'])
            start = int(key) + 1
            print(f"Starting preload index {start}")
    else:
        start = 0
    for i in range(start, 1000):
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "e2e-transportation",
                                memory_capacity=1500, memory_depolar_rate=63109,
                                node_distance=1, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes,
                                                                             num_runs=1,
                                                                             memory_depolar_rate=63109,
                                                                             node_distance=1,
                                                                             max_entangle_pairs=1500,
                                                                             target_fidelity=0.98,
                                                                             qubits_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run(duration=1e9)
        # Collect the data
        collected_data = dc.dataframe
        final_row = collected_data.tail(1)
        node_data = {
            "total_count": 0,
            "teleport_success_count": 0,
            "average_fidelity": 0,
        }
        all_fid = []
        for c in final_row.columns:
            data = final_row[c].iloc[0]
            for fid in data.values():
                all_fid.append(fid)
                if fid > 0.99:
                    node_data["teleport_success_count"] += 1
                node_data["total_count"] += 1
        node_data["average_fidelity"] = np.mean(all_fid)
        final_data_raw[i] = node_data
        print(f"Finished {i}/1000\n{node_data}")
        total_count.append(node_data["total_count"])
        average_fids.append(node_data["average_fidelity"])
        success_count.append(node_data["teleport_success_count"])

        transport_example.stop()
        ns.set_random_state(rng=np.random.RandomState())
        ns.sim_stop()
        ns.sim_reset()
        gc.collect()
        final_data["total_count"] = np.mean(total_count)
        final_data["average_fidelity"] = np.mean(average_fids)
        final_data["teleport_success_count"] = np.mean(success_count)

        with open(f"./transportation_results/e2e_5nodes_throughput_raw.json", "w") as f:
            json.dump(final_data_raw, f)

        with open(f"./transportation_results/e2e_5nodes_throughput.json", "w") as f:
            json.dump(final_data, f)

    print(f"Final Data: {final_data}")


def run_5_node_e2e_purification_throughput_distance(qubit_number=1000, max_dis=10):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    if os.path.exists(f"./transportation_results/e2e_5nodes_throughput_raw_{max_dis}_km.json"):
        with open(f"./transportation_results/e2e_5nodes_throughput_raw_{max_dis}_km.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for dis, data in final_data_raw.items():
                if len(data) < 1000:
                    start_dis = int(dis)
                    for key, val in data.items():
                        print(f"Loading {key}")
                        total_count.append(val['total_count'])
                        success_count.append(val['teleport_success_count'])
                        average_fids.append(val['average_fidelity'])
                    start = int(key) + 1
                    print(f"Starting preload dis {dis} km and run {start}")
    else:
        start_dis = 1
        start = 0
    for d in range(start_dis, max_dis + 1):
        final_data_raw[d] = {}
        final_data[d] = {}
        for i in range(start, 1000):
            print(f"Starting preload dis {d} km and run {i}")
            nodes_list = [f"Node_{j}" for j in range(5)]
            network = setup_network(nodes_list, "e2e-transportation",
                                    memory_capacity=1500, memory_depolar_rate=63109,
                                    node_distance=d, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes,
                                                                                 num_runs=d,
                                                                                 memory_depolar_rate=63109,
                                                                                 node_distance=1,
                                                                                 max_entangle_pairs=1500,
                                                                                 target_fidelity=0.98,
                                                                                 qubits_to_transport=qubit_number)
            # Run the simulation
            transport_example.start()
            ns.sim_run(duration=1e9)
            # Collect the data
            collected_data = dc.dataframe
            final_row = collected_data.tail(1)
            node_data = {
                "total_count": 0,
                "teleport_success_count": 0,
                "average_fidelity": 0,
            }
            all_fid = []
            for c in final_row.columns:
                data = final_row[c].iloc[0]
                for fid in data.values():
                    all_fid.append(fid)
                    if fid > 0.99:
                        node_data["teleport_success_count"] += 1
                    node_data["total_count"] += 1
            node_data["average_fidelity"] = np.mean(all_fid)
            final_data_raw[d][i] = node_data
            print(f"Finished {i}/1000\n{node_data}")
            total_count.append(node_data["total_count"])
            average_fids.append(node_data["average_fidelity"])
            success_count.append(node_data["teleport_success_count"])

            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            final_data[d]["total_count"] = np.mean(total_count)
            final_data[d]["average_fidelity"] = np.mean(average_fids)
            final_data[d]["teleport_success_count"] = np.mean(success_count)

            with open(f"./transportation_results/e2e_5nodes_throughput_raw_{max_dis}_km.json", "w") as f:
                json.dump(final_data_raw, f)

            with open(f"./transportation_results/e2e_5nodes_throughput_{max_dis}_km.json", "w") as f:
                json.dump(final_data, f)

    print(f"Final Data: {final_data}")


def run_5_node_e2e_purification_throughput_node(qubit_number=1000, max_node=10):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    start = 0
    if os.path.exists(f"./transportation_results/e2e_5nodes_throughput_raw_{max_node}_node.json"):
        with open(f"./transportation_results/e2e_5nodes_throughput_raw_{max_node}_node.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for node, data in final_data_raw.items():
                if len(data) < 1000:
                    start_node = int(node)
                    for key, val in data.items():
                        print(f"Loading {key}")
                        total_count.append(val['total_count'])
                        success_count.append(val['teleport_success_count'])
                        average_fids.append(val['average_fidelity'])
                    start = int(key) + 1
                    print(f"Starting preload dis {node} node and run {start}")
    else:
        start_node = 3
        start = 0
    for node in range(start_node, max_node):
        if str(node) not in final_data_raw:
            final_data_raw[str(node)] = {}
        final_data[str(node)] = {}
        if start != 0:
            run_count = start
            start = 0
        else:
            run_count = 0
        while run_count < 1000:
            try:
                print(f"Starting {node} node and run {run_count}/1000")
                nodes_list = [f"Node_{j}" for j in range(node)]
                network = setup_network(nodes_list, "e2e-transportation",
                                        memory_capacity=1500, memory_depolar_rate=63109,
                                        node_distance=1, source_delay=1)
                # create a protocol to entangle two nodes
                sample_nodes = [node for node in network.nodes.values()]
                transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes,
                                                                                     num_runs=1,
                                                                                     memory_depolar_rate=63109,
                                                                                     node_distance=1,
                                                                                     max_entangle_pairs=1500,
                                                                                     target_fidelity=0.98,
                                                                                     qubits_to_transport=qubit_number)
                # Run the simulation
                transport_example.start()
                ns.sim_run(duration=1e9)
                # Collect the data
                collected_data = dc.dataframe
                final_row = collected_data.tail(1)
                node_data = {
                    "total_count": 0,
                    "teleport_success_count": 0,
                    "average_fidelity": 0,
                }
                all_fid = []
                for c in final_row.columns:
                    data = final_row[c].iloc[0]
                    for fid in data.values():
                        all_fid.append(fid)
                        if fid > 0.99:
                            node_data["teleport_success_count"] += 1
                        node_data["total_count"] += 1
                node_data["average_fidelity"] = np.mean(all_fid)
                final_data_raw[str(node)][run_count] = node_data
                print(f"Finished {run_count}/1000\n{node_data}")
                total_count.append(node_data["total_count"])
                average_fids.append(node_data["average_fidelity"])
                success_count.append(node_data["teleport_success_count"])

                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
                final_data[str(node)]["total_count"] = np.mean(total_count)
                final_data[str(node)]["average_fidelity"] = np.mean(average_fids)
                final_data[str(node)]["teleport_success_count"] = np.mean(success_count)

                with open(f"./transportation_results/e2e_5nodes_throughput_raw_{max_node}_node.json", "w") as f:
                    json.dump(final_data_raw, f)

                with open(f"./transportation_results/e2e_5nodes_throughput_{max_node}_node.json", "w") as f:
                    json.dump(final_data, f)
                run_count += 1
            except Exception as e:
                print(f"error: {e}")
                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()

    print(f"Final Data: {final_data}")

def run_e2e_purification_throughput_target_node(qubit_number=1000, node_count=4):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    run_count = 0
    if os.path.exists(f"./transportation_results/e2e_{node_count}nodes_purification_throughput_raw.json"):
        with open(f"./transportation_results/e2e_{node_count}nodes_purification_throughput_raw.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for node, data in final_data_raw.items():
                if len(data) < 1000:
                    for key, val in data.items():
                        print(f"Loading {key}")
                        total_count.append(val['total_count'])
                        success_count.append(val['teleport_success_count'])
                        average_fids.append(val['average_fidelity'])
                    run_count = int(key) + 1
                    print(f"Starting preload dis {node} node and run {run_count}")

    if str(node_count) not in final_data_raw:
        final_data_raw[str(node_count)] = {}
    final_data[str(node_count)] = {}

    while run_count < 1000:
        try:
            print(f"Starting {node_count} node and run {run_count}/1000")
            nodes_list = [f"Node_{j}" for j in range(node_count)]
            network = setup_network(nodes_list, "e2e-transportation",
                                    memory_capacity=1500, memory_depolar_rate=63109,
                                    node_distance=1, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes,
                                                                                 num_runs=1,
                                                                                 memory_depolar_rate=63109,
                                                                                 node_distance=1,
                                                                                 max_entangle_pairs=1500,
                                                                                 target_fidelity=0.98,
                                                                                 qubits_to_transport=qubit_number)
            # Run the simulation
            transport_example.start()
            ns.sim_run(duration=1e9)
            # Collect the data
            collected_data = dc.dataframe
            final_row = collected_data.tail(1)
            node_data = {
                "total_count": 0,
                "teleport_success_count": 0,
                "average_fidelity": 0,
            }
            all_fid = []
            for c in final_row.columns:
                data = final_row[c].iloc[0]
                for fid in data.values():
                    all_fid.append(fid)
                    if fid > 0.99:
                        node_data["teleport_success_count"] += 1
                    node_data["total_count"] += 1
            node_data["average_fidelity"] = np.mean(all_fid)
            node_data["all_fid"] = all_fid
            final_data_raw[str(node_count)][run_count] = node_data
            print(f"Finished {run_count}/1000\n{node_data}")
            total_count.append(node_data["total_count"])
            average_fids.append(node_data["average_fidelity"])
            success_count.append(node_data["teleport_success_count"])

            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            final_data[str(node_count)]["total_count"] = np.mean(total_count)
            final_data[str(node_count)]["average_fidelity"] = np.mean(average_fids)
            final_data[str(node_count)]["teleport_success_count"] = np.mean(success_count)

            with open(f"./transportation_results/e2e_{node_count}nodes_purification_throughput_raw.json", "w") as f:
                json.dump(final_data_raw, f)

            with open(f"./transportation_results/e2e_{node_count}nodes_purification_throughput.json", "w") as f:
                json.dump(final_data, f)
            run_count += 1
        except Exception as e:
            print(f"error: {e}")
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()

    print(f"Final Data: {final_data}")


def run_e2e_verification_throughput(qubit_number=1000, node_count=4, distance=1.0):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    start = 0
    CU_matrix = controlled_unitary(4)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj
    # CU_matrix = None
    # CU_gate = None
    # CCU_gate = None
    if os.path.exists(f"./transportation_results/e2e_{node_count}nodes_throughput_{distance}km_verify_raw.json"):
        with open(f"./transportation_results/e2e_{node_count}nodes_throughput_{distance}km_verify_raw.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            data = final_data_raw[str(node_count)]
            if len(data) < 1000:
                for key, val in data.items():
                    print(f"Loading {key}")
                    total_count.append(val['total_count'])
                    success_count.append(val['teleport_success_count'])
                    average_fids.append(val['average_fidelity'])
                start = int(key) + 1
                print(f"Starting preload {node_count} node and run {start}")
    else:
        start = 0
    if start != 0:
        run_count = start
        start = 0
    else:
        run_count = 0
    if str(node_count) not in final_data_raw:
        final_data_raw[str(node_count)] = {}
    final_data[str(node_count)] = {}
    while run_count < 1000:
        try:
            print(f"Starting {node_count} node and run {run_count}/1000")
            nodes_list = [f"Node_{j}" for j in range(node_count)]
            network = setup_network(nodes_list, "e2e-transportation",
                                    memory_capacity=1500, memory_depolar_rate=63109,
                                    node_distance=distance, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_verification_throughput(sample_nodes,
                                                                                 num_runs=1,
                                                                                 memory_depolar_rate=63109,
                                                                                 node_distance=distance,
                                                                                 max_entangle_pairs=1500,
                                                                                 target_fidelity=0.98,
                                                                                 qubit_to_transport=qubit_number,
                                                                                 CU_gate=CCU_gate,
                                                                                 CCU_gate=CCU_gate,
                                                                                 batch_size=4,
                                                                                 m_size=3)

            # Run the simulation
            transport_example.start()
            ns.sim_run(duration=1e9)
            # Collect the data
            collected_data = dc.dataframe
            final_row = collected_data.tail(1)
            node_data = {
                "total_count": 0,
                "teleport_success_count": 0,
                "average_fidelity": 0,
            }
            all_fid = []
            for c in final_row.columns:
                data = final_row[c].iloc[0]
                for fid in data.values():
                    all_fid.append(fid)
                    if fid > 0.99:
                        node_data["teleport_success_count"] += 1
                    node_data["total_count"] += 1
            # if len(all_fid) > 0:
            #     node_data["average_fidelity"] = np.mean(all_fid)
            # else:
            #     node_data["average_fidelity"] = 0.0
            node_data["average_fidelity"] = np.mean(all_fid)
            node_data["all_fids"] = all_fid
            final_data_raw[str(node_count)][str(run_count)] = node_data
            print(f"Finished {run_count}/1000\n{node_data}")
            total_count.append(node_data["total_count"])
            average_fids.append(node_data["average_fidelity"])
            success_count.append(node_data["teleport_success_count"])

            transport_example.stop()
            transport_example = None
            gc.collect()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            final_data[str(node_count)]["total_count"] = np.mean(total_count)
            final_data[str(node_count)]["average_fidelity"] = np.mean(average_fids)
            final_data[str(node_count)]["teleport_success_count"] = np.mean(success_count)

            with open(f"./transportation_results/e2e_{node_count}nodes_throughput_{distance}km_verify_raw.json", "w") as f:
                json.dump(final_data_raw, f)

            with open(f"./transportation_results/e2e_{node_count}nodes_{distance}km_verify_throughput.json", "w") as f:
                json.dump(final_data, f)
            run_count += 1
        except Exception as e:
            print(f"error: {e}")
            traceback.print_exc()
            transport_example.stop()
            transport_example = None
            gc.collect()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()

    print(f"Final Data: {final_data}")


def run_4_node_e2e_verification(qubit_number=1, node_count=3, distance=1.0):
    run_count = 0
    node_data = {}
    CU_matrix = controlled_unitary(4)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj
    while run_count < 1000:
        try:
            print(f"Start {run_count}/1000")
            nodes_list = [f"Node_{i}" for i in range(node_count)]
            network = setup_network(nodes_list, "hop-by-hop-transportation",
                                    memory_capacity=1000, memory_depolar_rate=63109,
                                    node_distance=distance, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_verification(sample_nodes,
                                                                      num_runs=1,
                                                                      memory_depolar_rate=63109,
                                                                      node_distance=distance,
                                                                      max_entangle_pairs=1000,
                                                                      target_fidelity=0.98,
                                                                      qubit_to_transport=qubit_number,
                                                                      m_size=3,
                                                                      batch_size=4,
                                                                      CU_gate=CCU_gate,
                                                                      CCU_gate=CCU_gate,
                                                                      )
            # Run the simulation
            transport_example.start()
            ns.sim_run()
            # Collect the data
            collected_data = dc.dataframe
            # collected_data.to_json(f"./transportation_results/e2e_4nodes_{qubit_number}_qubit_verification_raw.json")
            for c in collected_data.columns:
                if c not in node_data:
                    node_data[c] = []
                if c == "teleport_fids":
                    s = []
                    for t in collected_data[c]:
                        s += t
                    node_data[c].append(np.mean(s))
                else:
                    node_data[c].append(collected_data[c].mean())
                # if c not in node_data:
                #     node_data[c] = []
                # node_data[c].append(collected_data[c].mean())
                # if len(collected_data[c]) < 1000:
                #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
                # print(f"5 Node ->{c}: {node_data[c]}")
            with open(f"./transportation_results/e2e_{node_count}nodes_{qubit_number}_qubit_verification_{distance}km_raw.json",
                      "w") as f:
                json.dump(node_data, f)
            final_data = {}
            for k, v in node_data.items():
                final_data[k] = np.mean(v)
                print(f"{node_count}Node -> {k}: {final_data[k]}")
            with open(f"./transportation_results/e2e_{node_count}nodes_{qubit_number}_qubit_verification_{distance}km.json",
                      'w') as f:
                json.dump(final_data, f)
            print(f"Finished {run_count}/1000")

            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            run_count += 1
        except Exception as e:
            print(f"error: {e}")
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()


def run_5_node_e2e_purification_new(qubit_number=1):
    for i in range(1000):
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=10, memory_depolar_rate=63109,
                                node_distance=1, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                  num_runs=1,
                                                                  memory_depolar_rate=63109,
                                                                  node_distance=1,
                                                                  max_entangle_pairs=10,
                                                                  target_fidelity=0.98,
                                                                  qubits_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        node_data = {}
        for c in collected_data.columns:
            if c == "teleport_fids":
                s = []
                for t in collected_data[c]:
                    s += t
                node_data[c] = np.mean(s)
                print(f"5 Node ->{c}: {node_data[c]}")
            else:
                node_data[c] = collected_data[c].mean()
            # if c not in node_data:
            #     node_data[c] = []
            # node_data[c].append(collected_data[c].mean())
            # if len(collected_data[c]) < 1000:
            #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            # print(f"5 Node ->{c}: {node_data[c]}")
        print("Stopping example")
        transport_example.stop()
        print("Stopping ns")
        ns.sim_stop()
        ns.set_random_state(rng=np.random.RandomState())
        print("Reseting network")
        ns.sim_reset()


if __name__ == '__main__':
    # seed = np.random.randint(0, 10000)
    # seed = 3020
    # np.random.seed(seed)
    # print(f'seed {seed}')
    # run_test_example_with_purification()
    # run_5_node_e2e_purification(1)
    # run_5_node_e2e_purification_throughput(1000)
    # run_5_node_e2e_purification_node(qubit_number=1, max_node=10)
    # exit()
    # run_4_node_e2e_verification(qubit_number=1, node_count=4)
    if len(sys.argv) == 2:
        opt = int(sys.argv[1])
        if opt == 1:
            # run_5_node_e2e_purification_distance(qubit_number=1, max_dis=10)
            run_e2e_purification_throughput_target_node(qubit_number=1200, node_count=4)
        elif opt == 2:
            run_e2e_purification_throughput_target_node(qubit_number=1200, node_count=3)
        elif opt == 3:
            run_5_node_e2e_purification_node(qubit_number=1, max_node=10)
        elif opt == 4:
            run_5_node_e2e_purification_throughput_node(qubit_number=1000, max_node=10)
        elif opt == 5:
            run_4_node_e2e_verification(qubit_number=1, node_count=3)
        elif opt == 6:
            run_4_node_e2e_verification(qubit_number=1, node_count=4)
        elif opt == 7:
            run_e2e_verification_throughput(qubit_number=1500, node_count=3, distance=1)
        elif opt == 8:
            run_e2e_verification_throughput(qubit_number=1500, node_count=4, distance=1)
        elif opt == 9:
            run_e2e_verification_throughput(qubit_number=1500, node_count=4, distance=0.5)
        elif opt == 10:
            run_4_node_e2e_verification(qubit_number=1, node_count=4, distance=0.5)
        elif opt == 11:
            run_4_node_e2e_verification(qubit_number=1, node_count=3, distance=0.5)
    else:
        print("Usage: python sim_end_to_end_transport.py opt")
    # run_4_node_e2e_verification(1)
    # run_5_node_e2e_purification_new(1)
    # run_test_example_with_verification()
    # run_multi_node(11)
    # run_multi_node_purification_distance(11)
