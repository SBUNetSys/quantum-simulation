import gc
import json
import os
from collections import defaultdict
from weakref import finalize

import numpy as np
import pydynaa as pd
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.qubits import ketstates as ks
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utils.SignalMessages import ProtocolFinishedSignalMessage


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
                 skip_noise=False,
                 CU_gate=None,
                 CCU_gate=None,):
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
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.skip_noise = skip_noise

        # Initialize the controlled unitary matrix and measurement operators
        # CU_matrix = controlled_unitary(batch_size)
        measurement_m0, measurement_m1 = measure_operator()
        # CU_gate = ops.Operator("CU_Gate", CU_matrix)
        # CCU_gate = CU_gate.conj

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

            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(
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
                # # get the qubit
                # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                #     mem_pos, skip_noise=True)[0]
                # # measure the state
                # fidelity = qapi.fidelity(qubit, ns.y0)
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
                result_dic['teleport_fids'].append(fid)
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
            for subprotocol_name, subprotocol in self.subprotocols.items():
                if "purify" in subprotocol_name:
                    subprotocol.cc_message_handler.send_signal(MessageType.VERIFICATION_FINISHED,
                                                               ProtocolFinishedSignalMessage(
                                                                   from_protocol=subprotocol,
                                                                   from_node=subprotocol.node.name,
                                                                   entangle_node=subprotocol.entangled_node
                                                               ))

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            p_done = False
            # print(f"Start Stop Purification of run index {index}")
            start_end_time = sim_time()
            while not p_done:
                yield self.await_timer(1000)
                all_done = True
                for subprotocol_name, subprotocol in self.subprotocols.items():
                    if "purify" in subprotocol_name:
                        if subprotocol.is_running:
                            all_done = False
                if all_done:
                    p_done = True
                if sim_time() - start_end_time > 100000:
                    break
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

class TransportWithVerificationThroughput(LocalProtocol):
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
                 skip_noise=False,
                 CU_gate = None,
                 CCU_gate = None,):
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
        self.logger = Logging.Logger(self.name, logging_enabled=False)
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
            while True:
                yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"],
                                        MessageType.TRANSPORT_SUCCESS)
                results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(
                    MessageType.TRANSPORT_SUCCESS, self)
                # print(results)
                self.send_signal(Signals.SUCCESS, results)


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
        self.logger = Logging.Logger(self.name, logging_enabled=False)
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
                                             max_purify_pair=self.max_entangle_pairs,
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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger)
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
        for i in range(self.num_runs):
            # print(f"Run {i}")
            start_time = sim_time()

            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            # yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[0].name}"],
            #                         MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()

            # new logic with
            # p = self.subprotocols[f"transport_{self.all_nodes[-1].name}"]
            # res = p.final_result[p.entangled_node]
            # t_count = 0
            # s_count = 0
            # result_dic = {"teleport_success_count": 0,
            #               "total_count": self.qubits_to_transport,
            #               "teleport_success_rate": 0,
            #               "duration": end_time - start_time,}
            # for mem_pos, fid in res.items():
            #     if fid > 0.99:
            #         result_dic["teleport_success_count"] += 1
            # result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
            # print(f"Success rate: {result_dic['teleport_success_rate']}")
            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_FINISHED, self)
            """
            result = {entangle_node: name, results:{}}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "teleport_fids": [],
                          "duration": end_time - start_time, }
            for mem_pos, fid in results["results"].items():
                result_dic["total_count"] += 1
                # # get the qubit
                # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                #     mem_pos, skip_noise=True)[0]
                # # measure the state
                # fidelity = qapi.fidelity(qubit, ns.y0)
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
                result_dic['teleport_fids'].append(fid)
            # print(f"Values {results['results'].values()}")
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
            # for subprotocol_name, subprotocol in self.subprotocols.items():
            #     if "transport" in subprotocol_name and subprotocol.is_running:
            #         subprotocol.stop()

            for subprotocol_name, subprotocol in self.subprotocols.items():
                if "purify" in subprotocol_name:
                    subprotocol.cc_message_handler.send_signal(MessageType.VERIFICATION_FINISHED,
                                                               ProtocolFinishedSignalMessage(
                                                                   from_protocol=subprotocol,
                                                                   from_node=subprotocol.node.name,
                                                                   entangle_node=subprotocol.entangled_node
                                                               ))
            # print(f"{i}/{self.num_runs} finished")
            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": i})
            # print(f"{i}/{self.num_runs} stopping")
            p_done = False
            p_start = sim_time()
            while not p_done:
                yield self.await_timer(1000)
                all_done = True
                for subprotocol_name, subprotocol in self.subprotocols.items():
                    if "purify" in subprotocol_name:
                        if subprotocol.is_running:
                            all_done = False
                if all_done:
                    p_done = True
                if sim_time() - p_start > 10000:
                    break
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()
            # for subprotocol_name, subprotocol in self.subprotocols.items():
            #     if "transport" not in subprotocol_name:
            #         subprotocol.reset()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()

class TransportWithPurificationThroughput(LocalProtocol):
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
        self.logger = Logging.Logger(self.name, logging_enabled=False)
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
                                             max_purify_pair=self.max_entangle_pairs,
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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger)
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
        while True:
            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_SUCCESS)
            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_SUCCESS, self)
            self.send_signal(Signals.SUCCESS, results)

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
                                      qubit_to_transport, CU_gate, CCU_gate,
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
                                                         qubits_to_transport=qubit_to_transport,
                                                         CU_gate=CU_gate,
                                                         CCU_gate=CCU_gate,)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Verification Run {result['run_index']} completed, fid{result['results']['teleport_fids']}, "
              f"sim_time {sim_time()}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc

def example_sim_run_with_verification_throughput(nodes, num_runs, memory_depolar_rate,
                                      node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size,
                                      qubit_to_transport, CU_gate, CCU_gate,
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
    transport_example = TransportWithVerificationThroughput(network_nodes=nodes,
                                                         num_runs=num_runs,
                                                         max_entangle_pairs=max_entangle_pairs,
                                                         memory_depolar_rate=memory_depolar_rate,
                                                         node_distance=node_distance,
                                                         target_fidelity=target_fidelity,
                                                         m_size=m_size,
                                                         batch_size=batch_size,
                                                         skip_noise=skip_noise,
                                                         qubits_to_transport=qubit_to_transport,
                                                         CU_gate=CU_gate,
                                                         CCU_gate=CCU_gate,)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Transported {len(result['results'])} Qubit")
        return result

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
        # result = protocol.get_signal_result(Signals.FINISHED)
        # print(f"Purification Run {result['run_index']} completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc

def example_sim_run_with_purification_throughput(nodes, num_runs, memory_depolar_rate,
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
    transport_example = TransportWithPurificationThroughput(network_nodes=nodes,
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
        # result = protocol.get_signal_result(Signals.FINISHED)
        # print(f"Purification Run {result['run_index']} completed: {result}")
        print(f"Transported {len(result['results'])} Qubit")
        return result

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc


def run_test_example_with_verification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(5)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=3, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=100, memory_depolar_rate=100,
                                                              node_distance=3,
                                                              max_entangle_pairs=10, target_fidelity=0.995, m_size=3,
                                                              batch_size=10,
                                                              skip_noise=True, qubit_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results.columns)
    print(results)


def run_test_example_with_purification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(5)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=10, memory_depolar_rate=0.001 * 1e9,
                            node_distance=1, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1, memory_depolar_rate=0.001 * 1e9,
                                                              node_distance=1,
                                                              max_entangle_pairs=9, target_fidelity=0.9,
                                                              skip_noise=True, qubit_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    collected_data = dc.dataframe
    # print(results.columns)
    # print(results)
    node_data = {}
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
        print(f"5 Node ->{c}: {node_data[c]}")


def run_multi_node_purification_example(max_node, qubit_number=1):
    os.makedirs("./transportation_results", exist_ok=True)
    final_data = {}
    for node_count in range(3, max_node + 1):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=3, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1000, memory_depolar_rate=100,
                                                                  node_distance=3,
                                                                  max_entangle_pairs=2, target_fidelity=0.995,
                                                                  skip_noise=True, qubit_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            # if c not in node_data:
            #     node_data[c] = []
            # node_data[c].append(collected_data[c].mean())
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_count}->{c}: {collected_data[c].mean()}")
        final_data[node_count] = node_data
        transport_example.stop()
        ns.sim_reset()
        # for i in range(1):
        #     nodes_list = [f"Node_{i}" for i in range(node_count)]
        #     network = setup_network(nodes_list, "hop-by-hop-transportation",
        #                             memory_capacity=128, memory_depolar_rate=100,
        #                             node_distance=3, source_delay=1)
        #     # create a protocol to entangle two nodes
        #     sample_nodes = [node for node in network.nodes.values()]
        #     transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=100, memory_depolar_rate=100,
        #                                                               node_distance=3,
        #                                                               max_entangle_pairs=2, target_fidelity=0.995,
        #                                                               skip_noise=True, qubit_to_transport=qubit_number)
        #     # Run the simulation
        #     transport_example.start()
        #     ns.sim_run()
        #     # Collect the data
        #     collected_data = dc.dataframe
        #     print(collected_data)
        #     for c in collected_data.columns:
        #         # node_data[c] = collected_data[c].mean()
        #         if c not in node_data:
        #             node_data[c] = []
        #         node_data[c].append(collected_data[c].mean())
        #         # if len(collected_data[c]) < 1000:
        #         #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
        #         # print(f"{node_count}->{c}: {collected_data[c].mean()}")
        #     transport_example.stop()
        #     ns.sim_reset()
        #     ns.set_random_state(rng=np.random.RandomState())
        #     gc.collect()
        # data = {k: np.mean(v) for k, v in node_data.items()}
        # print(f"{node_count}: {data}")
        # final_data[node_count] = data
    # final_data = {k: np.mean(list(val)) for k, val in final_data.items()}
    with open(f"./transportation_results/max_{max_node}_nodes_purification.json", "w") as f:
        json.dump(final_data, f)


def run_multi_node_purification_example_distance(max_distance, qubit_number=1):
    os.makedirs("./transportation_results", exist_ok=True)
    final_data = {}
    for node_distance in range(1, max_distance):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=node_distance, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1000, memory_depolar_rate=100,
                                                                  node_distance=node_distance,
                                                                  max_entangle_pairs=2, target_fidelity=0.995,
                                                                  skip_noise=True, qubit_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            # if c not in node_data:
            #     node_data[c] = []
            # node_data[c].append(collected_data[c].mean())
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_distance}->{c}: {collected_data[c].mean()}")
        final_data[node_distance] = node_data
        transport_example.stop()
        ns.sim_reset()
        # for i in range(1):
        #     nodes_list = [f"Node_{i}" for i in range(node_count)]
        #     network = setup_network(nodes_list, "hop-by-hop-transportation",
        #                             memory_capacity=128, memory_depolar_rate=100,
        #                             node_distance=3, source_delay=1)
        #     # create a protocol to entangle two nodes
        #     sample_nodes = [node for node in network.nodes.values()]
        #     transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=100, memory_depolar_rate=100,
        #                                                               node_distance=3,
        #                                                               max_entangle_pairs=2, target_fidelity=0.995,
        #                                                               skip_noise=True, qubit_to_transport=qubit_number)
        #     # Run the simulation
        #     transport_example.start()
        #     ns.sim_run()
        #     # Collect the data
        #     collected_data = dc.dataframe
        #     print(collected_data)
        #     for c in collected_data.columns:
        #         # node_data[c] = collected_data[c].mean()
        #         if c not in node_data:
        #             node_data[c] = []
        #         node_data[c].append(collected_data[c].mean())
        #         # if len(collected_data[c]) < 1000:
        #         #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
        #         # print(f"{node_count}->{c}: {collected_data[c].mean()}")
        #     transport_example.stop()
        #     ns.sim_reset()
        #     ns.set_random_state(rng=np.random.RandomState())
        #     gc.collect()
        # data = {k: np.mean(v) for k, v in node_data.items()}
        # print(f"{node_count}: {data}")
        # final_data[node_count] = data
    # final_data = {k: np.mean(list(val)) for k, val in final_data.items()}
    with open(f"./transportation_results/max_{max_distance}_km_5_nodes_purification.json", "w") as f:
        json.dump(final_data, f)


def run_multi_node_verification_example(max_node, qubit_number=1):
    os.makedirs("./transportation_results", exist_ok=True)
    final_data = {}
    for node_count in range(3, max_node + 1):
        node_data = {}
        # nodes_list = [f"Node_{i}" for i in range(node_count)]
        # network = setup_network(nodes_list, "hop-by-hop-transportation",
        #                         memory_capacity=128, memory_depolar_rate=100,
        #                         node_distance=3, source_delay=1)
        # # create a protocol to entangle two nodes
        # sample_nodes = [node for node in network.nodes.values()]
        # transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=1000, memory_depolar_rate=100,
        #                                                           node_distance=3,
        #                                                           max_entangle_pairs=10, target_fidelity=0.995,
        #                                                           m_size=3, batch_size=4,
        #                                                           skip_noise=True, qubit_to_transport=qubit_number)
        # # Run the simulation
        # transport_example.start()
        # ns.sim_run()
        # # Collect the data
        # collected_data = dc.dataframe
        # for c in collected_data.columns:
        #     node_data[c] = collected_data[c].mean()
        #     # if c not in node_data:
        #     #     node_data[c] = []
        #     # node_data[c].append(collected_data[c].mean())
        #     if len(collected_data[c]) < 1000:
        #         print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
        #     print(f"{node_count}->{c}: {collected_data[c].mean()}")
        # final_data[node_count] = node_data
        # transport_example.stop()
        # ns.sim_reset()
        # gc.collect()
        for i in range(1000):
            print(f"Run {i}/{1000}, {node_count}")
            nodes_list = [f"Node_{i}" for i in range(node_count)]
            network = setup_network(nodes_list, "hop-by-hop-transportation",
                                    memory_capacity=128, memory_depolar_rate=100,
                                    node_distance=3, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1, memory_depolar_rate=100,
                                                                      node_distance=3,
                                                                      max_entangle_pairs=2, target_fidelity=0.995,
                                                                      skip_noise=True, qubit_to_transport=qubit_number)
            # Run the simulation
            transport_example.start()
            ns.sim_run()
            # Collect the data
            collected_data = dc.dataframe
            # print(collected_data)
            for c in collected_data.columns:
                # node_data[c] = collected_data[c].mean()
                if c not in node_data:
                    node_data[c] = []
                node_data[c].append(collected_data[c].mean())
                # if len(collected_data[c]) < 1000:
                #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
                # print(f"{node_count}->{c}: {collected_data[c].mean()}")
            transport_example.stop()
            ns.sim_reset()
            ns.set_random_state(rng=np.random.RandomState())
            gc.collect()
        data = {k: np.mean(v) for k, v in node_data.items()}
        print(f"{node_count}: {data}\n{len(list(node_data.values())[0])}")
        final_data[node_count] = data
    # final_data = {k: np.mean(list(val)) for k, val in final_data.items()}
    with open(f"./transportation_results/max_{max_node}_nodes_verification.json", "w") as f:
        json.dump(final_data, f)


def run_multi_node_verification_example_one_run(max_node, qubit_number=1):
    os.makedirs("./transportation_results", exist_ok=True)
    final_data = {}
    if os.path.exists(f"./transportation_results/max_{max_node}_nodes_verification.json"):
        with open(f"./transportation_results/max_{max_node}_nodes_verification.json", "r") as f:
            final_data = json.load(f)
    for node_count in range(3, max_node + 1):
        if str(node_count) in final_data:
            print(f"Skipping {node_count} / {max_node}, as we already have data.")
            continue
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=3, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=1000, memory_depolar_rate=100,
                                                                  node_distance=3,
                                                                  max_entangle_pairs=25, target_fidelity=0.995,
                                                                  m_size=3, batch_size=4,
                                                                  skip_noise=True, qubit_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            # if c not in node_data:
            #     node_data[c] = []
            # node_data[c].append(collected_data[c].mean())
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_count}->{c}: {collected_data[c].mean()}")
        final_data[node_count] = node_data
        transport_example.stop()
        ns.sim_reset()
        gc.collect()
        # for i in range(1000):
        #     print(f"Run {i}/{1000}, {node_count}")
        #     nodes_list = [f"Node_{i}" for i in range(node_count)]
        #     network = setup_network(nodes_list, "hop-by-hop-transportation",
        #                             memory_capacity=128, memory_depolar_rate=100,
        #                             node_distance=3, source_delay=1)
        #     # create a protocol to entangle two nodes
        #     sample_nodes = [node for node in network.nodes.values()]
        #     transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1, memory_depolar_rate=100,
        #                                                               node_distance=3,
        #                                                               max_entangle_pairs=2, target_fidelity=0.995,
        #                                                               skip_noise=True, qubit_to_transport=qubit_number)
        #     # Run the simulation
        #     transport_example.start()
        #     ns.sim_run()
        #     # Collect the data
        #     collected_data = dc.dataframe
        #     # print(collected_data)
        #     for c in collected_data.columns:
        #         # node_data[c] = collected_data[c].mean()
        #         if c not in node_data:
        #             node_data[c] = []
        #         node_data[c].append(collected_data[c].mean())
        #         # if len(collected_data[c]) < 1000:
        #         #     print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
        #         # print(f"{node_count}->{c}: {collected_data[c].mean()}")
        #     transport_example.stop()
        #     ns.sim_reset()
        #     ns.set_random_state(rng=np.random.RandomState())
        #     gc.collect()
        # data = {k: np.mean(v) for k, v in node_data.items()}
        # print(f"{node_count}: {data}\n{len(list(node_data.values())[0])}")
        # final_data[node_count] = data
        # final_data = {k: np.mean(list(val)) for k, val in final_data.items()}
        with open(f"./transportation_results/max_{max_node}_nodes_verification.json", "w") as f:
            json.dump(final_data, f)


def run_evaluation_5_node(qubit_number=5):
    nodes_list = [f"Node_{i}" for i in range(5)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=10, memory_depolar_rate=63109,
                            node_distance=2, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1000, memory_depolar_rate=63109,
                                                              node_distance=2,
                                                              max_entangle_pairs=9, target_fidelity=0.98,
                                                              skip_noise=True, qubit_to_transport=qubit_number,
                                                              )
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    collected_data = dc.dataframe
    node_data = {}
    collected_data.to_json(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw.json")
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
    with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification.json", "w") as f:
        json.dump(node_data, f)

def run_evaluation_5_node_distance(qubit_number=1, max_dis=10):
    final_data = {}
    final_data_raw = {}
    if os.path.exists(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_{max_dis}km.json"):
        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_{max_dis}km.json", 'r') as f:
            final_data = json.load(f)
    if os.path.exists(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw_{max_dis}km.json"):
        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw_{max_dis}km.json", 'r') as f:
            final_data_raw = json.load(f)
    for d in range(1, max_dis+1):
        print(f"Run Distance: {d} / {max_dis} km")
        if str(d) in final_data_raw:
            print(f"Skipping {d} / {max_dis} km, loaded from file")
            continue
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=10, memory_depolar_rate=63109,
                                node_distance=d, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1000, memory_depolar_rate=63109,
                                                                  node_distance=d,
                                                                  max_entangle_pairs=9, target_fidelity=0.98,
                                                                  skip_noise=True, qubit_to_transport=qubit_number,
                                                                  )
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
            # print(f"5 Node ->{c}: {collected_data[c]}")
        print(f"{d}km result {node_data}")
        final_data[d] = node_data

        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_{max_dis}km.json", "w") as f:
            json.dump(final_data, f)
        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw_{max_dis}km.json", "w") as f:
            json.dump(final_data_raw, f)

def run_evaluation_5_node_node(qubit_number=1, max_node=10):
    final_data = {}
    final_data_raw = {}
    if os.path.exists(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_{max_node}_node.json"):
        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_{max_node}_node.json", "r") as f:
            final_data = json.load(f)
    if os.path.exists(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw_{max_node}_node.json"):
        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw_{max_node}_node.json", "r") as f:
            final_data_raw = json.load(f)
    for d in range(3, max_node):
        print(f"Run node: {d} / {max_node} node")
        if str(d) in final_data_raw:
            print(f"Skipping {d} / {max_node} node, loaded from file")
            continue
        nodes_list = [f"Node_{i}" for i in range(d)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=10, memory_depolar_rate=63109,
                                node_distance=1, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1000, memory_depolar_rate=63109,
                                                                  node_distance=1,
                                                                  max_entangle_pairs=9, target_fidelity=0.98,
                                                                  skip_noise=True, qubit_to_transport=qubit_number,
                                                                  )
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
            # print(f"5 Node ->{c}: {collected_data[c]}")
        print(f"{d} node result {node_data}")
        final_data[d] = node_data

        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_{max_node}_node.json", "w") as f:
            json.dump(final_data, f)
        with open(f"./transportation_results/5nodes_{qubit_number}_qubit_purification_raw_{max_node}_node.json", "w") as f:
            json.dump(final_data_raw, f)

def run_evaluation_target_node(qubit_number=1, target_node=3):
    final_data_raw = {}
    run_count = 0
    # if os.path.exists(f"./transportation_results/{target_node}nodes_{qubit_number}_qubit_purification.json"):
    #     with open(f"./transportation_results/{target_node}nodes_{qubit_number}_qubit_purification.json", "r") as f:
    #         final_data = json.load(f)
    if os.path.exists(f"./transportation_results/{target_node}nodes_{qubit_number}_qubit_purification_raw.json"):
        with open(f"./transportation_results/{target_node}nodes_{qubit_number}_qubit_purification_raw.json", "r") as f:
            final_data_raw = json.load(f)
            run_count = len(final_data_raw['teleport_fids'])
    while run_count < 1000:
        print(f"Run {target_node} node: {run_count} / 1000")
        nodes_list = [f"Node_{i}" for i in range(target_node)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=10, memory_depolar_rate=63109,
                                node_distance=1, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes, num_runs=1, memory_depolar_rate=63109,
                                                                  node_distance=1,
                                                                  max_entangle_pairs=10, target_fidelity=0.98,
                                                                  skip_noise=True, qubit_to_transport=qubit_number,
                                                                  )
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        # node_data = {}
        # final_data_raw= collected_data.to_dict()
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
            # print(f"5 Node ->{c}: {collected_data[c]}")
        final_data = {k:np.mean(v) for k,v in final_data_raw.items()}
        # print(f"{target_node} node result {final_data}")
        transport_example.stop()
        ns.set_random_state(rng=np.random.RandomState())
        ns.sim_reset()

        with open(f"./transportation_results/{target_node}nodes_{qubit_number}_qubit_purification.json", "w") as f:
            json.dump(final_data, f)
        with open(f"./transportation_results/{target_node}nodes_{qubit_number}_qubit_purification_raw.json", "w") as f:
            json.dump(final_data_raw, f)
        run_count += 1

def run_evaluation_5_node_throughput(qubit_number=1000):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    if os.path.exists("./transportation_results/5nodes_throughput_raw.json"):
        with open(f"./transportation_results/5nodes_throughput_raw.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for key, val in final_data_raw.items():
                print(f"Loading {key}")
                total_count.append(val['total_count'])
                success_count.append(val['teleport_success_count'])
                average_fids.append(val['average_fidelity'])
            start = int(key)+1
            print(f"Starting preload index {start}")
    else:
        start = 0
    for i in range(start, 1000):
        nodes_list = [f"Node_{i}" for i in range(5)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=1500, memory_depolar_rate=63109,
                                node_distance=1, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes, num_runs=1, memory_depolar_rate=63109,
                                                                  node_distance=1,
                                                                  max_entangle_pairs=1500, target_fidelity=0.98,
                                                                  skip_noise=True, qubit_to_transport=qubit_number,
                                                                  )
        # Run the simulation
        transport_example.start()
        ns.sim_run(duration=1e9)
        # Collect the data
        collected_data = dc.dataframe
        # print(collected_data)
        final_row = collected_data.tail(1)
        node_data = {
            "total_count" : 0,
            "teleport_success_count": 0,
            "average_fidelity" : 0,
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

        with open(f"./transportation_results/5nodes_throughput_raw.json", "w") as f:
            json.dump(final_data_raw, f)

        with open(f"./transportation_results/5nodes_throughput.json", "w") as f:
            json.dump(final_data, f)

    print(f"Final Data: {final_data}")


def run_evaluation_5_node_throughput_distance(qubit_number=1000, max_dis=10):

    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    start = 0
    if os.path.exists(f"./transportation_results/5nodes_throughput_raw_{max_dis}_km.json"):
        with open(f"./transportation_results/5nodes_throughput_raw_{max_dis}_km.json", "r") as f:
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
                    start = int(key)+1
                    print(f"Starting preload dis {dis} km and run {start}")
    else:
        start_dis = 1
        start = 0
    for d in range(start_dis, max_dis+1):
        if str(d) not in final_data_raw:
            final_data_raw[str(d)] = {}
        final_data[str(d)] = {}
        if start != 0:
            run_count = start
            start = 0
        else:
            run_count = 0
        while run_count < 1000:
            try:
                print(f"Starting dis {d} km and run {run_count}")
                nodes_list = [f"Node_{j}" for j in range(5)]
                network = setup_network(nodes_list, "hop-by-hop-transportation",
                                        memory_capacity=1500, memory_depolar_rate=63109,
                                        node_distance=d, source_delay=1)
                # create a protocol to entangle two nodes
                sample_nodes = [node for node in network.nodes.values()]
                transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes, num_runs=1, memory_depolar_rate=63109,
                                                                          node_distance=d,
                                                                          max_entangle_pairs=1500, target_fidelity=0.98,
                                                                          skip_noise=True, qubit_to_transport=qubit_number,
                                                                          )
                # Run the simulation
                transport_example.start()
                ns.sim_run(duration=1e9)
                # Collect the data
                collected_data = dc.dataframe
                # print(collected_data)
                final_row = collected_data.tail(1)
                node_data = {
                    "total_count" : 0,
                    "teleport_success_count": 0,
                    "average_fidelity" : 0,
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
                final_data_raw[str(d)][run_count] = node_data
                print(f"Finished {run_count}/1000\n{node_data}")
                total_count.append(node_data["total_count"])
                average_fids.append(node_data["average_fidelity"])
                success_count.append(node_data["teleport_success_count"])

                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
                final_data[str(d)]["total_count"] = np.mean(total_count)
                final_data[str(d)]["average_fidelity"] = np.mean(average_fids)
                final_data[str(d)]["teleport_success_count"] = np.mean(success_count)

                with open(f"./transportation_results/5nodes_throughput_raw_{max_dis}_km.json", "w") as f:
                    json.dump(final_data_raw, f)

                with open(f"./transportation_results/5nodes_throughput_{max_dis}_km.json", "w") as f:
                    json.dump(final_data, f)
                run_count += 1
                gc.collect()
            except Exception as e:
                print(e)
                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
                gc.collect()
        # reset
        success_count = []
        total_count = []
        average_fids = []
    print(f"Final Data: {final_data}")


def run_evaluation_5_node_throughput_node(qubit_number=1000, max_node=10):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    start = 0
    if os.path.exists(f"./transportation_results/5nodes_throughput_raw_{max_node}_node.json"):
        with open(f"./transportation_results/5nodes_throughput_raw_{max_node}_node.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for node, data in final_data_raw.items():
                if len(data) < 1000:
                    start_node = int(node)
                    for key, val in data.items():
                        print(f"Loading {key}, {val}")
                        total_count.append(val['total_count'])
                        success_count.append(val['teleport_success_count'])
                        average_fids.append(val['average_fidelity'])
                    start_run = int(key) + 1
                    print(f"Starting preload dis {node} node and run {start_run}")
    else:
        start_node = 3
        start_run = 0
    for d in range(start_node, max_node):
        if str(d) not in final_data_raw:
            final_data_raw[str(d)] = {}
        final_data[str(d)] = {}
        if start!=0:
            run_count = start
        else:
            run_count = 0
        while run_count < 1000:
            try:
                print(f"Starting {d} node and run {run_count}")
                nodes_list = [f"Node_{j}" for j in range(d)]
                network = setup_network(nodes_list, "hop-by-hop-transportation",
                                        memory_capacity=1500, memory_depolar_rate=63109,
                                        node_distance=1, source_delay=1)
                # create a protocol to entangle two nodes
                sample_nodes = [node for node in network.nodes.values()]
                transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes, num_runs=1,
                                                                                     memory_depolar_rate=63109,
                                                                                     node_distance=1,
                                                                                     max_entangle_pairs=1500,
                                                                                     target_fidelity=0.98,
                                                                                     skip_noise=True,
                                                                                     qubit_to_transport=qubit_number,
                                                                                     )
                # Run the simulation
                transport_example.start()
                ns.sim_run(duration=1e9)
                # Collect the data
                collected_data = dc.dataframe
                # print(collected_data)
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
                final_data_raw[str(d)][run_count] = node_data
                print(f"Finished {run_count}/1000\n{node_data}")
                total_count.append(node_data["total_count"])
                average_fids.append(node_data["average_fidelity"])
                success_count.append(node_data["teleport_success_count"])

                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
                final_data[str(d)]["total_count"] = np.mean(total_count)
                final_data[str(d)]["average_fidelity"] = np.mean(average_fids)
                final_data[str(d)]["teleport_success_count"] = np.mean(success_count)

                with open(f"./transportation_results/5nodes_throughput_raw_{max_node}_node.json", "w") as f:
                    json.dump(final_data_raw, f)

                with open(f"./transportation_results/5nodes_throughput_{max_node}_node.json", "w") as f:
                    json.dump(final_data, f)
                run_count += 1
            except Exception as e:
                print(e)
                transport_example.stop()
                ns.set_random_state(rng=np.random.RandomState())
                ns.sim_reset()
        # reset
        success_count = []
        total_count = []
        average_fids = []
    print(f"Final Data: {final_data}")

def run_evaluation_throughput_target_node(qubit_number=1500, target_node=4):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    start = 0
    if os.path.exists(f"./transportation_results/{target_node}nodes_purification_throughput_raw.json"):
        with open(f"./transportation_results/{target_node}nodes_purification_throughput_raw.json", "r") as f:
            final_data_raw = json.load(f)
            # start = list(final_data_raw.keys())[-1]
            # print(f"Loading throughput data from {start}...")
            for node, data in final_data_raw.items():
                if len(data) < 1000:
                    start_node = int(node)
                    for key, val in data.items():
                        print(f"Loading {key}, {val}")
                        total_count.append(val['total_count'])
                        success_count.append(val['teleport_success_count'])
                        average_fids.append(val['average_fidelity'])
                    start_run = int(key) + 1
                    print(f"Starting preload dis {node} node and run {start_run}")

    if str(target_node) not in final_data_raw:
        final_data_raw[str(target_node)] = {}
    final_data[str(target_node)] = {}
    if start!=0:
        run_count = start
    else:
        run_count = 0
    while run_count < 1000:
        try:
            print(f"Starting {target_node} node and run {run_count}")
            nodes_list = [f"Node_{j}" for j in range(target_node)]
            network = setup_network(nodes_list, "hop-by-hop-transportation",
                                    memory_capacity=1500, memory_depolar_rate=63109,
                                    node_distance=1, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_purification_throughput(sample_nodes, num_runs=1,
                                                                                 memory_depolar_rate=63109,
                                                                                 node_distance=1,
                                                                                 max_entangle_pairs=1500,
                                                                                 target_fidelity=0.98,
                                                                                 skip_noise=True,
                                                                                 qubit_to_transport=qubit_number,
                                                                                 )
            # Run the simulation
            transport_example.start()
            ns.sim_run(duration=1e9)
            # Collect the data
            collected_data = dc.dataframe
            # print(collected_data)
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
            final_data_raw[str(target_node)][run_count] = node_data
            print(f"Finished {run_count}/1000")
            total_count.append(node_data["total_count"])
            average_fids.append(node_data["average_fidelity"])
            success_count.append(node_data["teleport_success_count"])

            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            final_data[str(target_node)]["total_count"] = np.mean(total_count)
            final_data[str(target_node)]["average_fidelity"] = np.mean(average_fids)
            final_data[str(target_node)]["teleport_success_count"] = np.mean(success_count)

            with open(f"./transportation_results/{target_node}nodes_purification_throughput_raw.json", "w") as f:
                json.dump(final_data_raw, f)

            with open(f"./transportation_results/{target_node}nodes_purification_throughput.json", "w") as f:
                json.dump(final_data, f)
            run_count += 1
        except Exception as e:
            print(e)
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()

    print(f"Final Data: {final_data}")

# def run_evaluation_4_node_verify(qubit_number=1):
#     nodes_list = [f"Node_{i}" for i in range(4)]
#     network = setup_network(nodes_list, "hop-by-hop-transportation",
#                             memory_capacity=10, memory_depolar_rate=63109,
#                             node_distance=1, source_delay=1)
#     # create a protocol to entangle two nodes
#     sample_nodes = [node for node in network.nodes.values()]
#     transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=1, memory_depolar_rate=63109,
#                                                               node_distance=1,
#                                                               max_entangle_pairs=10, target_fidelity=0.98,
#                                                               skip_noise=True, qubit_to_transport=qubit_number,
#                                                               m_size=3, batch_size=4, )
#     # Run the simulation
#     transport_example.start()
#     ns.sim_run()
#     # Collect the data
#     collected_data = dc.dataframe
#     node_data = {}
#     collected_data.to_json(f"./transportation_results/4nodes_{qubit_number}_qubit_verification_raw.json")
#     for c in collected_data.columns:
#         if c == "teleport_fids":
#             s = []
#             for t in collected_data[c]:
#                 s += t
#             node_data[c] = np.mean(s)
#         else:
#             node_data[c] = collected_data[c].mean()
#         # if c not in node_data:
#         #     node_data[c] = []
#         # node_data[c].append(collected_data[c].mean())
#         if len(collected_data[c]) < 1000:
#             print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
#         print(f"5 Node ->{c}: {collected_data[c]}")
#     with open(f"./transportation_results/4nodes_{qubit_number}_qubit_verification.json", "w") as f:
#         json.dump(node_data, f)

def run_evaluation_4_node_verify_new(qubit_number=1, node_count=3, distance=1.0):
    node_data = {}
    CU_matrix = controlled_unitary(4)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj
    for i in range(1000):
        print(f"Run {i}/1000")
        nodes_list = [f"Node_{j}" for j in range(node_count)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=1500, memory_depolar_rate=63109,
                                node_distance=distance, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=1, memory_depolar_rate=63109,
                                                                  node_distance=distance,
                                                                  max_entangle_pairs=1500, target_fidelity=0.98,
                                                                  skip_noise=True, qubit_to_transport=qubit_number,
                                                                  m_size=3, batch_size=4,
                                                                  CU_gate=CU_gate,
                                                                  CCU_gate=CCU_gate)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        # collected_data.to_json(f"./transportation_results/4nodes_{qubit_number}_qubit_verification_raw.json")
        # raw_data = collected_data.to_dict()
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
        transport_example.stop()
        ns.set_random_state(rng=np.random.RandomState())
        print("Resetting network")
        ns.sim_reset()
        transport_example = None
        gc.collect()
        with open(f"./transportation_results/{node_count}nodes_{qubit_number}_qubit_verification_{distance}km_raw.json", 'w') as f:
            json.dump(node_data, f)
        final_result = {}
        for k, v in node_data.items():
            final_result[k] = np.mean(v)
            print(f"5 Node ->{k}: {final_result[k]}")
        with open(f"./transportation_results/{node_count}nodes_{qubit_number}_qubit_verification_{distance}km.json", "w") as f:
            json.dump(final_result, f)

def run_evaluation_4_node_verify_depolar(qubit_number=1, node_count=3, distance=1.0, rate=24583):
    node_data = {}
    CU_matrix = controlled_unitary(4)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj
    for i in range(1000):
        print(f"Run {i}/1000")
        nodes_list = [f"Node_{j}" for j in range(node_count)]
        network = setup_network(nodes_list, "hop-by-hop-transportation",
                                memory_capacity=1500, memory_depolar_rate=rate,
                                node_distance=distance, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_verification(sample_nodes, num_runs=1, memory_depolar_rate=rate,
                                                                  node_distance=distance,
                                                                  max_entangle_pairs=1500, target_fidelity=0.98,
                                                                  skip_noise=True, qubit_to_transport=qubit_number,
                                                                  m_size=3, batch_size=4,
                                                                  CU_gate=CU_gate,
                                                                  CCU_gate=CCU_gate)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        # collected_data.to_json(f"./transportation_results/4nodes_{qubit_number}_qubit_verification_raw.json")
        # raw_data = collected_data.to_dict()
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
        transport_example.stop()
        ns.set_random_state(rng=np.random.RandomState())
        print("Resetting network")
        ns.sim_reset()
        transport_example = None
        gc.collect()
        with open(f"./transportation_results/{node_count}nodes_{qubit_number}_qubit_verification_{distance}km_{rate}hz_raw.json", 'w') as f:
            json.dump(node_data, f)
        final_result = {}
        for k, v in node_data.items():
            final_result[k] = np.mean(v)
            print(f"5 Node ->{k}: {final_result[k]}")
        with open(f"./transportation_results/{node_count}nodes_{qubit_number}_qubit_verification_{distance}km_{rate}hz.json", "w") as f:
            json.dump(final_result, f)

def run_evaluation_4_node_verify_throughput(qubit_number=1000, node_count=3, batch_size=4, distance=1.0):
    final_data_raw = {}
    final_data = {}
    success_count = []
    total_count = []
    average_fids = []
    CU_matrix = controlled_unitary(batch_size)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj
    run_count = 0
    while run_count < 1000:
        try:
            print(f"Run {run_count}/1000")
            nodes_list = [f"Node_{j}" for j in range(node_count)]
            network = setup_network(nodes_list, "hop-by-hop-verify-transportation",
                                    memory_capacity=1500, memory_depolar_rate=63109,
                                    node_distance=distance, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            transport_example, dc = example_sim_run_with_verification_throughput(sample_nodes, num_runs=1, memory_depolar_rate=63109,
                                                                      node_distance=distance,
                                                                      max_entangle_pairs=1500, target_fidelity=0.98,
                                                                      skip_noise=True, qubit_to_transport=qubit_number,
                                                                      m_size=3, batch_size=batch_size,
                                                                      CU_gate=CU_gate,
                                                                      CCU_gate=CCU_gate)
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
            if len(all_fid)> 0:
                node_data["average_fidelity"] = float(np.mean(all_fid))
            else:
                node_data["average_fidelity"] = 0.0
            node_data["all_fidelity"] = all_fid
            final_data_raw[run_count] = node_data
            print(f"Finished {run_count}/1000\n{node_data}")
            total_count.append(node_data["total_count"])
            average_fids.append(node_data["average_fidelity"])
            success_count.append(node_data["teleport_success_count"])

            transport_example.stop()
            transport_example = None
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()

            final_data["total_count"] = np.mean(total_count)
            final_data["average_fidelity"] = np.mean(average_fids)
            final_data["teleport_success_count"] = np.mean(success_count)

            with open(f"./transportation_results/{node_count}nodes_{distance}km_verification_throughput_raw.json", "w") as f:
                json.dump(final_data_raw, f)

            with open(f"./transportation_results/{node_count}nodes_{distance}km_verification_throughput.json", "w") as f:
                json.dump(final_data, f)
            run_count += 1
        except Exception as e:
            print(e)
            transport_example.stop()
            transport_example = None
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()

if __name__ == '__main__':
    # exit()
    # seed = np.random.randint(0, 10000)
    # seed = 524
    # np.random.seed(seed)
    # print(f'seed {seed}')
    # run_evaluation_5_node(1)
    # run_evaluation_4_node_verify_throughput(qubit_number=1500, node_count=3, batch_size=100)
    # exit()
    # run_evaluation_4_node_verify_new(qubit_number=1, node_count=5)
    # exit()
    # run_evaluation_target_node(target_node=3, qubit_number=1)
    # run_evaluation_target_node(target_node=4, qubit_number=1)

    if len(sys.argv) == 2:
        opt = int(sys.argv[1])
        if opt == 0:
            # run_evaluation_5_node_throughput(1000)
            run_evaluation_4_node_verify_new(qubit_number=1,node_count=5)
        elif opt == 1:
            run_evaluation_4_node_verify_new(qubit_number=1,node_count=3)
        elif opt == 2:
            # run_evaluation_4_node_verify_throughput(1000)
            run_evaluation_4_node_verify_new(qubit_number=1, node_count=4)
        elif opt == 3:
            # run_evaluation_5_node_throughput_distance(qubit_number=1000,max_dis=10)
            run_evaluation_throughput_target_node(qubit_number=1200, target_node=4)
        elif opt == 4:
            # run_evaluation_5_node_throughput_node(qubit_number=1000, max_node=10)
            run_evaluation_throughput_target_node(qubit_number=1200, target_node=3)
        elif opt == 5:
            # run_evaluation_4_node_verify_new(qubit_number=1,node_count=3)
            run_evaluation_4_node_verify_throughput(qubit_number=1500, node_count=3)
        elif opt == 6:
            # run_evaluation_4_node_verify_new(qubit_number=1, node_count=4)
            run_evaluation_4_node_verify_throughput(qubit_number=1500, node_count=4)
        elif opt == 7:
            # run_evaluation_5_node_distance(qubit_number=1, max_dis=10)
            run_evaluation_5_node_node(qubit_number=1, max_node=10)
        elif opt == 8:
            run_evaluation_5_node_throughput_node(qubit_number=1000, max_node=10)
        elif opt == 9:
            run_evaluation_throughput_target_node(qubit_number=1000, target_node=4)
        elif opt == 10:
            run_evaluation_4_node_verify_throughput(qubit_number=1500, node_count=3, batch_size=4, distance=0.5)
        elif opt == 11:
            run_evaluation_4_node_verify_throughput(qubit_number=1500, node_count=4, batch_size=4, distance=0.5)
        elif opt == 12:
            run_evaluation_4_node_verify_new(qubit_number=1, node_count=4, distance=0.5)
        elif opt == 13:
            run_evaluation_4_node_verify_new(qubit_number=1, node_count=3, distance=0.5)
        elif opt == 14:
            run_evaluation_4_node_verify_depolar(qubit_number=1, node_count=4, distance=1.0, rate=24583)
        elif opt == 15:
            run_evaluation_4_node_verify_depolar(qubit_number=1, node_count=4, distance=1.0, rate=6492)
        elif opt == 16:
            run_evaluation_4_node_verify_depolar(qubit_number=1, node_count=3, distance=1.0, rate=24583)
        elif opt == 17:
            run_evaluation_4_node_verify_depolar(qubit_number=1, node_count=3, distance=1.0, rate=6492)
    else:
        print("arg 0 = purification_throughput, 1 = verification")

    # if len(sys.argv) == 2:
    #     run_multi_node_verification_example_one_run(int(sys.argv[1]))
    # else:
    #     # run_test_example_with_purification()
        # run_evaluation_5_node(1)
    #     run_evaluation_4_node_verify_new(1)
    #     # run_evaluation_5_node_throughput(1000)
    #     # run_test_example_with_verification()
    #     # run_multi_node_purification_example(11)
    #     # run_multi_node_verification_example_one_run(3)
    #     # run_multi_node_verification_example(5)
    #     # run_multi_node_purification_example_distance(11)
