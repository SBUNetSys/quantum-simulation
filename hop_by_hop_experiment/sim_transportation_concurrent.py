import gc
import json
import os

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
from utils.NetworkSetup import setup_network_parallel
from utils import Logging
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent
from protocols.GenEntanglementConcurrent import GenEntanglementConcurrent
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
                 CCU_gate=None,
                 with_purify=True):
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger
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
                                                 is_top_layer=False,
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                if with_purify:
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

                else:
                    # no purification
                    verify_protocol = Verification(node=node,
                                                   name=f"verify_{node.name}->{network_nodes[index - 1].name}",
                                                   entangled_node=network_nodes[index - 1].name,
                                                   purification_protocol=eh_handler,
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger
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
                                                 is_top_layer=False,
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                if with_purify:
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
                else:
                    # case no purification
                    verify_protocol = Verification(node=node,
                                                   name=f"verify_{node.name}->{network_nodes[index + 1].name}",
                                                   entangled_node=network_nodes[index + 1].name,
                                                   purification_protocol=eh_handler,
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
            # p_done = False
            # # print(f"Start Stop Purification of run index {index}")
            # start_end_time = sim_time()
            # while not p_done:
            #     yield self.await_timer(1000)
            #     all_done = True
            #     for subprotocol_name, subprotocol in self.subprotocols.items():
            #         if "purify" in subprotocol_name:
            #             if subprotocol.is_running:
            #                 all_done = False
            #     if all_done:
            #         p_done = True
            #     if sim_time() - start_end_time > 100000:
            #         break
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
                 CCU_gate = None,
                 with_purification=True,):
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger
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
                                                 is_top_layer=False,
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                if with_purification:
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
                else:
                    verify_protocol = Verification(node=node,
                                                   name=f"verify_{node.name}->{network_nodes[index - 1].name}",
                                                   entangled_node=network_nodes[index - 1].name,
                                                   purification_protocol=eh_handler,
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger
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
                                                 is_top_layer=False,
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                if with_purification:
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
                else:
                    verify_protocol = Verification(node=node,
                                                   name=f"verify_{node.name}->{network_nodes[index + 1].name}",
                                                   entangled_node=network_nodes[index + 1].name,
                                                   purification_protocol=eh_handler,
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger
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
                gen_protocol = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger
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
                gen_protocol = GenEntanglementConcurrent(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger
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
                                      qubit_to_transport,
                                      CU_gate,
                                      CCU_gate,
                                      with_purification,
                                      is_throughput,
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
    :param CU_gate: CU gate for verification
    :param CCU_gate: CCU gate for verification
    :param with_purification: skip purification or not
    :param is_throughput: is throughput or not
    :param skip_noise: skip noise when popping qubits
    :return:
    """
    # Create the protocol
    if is_throughput:
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
                                                                CCU_gate=CCU_gate,
                                                                with_purification=with_purification, )
    else:
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
                                                             CCU_gate=CCU_gate,
                                                             with_purify=with_purification,)

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



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def simulate_transport_with_verification_increasing_distance(max_dis, max_node, depolar_rate,
                                                             with_purify, batch_size=4, is_throughput=False,
                                                             preload=False):
    # save path
    if not is_throughput:
        save_path = (f"./transportation_results/"
                     f"concurrent_transport_verification_{max_node}_nodes_{max_dis}km_purify_{with_purify}_1_qubit.json")
    else:
        save_path = (f"./transportation_results/"
                     f"concurrent_transport_verification_{max_node}_nodes_{max_dis}km_purify_{with_purify}_throughput.json")
    if preload:
        if os.path.exists(save_path):
            with open(save_path,"r") as f:
                experiment_result = json.load(f)
        else:
            experiment_result = {}
    else:
        experiment_result = {}

    target_fid_dic = {
        "500": [0.99],
        "1000": [0.98],
        "1500": [0.97],
        "2000": [0.95],
        "2500": [0.94],
        "3000": [0.92],
        "3500": [0.91],
        "4000": [0.90],
        "4500": [0.88],
        "5000": [0.87],
    }

    # start running
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=int(max_dis) * 4)

        dis_m = int(max_dis * 1000)
        for node_dis in range(500, dis_m + 1, 500):
            if str(node_dis) in experiment_result:
                print(f"Skipping distance: {node_dis}, loaded from file")
                progress.update(task, advance=1)
                continue
            experiment_result[str(node_dis)] = {}
            for target_fid in target_fid_dic[str(node_dis)]:
                experiment_result[str(node_dis)][target_fid] = run_transport_sim(node_dis / 1000,
                                                                                 target_fid,
                                                                                 depolar_rate,
                                                                                 max_node,
                                                                                 batch_size,
                                                                                 with_purify,
                                                                                 is_throughput,
                                                                                 with_verification=True)
            with open(save_path, "w") as f:
                json.dump(experiment_result, f, indent=4, cls=NumpyEncoder)
            progress.update(task, advance=1)




def run_transport_sim(distance, target_fid, depolar_rate, node_count,batch_size,
                      with_purify, is_throughput,
                      with_verification):
    CU_matrix = controlled_unitary(batch_size)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj

    # non throughput mode we care only one qubit to transmitted
    if is_throughput:
        qubit_number = 1500
    else:
        qubit_number = 1
    nodes_list = [f"Node_{i}" for i in range(node_count)]
    network = setup_network_parallel(nodes_list, "hop-by-hop-purification",
                                     memory_capacity=1501, memory_depolar_rate=depolar_rate,
                                     node_distance=distance)
    node_data = {}
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    if with_verification:
        transport_example, dc = example_sim_run_with_verification(sample_nodes,
                                                                  num_runs=2,
                                                                  memory_depolar_rate=depolar_rate,
                                                                  node_distance=distance,
                                                                  max_entangle_pairs=1500,
                                                                  target_fidelity=target_fid,
                                                                  skip_noise=True,
                                                                  qubit_to_transport=qubit_number,
                                                                  m_size=3,
                                                                  batch_size=4,
                                                                  CU_gate=CU_gate,
                                                                  CCU_gate=CCU_gate,
                                                                  with_purification=with_purify,
                                                                  is_throughput=is_throughput,)
    else:
        # TODO add purification part
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
    ns.sim_run()
    # Collect the data
    collected_data = dc.dataframe
    # collected_data.to_json(f"./transportation_results/4nodes_{qubit_number}_qubit_verification_raw.json")
    # raw_data = collected_data.to_dict()
    for c in collected_data.columns:
        if c == "teleport_fids":
            node_data[c] = []
            for v in collected_data[c].values:
                node_data[c] += list(v)
        else:
            node_data[c] = list(collected_data[c].values)
        # if c not in node_data:
        #     node_data[c] = []
        # if c == "teleport_fids":
        #     s = []
        #     for t in collected_data[c]:
        #         s += t
        #     node_data[c].append(np.mean(s))
        # else:
        #     node_data[c].append(collected_data[c].mean())
        # if c not in node_data:
        #     node_data[c] = []
        # node_data[c].append(collected_data[c].values)
    transport_example.stop()
    # ns.set_random_state(rng=np.random.RandomState())
    print(f"Teleportation Done\n"
          f"\tVerification {with_verification}\n"
          f"\tTransmitted Qubits {qubit_number}\n"
          f"\tWith Purify {with_purify}\n"
          f"\tIs Throughput {is_throughput}\n"
          f"\tResult")
    for k, v in node_data.items():
        print(f"\t\t{k}: {np.mean(v)}")

    ns.sim_reset()
    transport_example = None
    gc.collect()
    return node_data

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
    # max_dis, max_node, depolar_rate, with_purify
    simulate_transport_with_verification_increasing_distance(max_dis=0.5,
                                                             max_node=3,
                                                             depolar_rate=24583,
                                                             with_purify=True,
                                                             is_throughput=False
                                                             )