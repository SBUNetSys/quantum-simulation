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
from utils.NetworkSetup import  setup_network_parallel
from utils import Logging, GenSwappingTree
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent
from protocols.GenEntanglementConcurrent import GenEntanglementConcurrent
from protocols.Purification import Purification
from protocols.EndToEnd import EndToEndProtocol
from protocols.Verification import Verification
from protocols.MultiHopTransport import Transportation
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
                 with_purification=True,
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
                    lower_protocols.append(pure_protocol)
                else:
                    lower_protocols.append(eh_handler)

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
                else:
                    lower_protocols.append(eh_handler)
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
                                    MessageType.MULTI_HOP_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.MULTI_HOP_FINISHED, self)
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
    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()


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
                 qubits_to_transport=1,
                 with_purification=True):
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
                    lower_protocols.append(pure_protocol)
                else:
                    lower_protocols.append(eh_handler)

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
                else:
                    lower_protocols.append(eh_handler)
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
                                    MessageType.MULTI_HOP_SUCCESS)
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.MULTI_HOP_SUCCESS, self)
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
                                    MessageType.MULTI_HOP_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.MULTI_HOP_FINISHED, self)
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
                                    MessageType.MULTI_HOP_SUCCESS)
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.MULTI_HOP_SUCCESS, self)
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
                                      qubits_to_transport,
                                      throughput_mode,
                                      with_purification
                                      ):
    if throughput_mode:
        e2e_example = EndToEndTransportWithPurificationThroughput(network_nodes=nodes,
                                                                  num_runs=num_runs,
                                                                  node_path=[node.name for node in nodes],
                                                                  max_entangle_pairs=max_entangle_pairs,
                                                                  memory_depolar_rate=memory_depolar_rate,
                                                                  node_distance=node_distance,
                                                                  target_fidelity=target_fidelity,
                                                                  qubits_to_transport=qubits_to_transport,
                                                                  with_purification=with_purification)
    else:
        e2e_example = EndToEndTransportWithPurificationExample(network_nodes=nodes,
                                                           num_runs=num_runs,
                                                           node_path=[node.name for node in nodes],
                                                           max_entangle_pairs=max_entangle_pairs,
                                                           memory_depolar_rate=memory_depolar_rate,
                                                           node_distance=node_distance,
                                                           target_fidelity=target_fidelity,
                                                           qubits_to_transport=qubits_to_transport,
                                                               with_purification=with_purification)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run completed: {result['run_index']}, fid{result['results']['teleport_fids']}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=e2e_example, event_type=Signals.SUCCESS.value))
    return e2e_example, dc




def example_sim_run_with_verification(nodes, num_runs, memory_depolar_rate,
                                      node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size,
                                      qubit_to_transport,
                                      skip_noise=True,
                                      CU_gate=None,
                                      CCU_gate=None,
                                      throughput_mode=False
                                      ):
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
    if throughput_mode:
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
                                                                        CU_gate=CU_gate,
                                                                        CCU_gate=CCU_gate)
    else:
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


def run_e2e_experiment(qubit_number=1, node_count=10, throughput_mode=False, with_purification=True,
                       target_fidelity=0.98, with_verification=False, batch_size=4, m_size=3,
                       node_distance=1.0, depolar_rate=63109, preload=False):
    save_file = (f"./transportation_results/e2e_{node_count}nodes_{node_distance}m_{qubit_number}_qubit_{depolar_rate}Hz_"
                 f"purification_{with_purification}_verification_{with_verification}_throughput_{throughput_mode}.json")
    save_file_raw = (
        f"./transportation_results/e2e_{node_count}nodes_{node_distance}m_{qubit_number}_qubit_{depolar_rate}Hz_"
        f"purification_{with_purification}_verification_{with_verification}_throughput_{throughput_mode}_raw.json")
    CU_matrix = None
    CU_gate = None
    CCU_gate = None
    if with_purification and with_verification:
        CU_matrix = controlled_unitary(4)
        CU_gate = ops.Operator("CU_Gate", CU_matrix)
        CCU_gate = CU_gate.conj

    final_data = {}
    final_data_raw = {}
    node_data = {"teleport_fids": []}
    while len(node_data["teleport_fids"]) < 1000:
        try:
            nodes_list = [f"Node_{j}" for j in range(node_count)]
            network = setup_network_parallel(nodes_list, "hop-by-hop-transportation",
                                    memory_capacity=101, memory_depolar_rate=depolar_rate,
                                    node_distance=node_distance)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            if with_verification and with_purification:
                transport_example, dc = example_sim_run_with_verification(sample_nodes,
                                                                          num_runs=1,
                                                                          memory_depolar_rate=depolar_rate,
                                                                          node_distance=node_distance,
                                                                          max_entangle_pairs=100,
                                                                          target_fidelity=target_fidelity,
                                                                          qubit_to_transport=qubit_number,
                                                                          m_size=m_size,
                                                                          batch_size=batch_size,
                                                                          CU_gate=CCU_gate,
                                                                          CCU_gate=CCU_gate,
                                                                          throughput_mode=throughput_mode
                                                                          )
            else:
                transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                          num_runs=1,
                                                                          memory_depolar_rate=depolar_rate,
                                                                          node_distance=node_distance,
                                                                          max_entangle_pairs=100,
                                                                          target_fidelity=target_fidelity,
                                                                          qubits_to_transport=qubit_number,
                                                                          with_purification=True,
                                                                          throughput_mode=throughput_mode
                                                                          )
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
            print(f"Run {node_count} node, {len(node_data['teleport_fids'])}/1000")
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
        except Exception as e:
            traceback.print_exc()
            print(e)
            transport_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            ns.sim_reset()
            # final_data[node] = node_data
    final_data_raw = node_data
    node_data_calculated = {k: np.mean(v) for k, v in node_data.items()}
    final_data = node_data_calculated
    # save data
    with open(save_file_raw, "w") as f:
        json.dump(final_data_raw, f)
    with open(save_file, "w") as f:
        json.dump(final_data, f)
    print(f"Run {node_count} node, res {node_data_calculated}")
    return final_data, final_data_raw

def e2e_increase_distance_experiment(max_distances, experiment_name, node_count, depolar_rate, with_purification,
                                     with_verification, throughput_mode, preload=False):
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
    save_path = (f"./e2e_increase_distance_{experiment_name}km_{node_count}nodes_"
                 f"purification_{with_purification}_verification_{with_verification}_"
                 f"throughput_{throughput_mode}.json")
    experiment_data = {}
    if preload:
        with open(save_path, "r") as f:
            experiment_data = json.load(f)
    # start running
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=len(max_distances))

        for distance in max_distances:
            distance = int(distance * 1000)
            if str(distance) in experiment_data:
                print(f"Skip distance {distance}")
                progress.update(task, advance=4)
                continue
            experiment_data[str(distance)] = {}
            target_fidelity = target_fid_dic[str(distance)][0]
            qubit_number = 1
            res, res_raw = run_e2e_experiment(qubit_number=qubit_number,
                                              node_count=node_count,
                                              throughput_mode=throughput_mode,
                                              with_purification=with_purification,
                                              target_fidelity=target_fidelity,
                                              with_verification=with_verification,
                                              node_distance=distance/1000,
                                              depolar_rate=depolar_rate,
                                              preload=preload)
            experiment_data[str(distance)] = res
            # save after each distance
            with open(save_path, "w") as f:
                json.dump(experiment_data, f)
            progress.update(task, advance=1)
    # save data
    with open(save_path, "w") as f:
        json.dump(experiment_data, f)

if __name__ == '__main__':
    # seed = np.random.randint(0, 10000)
    # seed = 3020
    # np.random.seed(seed)
    # print(f'seed {seed}')
    e2e_increase_distance_experiment(max_distances=[1.5], experiment_name="max 1.5", node_count=3, depolar_rate=6000,
                                     with_purification=True, with_verification=False,
                                 throughput_mode=False, preload=False)
    pass