import copy
import gc
import json
import operator
import os
import sys
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network, setup_network_parallel
from utils import Logging
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent
from protocols.GenEntanglementConcurrent import GenEntanglementConcurrent
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
                 max_verify_pair=1,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 m_size=3,
                 batch_size=10,
                 skip_noise=False):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.max_verify_pair = max_verify_pair
        self.m_size = m_size
        self.batch_size = batch_size
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleVerification")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False, save_to_file=False, file_name="v.log")
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
                                               logger=self.logger,
                                               is_top_layer=True,
                                               max_verify_pairs=self.max_verify_pair,
                                               )
                self.add_subprotocol(verify_protocol)

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
                                               logger=self.logger,
                                               is_top_layer=True,
                                               max_verify_pairs=self.max_verify_pair,
                                               )
                self.add_subprotocol(verify_protocol)

    def run(self):
        self.start_subprotocols()
        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            verify_protocols = []
            for subprotocol in self.subprotocols.values():
                if "verify" in subprotocol.name:
                    verify_protocols.append(subprotocol)

            wait_signals = [self.await_signal(p, MessageType.VERIFICATION_FINISHED)
                            for p in verify_protocols]

            yield reduce(operator.and_, wait_signals)
            end_time = sim_time()
            results = [p.get_signal_result(MessageType.VERIFICATION_FINISHED)
                       for p in verify_protocols]
            """
            result = {batch_id: [mem_pos1, mem_pos2, ...]}
            """
            result_dic = {}
            node_index = 0
            all_success_teleported = True
            for i in range(0, len(results), 2):
                entangle_node = self.all_nodes[node_index + 1].name
                node = self.all_nodes[node_index].name
                success_batch = results[i]["verification_batches"]
                success_probability = results[i]["verification_probability"]
                total_verification = results[i]["verification_total_count"]
                success_verification = results[i]["verification_success_count"]
                node_pair_res = []
                for res in success_batch.values():
                    node_pair_res += res
                # measure the actual fidelity
                actual_fidelities = {}
                all_teleport_fid = []
                total_batch = len(node_pair_res)
                teleport_success_count = 0
                for mem_pos in node_pair_res:
                    qubit_a = self.all_nodes[node_index].subcomponents[f"{entangle_node}_qmemory"].pop(
                        mem_pos, skip_noise=self.skip_noise)[0]
                    qubit_b = self.all_nodes[node_index + 1].subcomponents[f"{node}_qmemory"].pop(
                        mem_pos, skip_noise=self.skip_noise)[0]
                    q_a_name = str(qubit_a.name).split("#")[-1].split("-")[0]
                    q_b_name = str(qubit_b.name).split("#")[-1].split("-")[0]
                    # print(f"Qubit names: {q_a_name}, {q_b_name}")
                    if q_a_name != q_b_name:
                        raise ValueError(f"Qubit names are not the same at {mem_pos}: {q_a_name}, {q_b_name}")
                    if qubit_a.qstate != qubit_b.qstate:
                        raise ValueError(f"Qubit states are not the same: {qubit_a.qstate}, {qubit_b.qstate}")
                    f = qapi.fidelity([qubit_a, qubit_b], ks.b00)
                    # if 0.01 < f < 0.99:
                    #     # raise ValueError(f"Fidelity is not correct: {f}, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}")
                    #     self.logger.error(f"Fidelity is not correct: {f}, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}",
                    #                       color="red")
                    if not isinstance(f, float) or str(f) == "nan":
                        f = 0
                        self.logger.error(f"Fidelity is Nan, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}",
                                          color="red")
                    actual_fidelities[mem_pos] = f
                    # start_teleportation, generate a qubit for teleportation
                    # rotate the qubit to y0 state
                    qubit = qapi.create_qubits(1)[0]
                    qapi.operate(qubit, ops.H)
                    qapi.operate(qubit, ops.S)
                    # teleport the qubit
                    fid = self.test_teleportation(qubit_a, qubit_b, qubit)
                    all_teleport_fid.append(fid)
                    # print(f"Teleportation fidelity: {fid}")
                    if fid > 0.99:
                        teleport_success_count += 1
                    else:
                        all_success_teleported = False
                result_dic[f"{node}->{entangle_node}"] = {
                    "total_verified_paris": total_batch,
                    "actual_fidelities": actual_fidelities,
                    "success_verification_probability": success_probability,
                    "total_verification_count": total_verification,
                    "success_verification_count": success_verification,
                    "teleport_success_count": teleport_success_count,
                    "end_to_end_success": all_success_teleported,
                    "duration": end_time - start_time,
                    "teleport_fids": all_teleport_fid,
                }
                node_index += 1
            # we need to do safety layer to make sure we have gracefully shutdown the subprotocols
            # for subprotocol_name, subprotocol in self.subprotocols.items():
            #     if "purify" in subprotocol_name and subprotocol.is_running:
            #         yield self.await_signal(subprotocol, MessageType.PURIFICATION_FINISHED)

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})

            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

        # remove any gates after finish running
        # for subprotocol in self.subprotocols.values():
        #     if "verify" in subprotocol.name:
        #         subprotocol.clean_gates()

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

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()


class VerifyExampleNoPurify(LocalProtocol):
    """
    Protocol for a complete verification example.
    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 max_verify_pair=1,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 m_size=3,
                 batch_size=10,
                 skip_noise=False):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.max_verify_pair = max_verify_pair
        self.m_size = m_size
        self.batch_size = batch_size
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleVerification")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False, save_to_file=True, file_name="v.log")
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
                                               logger=self.logger,
                                               is_top_layer=True,
                                               max_verify_pairs=self.max_verify_pair,
                                               )
                self.add_subprotocol(verify_protocol)

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
                                               logger=self.logger,
                                               is_top_layer=True,
                                               max_verify_pairs=self.max_verify_pair,
                                               )
                self.add_subprotocol(verify_protocol)

    def run(self):
        self.start_subprotocols()
        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            verify_protocols = []
            for subprotocol in self.subprotocols.values():
                if "verify" in subprotocol.name:
                    verify_protocols.append(subprotocol)

            wait_signals = [self.await_signal(p, MessageType.VERIFICATION_FINISHED)
                            for p in verify_protocols]

            yield reduce(operator.and_, wait_signals)
            end_time = sim_time()
            results = [p.get_signal_result(MessageType.VERIFICATION_FINISHED)
                       for p in verify_protocols]
            """
            result = {batch_id: [mem_pos1, mem_pos2, ...]}
            """
            result_dic = {}
            node_index = 0
            all_success_teleported = True
            for i in range(0, len(results), 2):
                entangle_node = self.all_nodes[node_index + 1].name
                node = self.all_nodes[node_index].name
                success_batch = results[i]["verification_batches"]
                success_probability = results[i]["verification_probability"]
                total_verification = results[i]["verification_total_count"]
                success_verification = results[i]["verification_success_count"]
                node_pair_res = []
                for res in success_batch.values():
                    node_pair_res += res
                # measure the actual fidelity
                actual_fidelities = {}
                all_teleport_fid = []
                total_batch = len(node_pair_res)
                teleport_success_count = 0
                for mem_pos in node_pair_res:
                    qubit_a = self.all_nodes[node_index].subcomponents[f"{entangle_node}_qmemory"].pop(
                        mem_pos, skip_noise=self.skip_noise)[0]
                    qubit_b = self.all_nodes[node_index + 1].subcomponents[f"{node}_qmemory"].pop(
                        mem_pos, skip_noise=self.skip_noise)[0]
                    q_a_name = str(qubit_a.name).split("#")[-1].split("-")[0]
                    q_b_name = str(qubit_b.name).split("#")[-1].split("-")[0]
                    # print(f"Qubit names: {q_a_name}, {q_b_name}")
                    if q_a_name != q_b_name:
                        raise ValueError(f"Qubit names are not the same at {mem_pos}: {q_a_name}, {q_b_name}")
                    if qubit_a.qstate != qubit_b.qstate:
                        raise ValueError(f"Qubit states are not the same: {qubit_a.qstate}, {qubit_b.qstate}")
                    f = qapi.fidelity([qubit_a, qubit_b], ks.b00)
                    # if 0.01 < f < 0.99:
                    #     # raise ValueError(f"Fidelity is not correct: {f}, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}")
                    #     self.logger.error(f"Fidelity is not correct: {f}, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}",
                    #                       color="red")
                    if not isinstance(f, float) or str(f) == "nan":
                        f = 0
                        self.logger.error(f"Fidelity is Nan, \n\t{qubit_a.qstate}\n\t{qubit_b.qstate}",
                                          color="red")
                    actual_fidelities[mem_pos] = f
                    # start_teleportation, generate a qubit for teleportation
                    # rotate the qubit to y0 state
                    qubit = qapi.create_qubits(1)[0]
                    qapi.operate(qubit, ops.H)
                    qapi.operate(qubit, ops.S)
                    # teleport the qubit
                    fid = self.test_teleportation(qubit_a, qubit_b, qubit)
                    all_teleport_fid.append(fid)
                    # print(f"Teleportation fidelity: {fid}")
                    if fid > 0.99:
                        teleport_success_count += 1
                    else:
                        all_success_teleported = False
                result_dic[f"{node}->{entangle_node}"] = {
                    "total_verified_paris": total_batch,
                    "actual_fidelities": actual_fidelities,
                    "success_verification_probability": success_probability,
                    "total_verification_count": total_verification,
                    "success_verification_count": success_verification,
                    "teleport_success_count": teleport_success_count,
                    "end_to_end_success": all_success_teleported,
                    "duration": end_time - start_time,
                    "teleport_fids": all_teleport_fid,
                }
                node_index += 1
            # we need to do safety layer to make sure we have gracefully shutdown the subprotocols
            # for subprotocol_name, subprotocol in self.subprotocols.items():
            #     if "purify" in subprotocol_name and subprotocol.is_running:
            #         yield self.await_signal(subprotocol, MessageType.PURIFICATION_FINISHED)

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})

            for subprotocol in self.subprotocols.values():
                subprotocol.reset()
        # remove any gates after finish running
        # for subprotocol in self.subprotocols.values():
        #     if "verify" in subprotocol.name:
        #         subprotocol.clean_gates()

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

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()


def example_sim_run(nodes, num_runs, memory_depolar_rate,
                    node_distance, max_entangle_pairs,
                    max_verify_pairs,
                    target_fidelity, m_size, batch_size,
                    skip_noise=False,
                    skip_purify=False):
    """
    Run the example verification protocol
    :param max_verify_pairs: max verify paris to pefrom
    :param nodes: list of nodes
    :param num_runs: number of runs
    :param memory_depolar_rate: memory depolar rate
    :param node_distance: node distance
    :param max_entangle_pairs: maximum entangle pairs
    :param target_fidelity: target fidelity
    :param m_size: m size
    :param batch_size: batch size
    :param skip_noise: skip noise when popping qubits
    :param skip_purify: skip purifying qubits
    :return:
    """
    # Create the protocol
    if skip_purify:
        verify_example = VerifyExampleNoPurify(network_nodes=nodes,
                                               num_runs=num_runs,
                                               max_entangle_pairs=max_entangle_pairs,
                                               memory_depolar_rate=memory_depolar_rate,
                                               node_distance=node_distance,
                                               target_fidelity=target_fidelity,
                                               m_size=m_size,
                                               max_verify_pair=max_verify_pairs,
                                               batch_size=batch_size,
                                               skip_noise=skip_noise)
    else:
        verify_example = VerifyExample(network_nodes=nodes,
                                       num_runs=num_runs,
                                       max_entangle_pairs=max_entangle_pairs,
                                       memory_depolar_rate=memory_depolar_rate,
                                       node_distance=node_distance,
                                       target_fidelity=target_fidelity,
                                       m_size=m_size,
                                       max_verify_pair=max_verify_pairs,
                                       batch_size=batch_size,
                                       skip_noise=skip_noise)

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # print(f"Purification Run {result['run_index']} completed: {result}")
        print(f"Verification protocol result: {result['run_index']}/1000")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=verify_example,
                                     event_type=Signals.SUCCESS.value))
    return verify_example, dc


def run_experiment(nodes_count):
    nodes_list = [f"Node_{i}" for i in range(nodes_count)]
    network = setup_network(nodes_list, "hop-by-hop-verification",
                            memory_capacity=16, memory_depolar_rate=10e-6,
                            node_distance=20, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    verify_example, dc = example_sim_run(sample_nodes, num_runs=2, memory_depolar_rate=10e-6, node_distance=20,
                                         max_entangle_pairs=16, max_verify_pairs=max, target_fidelity=0.995, m_size=3,
                                         batch_size=8)
    # Run the simulation
    verify_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results)


def run_verification_experiment_increasing_distance(max_distance, skip_purify=False, preload=False):
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
    if preload:
        if skip_purify:
            if os.path.exists(
                    f"./verification_results/concurrent_verification_result_2_nodes_1_paris_{max_distance}km_no_verify.json"):
                with open(
                        f"./verification_results/concurrent_verification_result_2_nodes_1_paris_{max_distance}km_no_verify.json",
                        'r') as f:
                    experiment_result = json.load(f)
            else:
                experiment_result = {}
        else:
            if os.path.exists(
                    f"./verification_results/concurrent_verification_result_2_nodes_1_paris_{max_distance}km.json"):
                with open(
                        f"./verification_results/concurrent_verification_result_2_nodes_1_paris_{max_distance}km.json",
                        'r') as f:
                    experiment_result = json.load(f)
            else:
                experiment_result = {}
    else:
        experiment_result = {}
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
    with Progress(TextColumn("[progress.description]{task.description}"),
                  BarColumn(),
                  TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                  TextColumn("[progress.completed]{task.completed}/{task.total}"),
                  TimeRemainingColumn(),
                  transient=True) as progress:
        task = progress.add_task("[green]Paris...", total=int(max_distance) * 4)
        dis_m = int(max_distance * 1000)
        for node_dis in range(500, dis_m + 1, 500):
            if str(node_dis) in experiment_result:
                print(f"Skipping distance: {node_dis}, loaded from file")
                progress.update(task, advance=1)
                continue
            experiment_result[str(node_dis)] = {}
            for target_fid in target_fid_dic[str(node_dis)]:
                experiment_result[str(node_dis)][target_fid] = run_verification(node_dis / 1000, target_fid,
                                                                                skip_purify)
            if skip_purify:
                with open(
                        f"./verification_results/concurrent_verification_result_2_nodes_1_paris_{max_distance}km_no_verify.json",
                        "w") as f:
                    json.dump(experiment_result, f, indent=4)
            else:
                with open(
                        f"./verification_results/concurrent_verification_result_2_nodes_1_paris_{max_distance}km.json",
                        "w") as f:
                    json.dump(experiment_result, f, indent=4)
            progress.update(task, advance=1)


def run_verification(distance, target_fid, skip_purify):
    all_actual_fidelities = []
    all_total_verified_pairs = []
    all_success_verification_probability = []
    all_total_verification_count = []
    all_success_verification_count = []
    all_teleport_success_count = []
    all_teleport_fids = []
    all_duration = []
    for i in range(10):
        nodes_list = [f"Node_{i}" for i in range(2)]
        network = setup_network_parallel(nodes_list, "hop-by-hop-purification",
                                         memory_capacity=100, memory_depolar_rate=24583,
                                         node_distance=distance)
        sample_nodes = [node for node in network.nodes.values()]
        verify_example, dc = example_sim_run(sample_nodes, num_runs=100, memory_depolar_rate=24583,
                                             node_distance=distance,
                                             max_entangle_pairs=100, max_verify_pairs=1, target_fidelity=target_fid,
                                             m_size=3, batch_size=4, skip_noise=True, skip_purify=skip_purify)
        # Run the simulation
        verify_example.start()
        ns.sim_run()
        # Collect the data
        results = dc.dataframe

        for column in dc.dataframe.columns:
            flattened_actual_fidelities = []
            flattened_total_verified_pairs = []
            flattened_success_verification_probability = []
            flattened_total_verification_count = []
            flattened_success_verification_count = []
            flattened_teleport_success_count = []
            flattened_teleport_fids = []
            flattened_duration = []
            for result in results[column]:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if "actual_fidelities" in key:
                            flattened_actual_fidelities += list(value.values())
                        elif "total_verified_paris" in key:
                            flattened_total_verified_pairs.append(value)
                        elif "success_verification_probability" in key:
                            flattened_success_verification_probability += value
                        elif "total_verification_count" in key:
                            flattened_total_verification_count.append(value)
                        elif "success_verification_count" in key:
                            flattened_success_verification_count.append(value)
                        elif "teleport_success_count" in key:
                            flattened_teleport_success_count.append(value)
                        elif "duration" in key:
                            flattened_duration.append(value)
                        elif "teleport_fids" in key:
                            flattened_teleport_fids += value
            all_actual_fidelities += flattened_actual_fidelities
            all_total_verified_pairs += flattened_total_verified_pairs
            all_success_verification_probability += flattened_success_verification_probability
            all_total_verification_count += flattened_total_verification_count
            all_success_verification_count += flattened_success_verification_count
            all_teleport_success_count += flattened_teleport_success_count
            all_duration += flattened_duration
            all_teleport_fids += flattened_teleport_fids

            verify_example.stop()
            del verify_example
            gc.collect()
            # check the time condition
            # if ns.possible_time_manipulation_accuracy_issue(0, ns.sim_time()):
            # case we have time overflow, reset the simulation and run again with different RNG
            ns.sim_reset()
            new_rng = np.random.RandomState()
            if new_rng == ns.get_random_state():
                raise ValueError("Random state is not resetting")
            ns.set_random_state(rng=new_rng)
    data = {"actual_fidelities":
                np.mean(all_actual_fidelities, dtype=np.float64),
            "total_verified_pairs":
                np.mean(all_total_verified_pairs, dtype=np.float64),
            "success_verification_probability":
                all_success_verification_probability,
            "total_verification_count":
                np.mean(all_total_verification_count, dtype=np.float64),
            "success_verification_count":
                np.mean(all_success_verification_count, dtype=np.float64),
            "teleport_success_count":
                np.mean(all_teleport_success_count, dtype=np.float64),
            "teleport_fids":
                np.mean(all_teleport_fids, dtype=np.float64),
            "duration":
                np.mean(all_duration, dtype=np.float64),
            "raw_data":
                {"actual_fidelities": all_actual_fidelities,
                 "total_verified_pairs": all_total_verified_pairs,
                 "total_verification_count": all_total_verification_count,
                 "success_verification_count": all_success_verification_count,
                 "teleport_success_count": all_teleport_success_count,
                 "duration": all_duration,
                 "teleport_fids": all_teleport_fids}
            }
    print("*" * 50)
    print(f"Batch size: {4}\n"
          f"\tEntangle Pairs: {100}\n"
          f"\tSkip Purify: {skip_purify}\n"
          f"\tDuration: {data['duration']}\n"
          f"\tActual Fidelities: {data['actual_fidelities']}\n"
          f"\tTeleport Fidelities: {data['teleport_fids']}\n"
          f"\tTotal Verified Pairs: {data['total_verified_pairs']}\n"
          f"\tTeleport Success Count: {data['teleport_success_count']}\n"
          f"\tTotal Verification Count: {data['total_verification_count']}\n"
          f"\tSuccess Verification Count: {data['success_verification_count']}\n"
          f"\tSuccess Verification Probability: {data['success_verification_probability']}\n")

    return data


if __name__ == '__main__':
    # run_experiment_with_batch_size4_5nodes_with_distance(25, pop_noise)
    # run_verification_experiment_increasing_distance(5.0, skip_purify=True)
    # run_verification_experiment_increasing_distance(5.0, skip_purify=False)
    run_verification_experiment_increasing_distance(5.0, skip_purify=True)
    # run_verification_experiment_increasing_distance(0.5, skip_purify=True)

    # run_experiment_multi(2, pop_noise, max_batch_size=8, only_max_batch=False)
    # # ns.sim_stop()
    # ns.sim_reset()
    # gc.collect()
    # run_experiment_multi(2, pop_noise, max_batch_size=9, only_max_batch=True)
    # run_experiment(2)
    # 1560165200000
    # 1000000000000
