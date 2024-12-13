import operator
from functools import reduce

import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.qdetector import defaultdict
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops, ketstates, operators
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.nodes.network import Network
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate, INSTR_Z, INSTR_SWAP, INSTR_H, INSTR_X
from netsquid.components.component import Message, Port
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.qprogram import QuantumProgram
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.components.models import DepolarNoiseModel
from netsquid.nodes.connections import DirectConnection
from pandas.compat import set_function_name
from pydynaa import EventExpression
from netsquid.qubits.qubitapi import measure
from netsquid.qubits.operators import CNOT, Z
from netsquid.components.instructions import INSTR_MEASURE
from netsquid.nodes import Node
from netsquid.qubits.qubitapi import fidelity

from protocols.MessageHandler import MessageType
from utils import Logging
from utils.ClassicalMessages import ClassicalMessage
from utils.SignalMessages import *


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def print_green(text):
    print(f"\033[92m{text}\033[0m")


def print_red(text):
    print(f"\033[91m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_purple(text):
    print(f"\033[95m{text}\033[0m")


def print_orange(text):
    print(f"\033[33m{text}\033[0m")


def print_cyan(text):
    print(f"\033[96m{text}\033[0m")


class SwappingPair:
    def __init__(self, left_node, right_node, left_pos, right_pos, left_fid, right_fid, left_ready, right_ready):
        self.left_node = left_node
        self.right_node = right_node
        self.left_fid = left_fid
        self.right_fid = right_fid
        self.left_ready = left_ready
        self.right_ready = right_ready
        self.left_pos = left_pos
        self.right_pos = right_pos


class EndToEndProtocol(NodeProtocol):
    """
    A protocol that generate end-to-end swapping between two nodes in the network
    Swapping Logic:

    A -> B -> C -> D -> E -> F -> G

    Swapping Tree will be
    Level 0: B, D, F
    A <- B -> C = A -> C, C <- D -> E = C -> E, E <- F -> G = E -> G
    Now we have A -> C -> E -> G

    Level 1: C
    A <- C -> E = A -> E, E <- C -> G = E -> G
    Now we have A -> E -> G

    Level 2: E
    A <- E -> G = A -> G
    Now we have A -> G
    """
    """
        Initialize the protocol
        :param node: the node that the protocol is attached to
        :param name: the name of the protocol
        :param swapping_nodes: the swapping tree that lays out the swapping path
        :param qubit_ready_protocols: the lower layers that will send the qubit ready signal 
        (i.e purfication or verification)
        :param cc_message_handler: the classical message handler
        :param final_entanglement: the final entanglement goal, (source node, target node)
        :param max_pairs: the maximum number of pairs that can be entangled
        :param logger: the logger
        :param is_top_layer: if we are the top layer of the simulation
    """

    def __init__(self, node,
                 name,
                 swapping_nodes,
                 qubit_ready_protocols,
                 cc_message_handler,
                 final_entanglement,
                 max_pairs,
                 logger,
                 is_top_layer=False):

        super().__init__(node=node, name=name)
        # check if we are a swapping node or not.
        # if we are the swapping node, we will perform the swap operation between the left and right neighbors
        self.swapping_node = None
        for node in swapping_nodes:
            if node.parent == self.node.name:
                self.swapping_node = node

        # qubits that are entangled node_name -> {memory_pos: fidelity}
        self.entangled_qubits = defaultdict(dict)
        # keep track of the pending swap operation as a swap node
        self.pending_swap_operation = {}  # key = (left, right, left_pos, right_pos) = SwappingPair
        # keep track of the swap success but wait response from the right node after applying correction
        self.pending_swap_success_confirmation = {}  # key = (left, right, left_pos, right_pos), value = SwappingPair
        # keep track of swap result as a non-swap node
        self.pending_swap_request = {}  # key = (source, target, mem_pos), value = SwapRequestMessage
        # keep track of the entangle node's origin in the stack
        self.swapping_stack = defaultdict(list)  # key = (source, target), value = [intermediate node]
        # qubit input signal from lower layers, can be purification or verification
        await_signals = [self.await_signal(protocol, Signals.SUCCESS) for protocol in qubit_ready_protocols]
        # have expression to wait for ANY qubit input signal
        self.qubit_input_signal = reduce(operator.or_, await_signals)
        # classical message handler
        self.cc_message_handler = cc_message_handler

        # final entanglement
        self.final_entanglement = final_entanglement
        # keep track of the node name and actual memory name
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        self.max_pairs = max_pairs
        self.is_top_layer = is_top_layer

    def add_new_signal(self, signal):
        self.add_signal(signal)

    def get_qmemory(self, memory_name):
        """
        Get the quantum memory of the node
        :param memory_name:
        :return:
        """
        return self.node.subcomponents[memory_name]

    def perform_swap(self, q1_mem_pos, q2_mem_pos, q1_mem_name, q2_mem_name, success_rate=0.5):
        """
        Perform swap measurement on the qubits
        :param q1_mem_pos:
        :param q2_mem_pos:
        :param q1_mem_name:
        :param q2_mem_name:
        :param success_rate:
        :return:
        """
        # Simulating a swap operation with possible failures
        yield self.await_timer(1)  # Simulate some operation time

        # Simulate Bell state measurement
        success_probability = 1.0  # 50% success rate for Bell state measurement
        if np.random.random() > success_probability:
            return False, None, None

        q1_qmemory = self.get_qmemory(q1_mem_name)
        q2_qmemory = self.get_qmemory(q2_mem_name)
        if q1_qmemory.busy:
            yield self.await_program(q1_qmemory)
        # q1, = q1_qmemory.peek(q1_mem_pos)
        # TODO: we do pop with skip_noise=False, noise applied to the qubit
        q1, = q1_qmemory.pop(q1_mem_pos, skip_noise=False)
        if q2_qmemory.busy:
            yield self.await_program(q2_qmemory)
        q2, = q2_qmemory.pop(q2_mem_pos, skip_noise=False)
        # q2, = q2_qmemory.peek(q2_mem_pos)
        # print_red(f"Swap {self.name} -> Fidelity before swap: {fidelity([q1, q2], ks.b00)}")
        # apply CNOT on both qubit
        qapi.operate(qubits=[q1, q2], operator=operators.CNOT)
        # apply Hadamard on q1
        qapi.operate(qubits=[q1], operator=operators.H)
        # measure the qubits so we get m1 and m2
        m1, _ = qapi.measure(q1)
        m2, _ = qapi.measure(q2)
        return True, m1, m2

    def apply_corrections(self, message: SwapApplyCorrectionMessage):
        node_name = self.get_qmemory_from_stack(message.intermediate_node)
        self.logger.info(f"Swap {self.name} -> Apply correction\n"
                         f"Target Node: {message.target_node}\n"
                         f"Source Node: {message.source_node}\n"
                         f"Mem Pos {message.memo_pos}\n"
                         f"Intermediate Node: {message.intermediate_node}\n"
                         f"Qmem_name: {node_name}", color="yellow")

        qmemory = self.get_qmemory(f"{node_name}_qmemory")

        if message.m1 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            self.logger.info(
                f"Sawp {self.name} -> Apply correction Z on qubit {message.memo_pos} with qmem_name: {qmemory.name}",
                color="green")
            qmemory.execute_instruction(INSTR_Z, [message.memo_pos])
        if message.m2 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            self.logger.info(
                f"Sawp {self.name} -> Apply correction X on qubit {message.memo_pos} with qmem_name: {qmemory.name}",
                color="green")
            qmemory.execute_instruction(INSTR_X, [message.memo_pos])
        # update the entangled qubits
        # TODO: what about the fidelity?
        self.entangled_qubits[message.target_node][message.memo_pos] = 1.0
        # update the stack
        self.swapping_stack[(message.source_node, message.target_node)].append(message.intermediate_node)
        # send the success signal to the intermediate node so it can send the success signal to the left node
        self.cc_message_handler.send_message(MessageType.SWAP_APPLY_CORRECTION_SUCCESS,
                                             ClassicalMessage(
                                                 from_node=self.node.name,
                                                 to_node=message.intermediate_node,
                                                 data=SwapApplyCorrectionSuccessMessage(
                                                     operation_key=message.operation_key)
                                             ))

    def get_qmemory_from_stack(self, node_name):
        """
        trace through the swapping stack to get the actual memory name of the node
        :param node_name: entangled node name
        :return:
        """
        edge = (self.node.name, node_name)
        while edge in self.swapping_stack:
            inter_nodes = self.swapping_stack[edge]
            from_node = inter_nodes[0]
            if from_node == self.node.name:
                return edge[1]
            edge = (self.node.name, from_node)
        return None

    def handle_swapping(self, swapping_pair: SwappingPair):
        """
        Handle the swapping operation
        :param swapping_pair: SwappingPair for this swapping operation
        :return:
        """

        left_node = swapping_pair.left_node
        right_node = swapping_pair.right_node
        # get the qubits memory name
        q1 = self.get_qmemory_from_stack(left_node)
        if q1 is None:
            self.logger.error(f"Swap {self.name} -> Qubit memory not found for {left_node}")
            return
        q2 = self.get_qmemory_from_stack(right_node)
        if q2 is None:
            self.logger.error(f"Swap {self.name} -> Qubit memory not found for {right_node}")
            return
        q1_mem_name = f"{q1}_qmemory"
        q2_mem_name = f"{q2}_qmemory"

        q1_mem_pos = self.entangled_qubits[swapping_pair.left_pos]
        q2_mem_pos = self.entangled_qubits[swapping_pair.right_pos]
        self.logger.info(f"Swap {self.name} -> Perform swap\n"
                         f"Between {left_node} and {right_node} "
                         f"Swapping Node {self.node.name}", color="cyan")
        success, m1, m2 = yield from self.perform_swap(q1_mem_pos, q2_mem_pos, q1_mem_name, q2_mem_name)
        if success:
            # case we success the swap
            key_pair = (left_node, right_node, q1_mem_pos, q2_mem_pos)
            self.pending_swap_success_confirmation[key_pair] = swapping_pair
            # send the message to the right node to perform the correction
            self.cc_message_handler.send_message(MessageType.SWAP_APPLY_CORRECTION,
                                                 swapping_pair.right_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.right_node,
                                                     data=SwapApplyCorrectionMessage(
                                                         source_node=swapping_pair.left_node,
                                                         target_node=swapping_pair.right_node,
                                                         intermediate_node=self.node.name,
                                                         operation_key=key_pair,
                                                         memo_pos=swapping_pair.right_pos,
                                                         m1=m1,
                                                         m2=m2)))

        else:
            # case we failed the swap
            # TODO we need to re-entangle the qubits, we need send the information to both left and right
            self.logger.info(f"Swap {self.name} -> Swap failed, re-entangle the qubits", color="red")
            # left node
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.left_node,
                                                     data=SwapFailedMessage(
                                                         source_node=swapping_pair.left_node,
                                                         target_node=swapping_pair.right_node,
                                                         memo_pos=swapping_pair.left_pos)))
            # right node
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.right_node,
                                                     data=SwapFailedMessage(
                                                         source_node=swapping_pair.right_node,
                                                         target_node=swapping_pair.left_node,
                                                         memo_pos=swapping_pair.right_pos)))
            # handle swap failed for our self
            self.handle_swap_failed(SwapFailedMessage(source_node=self.node.name,
                                                      target_node=swapping_pair.left_node,
                                                      memo_pos=swapping_pair.left_pos))
            self.handle_swap_failed(SwapFailedMessage(source_node=self.node.name,
                                                        target_node=swapping_pair.right_node,
                                                        memo_pos=swapping_pair.right_pos))

    def check_swap_condition(self):
        """
        Check if we can perform swap operation
        1. if we are a swap node, we will check if we have left and right qubits ready
            - if we have both qubits ready, we will send the swap request to left and right node
        2. if we are not a swap node, we will check swap node request. If we have the qubit ready, we will send the
        swap ready signal to the swap node

        :return:
        """
        if self.swapping_node is not None:
            # case we are the swap node
            # we check if we have the left and right qubits ready
            if self.swapping_node.left in self.entangled_qubits and self.swapping_node.right in self.entangled_qubits:
                # pop the qubits from the entangled qubits
                left_qubit_pos, left_fid = self.entangled_qubits[self.swapping_node.left].pop(0)
                right_qubit_pos, right_fid = self.entangled_qubits[self.swapping_node.right].pop(0)
                # create a swapping pair
                swapping_pair = SwappingPair(self.swapping_node.left, self.swapping_node.right,
                                             left_qubit_pos, right_qubit_pos, left_fid, right_fid,
                                             False, False)
                # store the pending swap request
                pair_key = (self.swapping_node.left, self.swapping_node.right, left_qubit_pos, right_qubit_pos)
                self.pending_swap_operation[pair_key] = swapping_pair
                # send swap signal to the left and right node
                self.cc_message_handler.send_message(MessageType.SWAP_NEED,
                                                     self.swapping_node.left,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=self.swapping_node.left,
                                                         data=SwapRequestResponseMessage(self.swapping_node.left,
                                                                                         self.swapping_node.right,
                                                                                         self.node.name,
                                                                                         left_qubit_pos,
                                                                                         pair_key)))
                self.cc_message_handler.send_message(MessageType.SWAP_NEED,
                                                     self.swapping_node.right,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=self.swapping_node.right,
                                                         data=SwapRequestResponseMessage(self.swapping_node.right,
                                                                                         self.swapping_node.left,
                                                                                         self.node.name,
                                                                                         right_qubit_pos,
                                                                                         pair_key)))
        else:
            # case we are not the swap node
            # we check if we have the swap request from the swap node
            for key, value in self.pending_swap_request.items():
                value: SwapRequestResponseMessage
                if (value.intermediate_node in self.entangled_qubits and
                        value.memo_pos in self.entangled_qubits[value.intermediate_node]):
                    # TODO send the swap ready signal to the swap node
                    self.cc_message_handler.send_message(MessageType.SWAP_READY,
                                                         ClassicalMessage(
                                                             from_node=self.node.name,
                                                             to_node=value.intermediate_node,
                                                             data=SwapRequestResponseMessage(value.source_node,
                                                                                             value.target_node,
                                                                                             value.intermediate_node,
                                                                                             value.memo_pos,
                                                                                             value.operation_key)))
                    # remove the pending swap result
                    self.pending_swap_request.pop(key)
                    # pop the qubit as it was being used for swapping operation
                    self.entangled_qubits[value.intermediate_node].pop(value.memo_pos)

    def handle_swap_apply_success(self, message: SwapApplyCorrectionSuccessMessage):
        """
        Handle the swap success message,
        1. remove the SwapPair from the pending swap success confirmation
        2. send the swap success message to the left node
        :param message: SwapApplyCorrectionSuccessMessage
        :return:
        """
        if message.operation_key in self.pending_swap_success_confirmation:
            swapping_pair = self.pending_swap_success_confirmation.pop(message.operation_key)
            # send the success signal to the left node
            self.cc_message_handler.send_message(MessageType.SWAP_SUCCESS,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.left_node,
                                                     data=SwapSuccessMessage(
                                                         source_node=swapping_pair.left_node,
                                                         target_node=swapping_pair.right_node,
                                                         intermediate_node=self.node.name,
                                                         memo_pos=swapping_pair.left_pos)))

    def handle_swap_success(self, message: SwapSuccessMessage):
        """
        Handle the swap success message from intermediate node
        1. we update the entangled qubits
        2. we update the swapping stack
        :param message: SwapSuccessMessage
        :return:
        """
        # update the entangled qubits
        # TODO: what about the fidelity?
        self.entangled_qubits[message.target_node][message.memo_pos] = 1.0
        # update the stack
        self.swapping_stack[(message.source_node, message.target_node)].append(message.intermediate_node)

    def handle_swap_need(self, result: SwapRequestResponseMessage):
        """
        Handle the swap need signal from swap node
        We first check if the request node already in entangled qubits
        if we have the qubit, we will send the swap ready signal to the swap node
        otherwise we store in self.pending_swap_result

        :param result: SwapRequestMessage
        :return:
        """
        if result.intermediate_node in self.entangled_qubits and \
                result.memo_pos in self.entangled_qubits[result.intermediate_node]:
            # we have the qubit, send the swap ready signal to the swap node
            self.cc_message_handler.send_message(MessageType.SWAP_READY,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=result.intermediate_node,
                                                     data=SwapRequestResponseMessage(result.source_node,
                                                                                     result.target_node,
                                                                                     result.intermediate_node,
                                                                                     result.memo_pos,
                                                                                     result.operation_key)))
        else:
            self.pending_swap_request[(result.source_node, result.target_node, result.memo_pos)] = result

    def handle_swap_ready(self, result: SwapRequestResponseMessage):
        """
        Handle the swap ready signal from the leaf node.
        We update self.pending_swap_operation and check if we can perform the swap operation
        If we are ready, we will perform the swap operation
        :param result: SwapReadyMessage
        :return:
        """
        # we need to check if the qubits are ready to swap
        if result.operation_key in self.pending_swap_operation:
            if result.source_node == self.swapping_node.left:
                self.pending_swap_operation[result.operation_key].left_ready = True
            elif result.source_node == self.swapping_node.right:
                self.pending_swap_operation[result.operation_key].right_ready = True
            # check if we can perform the swap operation
            if self.pending_swap_operation[result.operation_key].left_ready and \
                    self.pending_swap_operation[result.operation_key].right_ready:
                # perform the swap operation
                swapping_pair = self.pending_swap_operation[result.operation_key]
                self.pending_swap_operation.pop(result.operation_key)
                yield from self.handle_swapping(swapping_pair)

    def handle_swap_failed(self, message):
        """
        Handle the swap failed signal. The swap node will send swap failed signal left, self, and right node
        When a node received the swap failed signal, it will loop through the stack to find the original
        entangled node and re-entangle the qubits by sending RE-ENTANGLE-UPPER with memo pos
        Handle the re-entangle process when swap failed for a pair.
        We will need
        1. find the original entangled node
        2. send re-entangle signal to the entangled node if we are not the swap node of this swap operation
        3. send the re-entangle signal lower layer
        :param message: SwapFailedMessage
        :return:
        """
        # find the original entangled node
        entangled_node = self.get_original_entangled_node((message.source_node, message.target_node))
        # send the re-entangle to entangled node if we are not the swap node
        if self.swapping_node and \
                (self.swapping_node.left != entangled_node and self.swapping_node.right != entangled_node):
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=entangled_node,
                                                     data=SwapFailedMessage(
                                                         source_node=entangled_node,
                                                         target_node=self.node.name,
                                                         memo_pos=message.memo_pos)))

        # send the re-entangle signal to the lower layer
        self.send_re_entangle(entangled_node, message.memo_pos)

    def get_original_entangled_node(self, edge: tuple):
        """
        find the original entangled node in the stack
        :param edge: the edge of the swapping stack (source, target)
        :return:
        """
        edge_copy = (edge[0], edge[1])
        while edge_copy in self.swapping_stack:
            inter_nodes = self.swapping_stack[edge_copy]
            from_node = inter_nodes[0]
            if from_node == self.node.name and len(inter_nodes) > 1:
                return inter_nodes[1]
            # avoid infinite loop
            if edge_copy[1] == from_node:
                return edge[1]
        return edge[1]

    def send_re_entangle(self, entangled_node, memo_pos):
        """
        Send the re-entangle to lower layer
        :param entangled_node: entangled node name
        :param memo_pos: the memory position
        :return:
        """
        self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER,
                                            ReEntangleSignalMessage(entangled_node,
                                                                    memo_pos))

    def reset(self):
        self.entangle_reset()
        super().reset()

    def run(self):
        """
        Run the protocol
        We will swap the qubits between the nodes if the qubits are entangled
        New logic implementation with multiple swapping paris. We will have continues entangled pairs being
            created, therefore we have to keep track of the qubits position at each stage.
            1. we check if we are swapping node
            2. if we are the swapping node, we check if the qubits are ready to swap, left and right qubits are ready
            3. if the qubits are ready, we perform the swap operation. (We removed swapping index logic as the nature
            process of swapping will handle the swapping index)
            4. if the swap is successful, we apply the correction to the node on the right side
            5. right side node will apply the correction and send the success signal to the left side node
            6. if the swap is failed, we re-entangle the qubits.
                - The idea is that the re-entangle signal will be send to the left, right and swapping node
                - The left and right node will free the qubits and re-entangle to its corresponding entangled node
                - we will keep a stack of how we end up with the entangled qubits, so we can send re-entangle signal
                along the node.
                - i.e  A -> B -> C -> D -> E -> F -> G
                 A               C                        E                          G
                 A -> C: B       C -> A: B, C -> E: D     E -> C: D, E -> G: F       G -> E: F
                                 C -> G: E                                           G -> C: E
                 A -> E: C                                E -> A: C                      (Let's say we failed at here)
                 A -> G: E                                                           G -> A: E
                - re-entangle signal will be sent to:
                    A            B           C            D            E            F            G
                    A -> C       B -> C      C -> E       D -> E       E -> C       F -> G       G -> E
                    A -> B       B -> B      C -> D       D -> D       E -> D       F -> F       G -> F
                    A -> A                   C -> C                    E -> E                    G -> G
        """

        swap_signals = (self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_NEED) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_APPLY_CORRECTION) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_READY) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_FAILED) |
                        self.await_signal(self.cc_message_handler,
                                          signal_label=MessageType.SWAP_APPLY_CORRECTION_SUCCESS))

        while True:

            # handle other operations first then we try to perform the swap operation
            expr = yield self.qubit_input_signal | swap_signals

            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    # result -> EntangleSignalMessage (Maybe Purification or Verification)
                    result = ready_signal.result
                    mem_pos = result.mem_pos
                    entangle_node = result.entangle_node
                    self.logger.info(f"Swap {self.name} -> Entangle signal from {entangle_node}, mem_pos: {mem_pos}",
                                     color="blue")
                    self.entangled_qubits[entangle_node][mem_pos] = result.fidelity

                    # keep track of the stack of node edge
                    if (result.source_node, result.entangle_node) not in self.swapping_stack:
                        self.swapping_stack[(result.source_node, result.entangle_node)].append(result.source_node)
                        # add additional stack as we are the source node
                        if result.is_source:
                            self.swapping_stack[(result.source_node, result.entangle_node)].append(entangle_node)
                    # check if the qubits are ready to swap
                    self.check_swap_condition()

            elif expr.second_term.value:
                # case we have any swap signal
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if ready_signal.label == MessageType.SWAP_NEED:
                        # the swap node tell the leaf node that they need to swap the qubits to target through source
                        message: SwapRequestResponseMessage = result.data
                        self.logger.info(f"Swap {self.name} -> Swap need signal\n"
                                         f"From: {result.from_node}\n"
                                         f"Source: {message.source_node}\n"
                                         f"Target: {message.target_node}\n"
                                         f"Intermediate: {message.intermediate_node}\n"
                                         f"Mem Pos: {message.memo_pos}", color="yellow")
                        self.handle_swap_need(message)

                    elif ready_signal.label == MessageType.SWAP_READY:
                        # the swap node knows that leaf node is ready to swap
                        message: SwapRequestResponseMessage = result.data
                        self.logger.info(f"Swap {self.name} -> Swap ready signal\n"
                                         f"From: {result.from_node}\n"
                                         f"Source: {message.source_node}\n"
                                         f"Target: {message.target_node}\n"
                                         f"Intermediate: {message.intermediate_node}\n"
                                         f"Mem Pos: {message.memo_pos}", color="purple")

                        yield self.handle_swap_ready(message)

                    elif ready_signal.label == MessageType.SWAP_APPLY_CORRECTION:
                        # apply the correction
                        message: SwapApplyCorrectionMessage = result.data
                        yield self.apply_corrections(message)
                    elif ready_signal.label == MessageType.SWAP_SUCCESS:
                        # swap success message from intermediate node
                        message: SwapSuccessMessage = result.data
                        self.handle_swap_success(message)
                    elif ready_signal.label == MessageType.SWAP_APPLY_CORRECTION_SUCCESS:
                        # the correction is successful and message send by the target node
                        message: SwapApplyCorrectionSuccessMessage = result.data
                        self.logger.info(f"Swap {self.name} -> Correction successful\n"
                                         f"Operation Key: {message.operation_key}", color="green")
                        self.handle_swap_apply_success(message)
                    elif ready_signal.label == MessageType.SWAP_FAILED:
                        # re-entangle the qubits
                        message: SwapFailedMessage = result.data
                        self.logger.info(f"Swap {self.name} -> Swap Failed\n"
                                         f"Source Node: {message.source_node}\n"
                                         f"Target Node: {message.target_node}\n"
                                         f"Mem Pos: {message.memo_pos}",
                                         color="red")
                        self.handle_swap_failed(message)

            # case we finish the final entanglement
            if self.node.name == self.final_entanglement[0] and self.final_entanglement[1] in self.entangled_qubits:
                if len(self.entangled_qubits[self.final_entanglement[1]]) == self.max_pairs:
                    # we finish the final entanglement
                    self.send_signal(Signals.SUCCESS,
                                     {self.final_entanglement[1]: self.entangled_qubits[self.final_entanglement[1]]})
                    break
            if self.node.name == self.final_entanglement[1] and self.final_entanglement[0] in self.entangled_qubits:
                if len(self.entangled_qubits[self.final_entanglement[0]]) == self.max_pairs:
                    # we finish the final entanglement
                    self.send_signal(Signals.SUCCESS,
                                     {self.final_entanglement[0]: self.entangled_qubits[self.final_entanglement[0]]})
                    break
