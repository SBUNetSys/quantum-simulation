import operator
from functools import reduce
from collections import defaultdict

import numpy as np
from netsquid import sim_time
from netsquid.qubits import operators
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import  INSTR_Z, INSTR_X

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
    def __init__(self, left_node, right_node, left_pos, right_pos, left_pos_on_right,
                 right_pos_on_left, left_ready, right_ready):
        self.left_node = left_node
        self.right_node = right_node
        self.left_pos_on_right = left_pos_on_right
        self.right_pos_on_left = right_pos_on_left
        self.left_ready = left_ready
        self.right_ready = right_ready
        self.left_pos = left_pos
        self.right_pos = right_pos

class SwappingResult:
    def __init__(self):
        self.m1 = None
        self.m2 = None
        self.success = False

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
                 is_top_layer=False,
                 delay_time=0):

        super().__init__(node=node, name=name)
        # check if we are a swapping node or not.
        # if we are the swapping node, we will perform the swap operation between the left and right neighbors
        self.swapping_node = None
        for node in swapping_nodes:
            if node.parent == self.node.name:
                self.swapping_node = node

        # qubits that are entangled node_name -> {memory_pos: target_node_memo_pos}
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
        # re-entangle tracking
        self.re_entangle_paris = defaultdict(dict) # entangle_node : {mem_pos:}
        self.re_entangle_edge = defaultdict(bool)
        # final entanglement
        self.final_entanglement = final_entanglement
        # keep track of the node name and actual memory name
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        self.max_pairs = max_pairs
        self.is_top_layer = is_top_layer
        self.add_signal(MessageType.SWAP_FINISHED)
        self.start_time = sim_time()
        self.delay_time = delay_time
        self.is_source = False

    def add_new_signal(self, signal):
        self.add_signal(signal)

    def get_qmemory(self, memory_name):
        """
        Get the quantum memory of the node
        :param memory_name:
        :return:
        """
        return self.node.subcomponents[memory_name]

    def perform_swap(self, q1_mem_pos, q2_mem_pos, q1_mem_name, q2_mem_name, result, success_rate=0.5):
        """
        Perform swap measurement on the qubits
        :param q1_mem_pos:
        :param q2_mem_pos:
        :param q1_mem_name:
        :param q2_mem_name:
        :param result: SwappingResult
        :param success_rate:
        :return:
        """
        # # Simulating a swap operation with possible failures
        # yield self.await_timer(1)  # Simulate some operation time

        # Simulate Bell state measurement
        success_probability = 0.8
        # 90% success rate for Bell state measurement
        np.random.seed(0)
        if np.random.random() > success_probability:
            result.success = False
            return

        q1_qmemory = self.get_qmemory(q1_mem_name)
        q2_qmemory = self.get_qmemory(q2_mem_name)
        self.logger.info(f"Sawp {self.name} -> Perform Swap\n"
                         f"Node Left: {q1_mem_name}\n"
                         f"Left Pos {q1_mem_pos}\n"
                         f"Node Right {q2_mem_name}\n"
                         f"Right Pos {q2_mem_pos}", color="cyan")
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
        result.m1 = m1
        result.m2 = m2
        result.success = True


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
        self.entangled_qubits[message.target_node][message.memo_pos] = message.target_mem_pos
        # update the stack
        self.swapping_stack[(message.source_node, message.target_node)].append(message.intermediate_node)
        # send to upper layer saying we have wanted e2d connection
        if self.node.name == self.final_entanglement[1] and message.target_node == self.final_entanglement[0]:
            self.send_signal(Signals.SUCCESS, SwapEntangledSuccess(
                source_node=self.node.name,
                entangle_node=message.target_node,
                memo_pos=message.memo_pos,
                actual_entangle_node=self.get_qmemory_from_stack(message.target_node),
                target_memo_pos=message.target_mem_pos,
                is_source=self.is_source,
            ))
        # send the success signal to the intermediate node so it can send the success signal to the left node
        self.cc_message_handler.send_message(MessageType.SWAP_APPLY_CORRECTION_SUCCESS,
                                                message.intermediate_node,
                                             ClassicalMessage(
                                                 from_node=self.node.name,
                                                 to_node=message.intermediate_node,
                                                 data=SwapApplyCorrectionSuccessMessage(
                                                     operation_key=message.operation_key)
                                             ))
        self.logger.info(f"Swap {self.name} -> Applied correction\n"
                         f"Node: {self.node.name}\n"
                         f"Entangled Pairs {self.entangled_qubits}\n"
                         f"Swapping Stack: {self.swapping_stack}", color="green")

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

        q1_mem_pos = swapping_pair.left_pos
        q2_mem_pos = swapping_pair.right_pos
        self.logger.info(f"Swap {self.name} -> Perform swap\n"
                         f"\tBetween {left_node} and {right_node}\n"
                         f"\tSwapping Node {self.node.name}\n"
                         f"\tEntangled Qubits {self.entangled_qubits}", color="cyan")
        swap_result = SwappingResult()
        yield from self.perform_swap(q1_mem_pos, q2_mem_pos, q1_mem_name, q2_mem_name, swap_result)
        if swap_result.success:
            # case we success the swap
            self.logger.info(f"Swap {self.name} -> Swap success\n"
                             f"Left Node {q1_mem_name}\n"
                             f"Right Node {q2_mem_name}", color="green")
            key_pair = (left_node, right_node, q1_mem_pos, q2_mem_pos)
            self.pending_swap_success_confirmation[key_pair] = swapping_pair
            # send the message to the right node to perform the correction
            self.cc_message_handler.send_message(MessageType.SWAP_APPLY_CORRECTION,
                                                 swapping_pair.right_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.right_node,
                                                     data=SwapApplyCorrectionMessage(
                                                         source_node=swapping_pair.right_node,
                                                         target_node=swapping_pair.left_node,
                                                         intermediate_node=self.node.name,
                                                         operation_key=key_pair,
                                                         memo_pos=swapping_pair.right_pos_on_left,
                                                         target_mem_pos=swapping_pair.left_pos_on_right,
                                                         m1=swap_result.m1,
                                                         m2=swap_result.m2)))

        else:
            # case we failed the swap
            # TODO we need to re-entangle the qubits, we need send the information to both left and right
            self.logger.info(f"Swap {self.name} -> Swap failed, re-entangle the qubits", color="red")
            # left node
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                 swapping_pair.left_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.left_node,
                                                     data=SwapFailedMessage(
                                                         source_node=swapping_pair.left_node,
                                                         target_node=self.node.name,
                                                         memo_pos=swapping_pair.left_pos_on_right)))
            # right node
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                    swapping_pair.right_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.right_node,
                                                     data=SwapFailedMessage(
                                                         source_node=swapping_pair.right_node,
                                                         target_node=self.node.name,
                                                         memo_pos=swapping_pair.right_pos_on_left)))
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
            left_ready = (self.swapping_node.left in self.entangled_qubits and
                          len(self.entangled_qubits[self.swapping_node.left]) > 0)
            right_ready = (self.swapping_node.right in self.entangled_qubits and
                            len(self.entangled_qubits[self.swapping_node.right]) > 0)
            if left_ready and right_ready:
                # pop the qubits from the entangled qubits
                self.logger.info(f"Swap {self.name} -> Start Swapping Operation\n"
                                 f"Operator {self.node.name}\n"
                                 f"Entangled Qubits {self.entangled_qubits}\n"
                                 , color="purple")
                left_qubit_pos, target_left_pos = self.entangled_qubits[self.swapping_node.left].popitem()
                right_qubit_pos, target_right_pos = self.entangled_qubits[self.swapping_node.right].popitem()
                # create a swapping pair
                swapping_pair = SwappingPair(self.swapping_node.left, self.swapping_node.right,
                                             left_qubit_pos, right_qubit_pos, target_left_pos, target_right_pos,
                                             False, False)
                # store the pending swap request
                pair_key = (self.swapping_node.left, self.swapping_node.right, left_qubit_pos, right_qubit_pos)
                self.pending_swap_operation[pair_key] = swapping_pair
                self.logger.info(f"Swap {self.name} -> Start Swapping Operation\n"
                                 f"Operator {self.node.name}\n"
                                 f"Left Node, Left Pos, Target left Pos:"
                                 f" {self.swapping_node.left}, {left_qubit_pos}, {target_left_pos}\n"
                                 f"Right Node, Right Pos, Target right Pos:"
                                 f" {self.swapping_node.right}, {right_qubit_pos}, {target_right_pos}\n"
                                 , color="purple")
                # send swap signal to the left and right node
                self.cc_message_handler.send_message(MessageType.SWAP_NEED,
                                                     self.swapping_node.left,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=self.swapping_node.left,
                                                         data=SwapRequestResponseMessage(self.swapping_node.left,
                                                                                         self.swapping_node.right,
                                                                                         self.node.name,
                                                                                         target_left_pos,
                                                                                         pair_key)))
                self.cc_message_handler.send_message(MessageType.SWAP_NEED,
                                                     self.swapping_node.right,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=self.swapping_node.right,
                                                         data=SwapRequestResponseMessage(self.swapping_node.right,
                                                                                         self.swapping_node.left,
                                                                                         self.node.name,
                                                                                         target_right_pos,
                                                                                         pair_key)))
                # remove any re-entangle info
                left_edge = (self.node.name, swapping_pair.left_node, swapping_pair.left_pos)
                right_edge = (self.node.name, swapping_pair.right_node, swapping_pair.right_pos)
                if left_edge in self.re_entangle_edge:
                    self.re_entangle_edge.pop(left_edge)
                if right_edge in self.re_entangle_edge:
                    self.re_entangle_edge.pop(right_edge)
        # we check if we have the swap request from the swap node
        remove_keys = []
        for key, value in self.pending_swap_request.items():
            value: SwapRequestResponseMessage
            if (value.intermediate_node in self.entangled_qubits and
                    value.memo_pos in self.entangled_qubits[value.intermediate_node]):
                # TODO send the swap ready signal to the swap node
                self.cc_message_handler.send_message(MessageType.SWAP_READY,
                                                        value.intermediate_node,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=value.intermediate_node,
                                                         data=SwapRequestResponseMessage(value.source_node,
                                                                                         value.target_node,
                                                                                         value.intermediate_node,
                                                                                         value.memo_pos,
                                                                                         value.operation_key)))
                # remove the pending swap result
                remove_keys.append(key)
                # pop the qubit as it was being used for swapping operation
                if value.memo_pos in self.entangled_qubits[value.intermediate_node]:
                    self.entangled_qubits[value.intermediate_node].pop(value.memo_pos)
                # remove re-entangle info
                edge = (self.node.name, value.intermediate_node, value.memo_pos)
                if edge in self.re_entangle_edge:
                    self.re_entangle_edge.pop(edge)
        for key in remove_keys:
            self.pending_swap_request.pop(key)
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
                                                    swapping_pair.left_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=swapping_pair.left_node,
                                                     data=SwapSuccessMessage(
                                                         source_node=swapping_pair.left_node,
                                                         target_node=swapping_pair.right_node,
                                                         intermediate_node=self.node.name,
                                                         memo_pos=swapping_pair.left_pos_on_right,
                                                         target_memo_pos=swapping_pair.right_pos_on_left,
                                                     )))

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
        self.entangled_qubits[message.target_node][message.memo_pos] = message.target_memo_pos
        # update the stack
        self.swapping_stack[(message.source_node, message.target_node)].append(message.intermediate_node)
        if self.node.name == self.final_entanglement[0] and message.target_node == self.final_entanglement[1]:
            self.logger.info(f"Swap {self.name} -> Swap success send signal to upper layer\n"
                             f"Source Node: {message.source_node}\n"
                             f"Target Node: {message.target_node}\n"
                             f"Intermediate Node: {message.intermediate_node}\n"
                             f"Memo Pos: {message.memo_pos}\n"
                             f"Entangled Pairs {self.entangled_qubits}\n", color="yellow")
            self.send_signal(Signals.SUCCESS, SwapEntangledSuccess(
                source_node=self.node.name,
                entangle_node=message.target_node,
                memo_pos=message.memo_pos,
                actual_entangle_node=self.get_qmemory_from_stack(message.target_node),
                target_memo_pos=message.target_memo_pos,
                is_source=self.is_source
            ))
        self.logger.info(f"Swap {self.name} -> Swap success update on left node\n"
                         f"Source Node: {message.source_node}\n"
                         f"Target Node: {message.target_node}\n"
                         f"Intermediate Node: {message.intermediate_node}\n"
                         f"Memo Pos: {message.memo_pos}\n"
                         f"Entangled Pairs {self.entangled_qubits}\n"
                         f"Swapping Stack {self.swapping_stack}", color="green")


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
            edge = (self.node.name, result.intermediate_node, result.memo_pos)
            if edge in self.re_entangle_edge:
                self.re_entangle_edge.pop(edge)
            del self.entangled_qubits[result.intermediate_node][result.memo_pos]
            # we have the qubit, send the swap ready signal to the swap node
            self.cc_message_handler.send_message(MessageType.SWAP_READY,
                                                 result.intermediate_node,
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
        self.logger.info(f"Swap {self.name} -> Handle Swap Failed\n"
                         f"Node {self.node.name}\n"
                         f"Source Node: {message.source_node}\n"
                         f"Target Node: {message.target_node}\n"
                         f"Swapping Stack {self.swapping_stack}", color="red")
        if (message.source_node, message.target_node, message.memo_pos) in self.re_entangle_edge:
            # we pop the duplicate
            self.logger.info(f"Swap {self.name} -> Handle Swap Failed Duplicated\n"
                             f"Node {self.node.name}\n"
                             f"Source Node: {message.source_node}\n"
                             f"Target Node: {message.target_node}\n"
                             f"Swapping Stack {self.swapping_stack}", color="yellow")
            return
        entangled_nodes, new_edge = self.get_original_entangled_node((message.source_node, message.target_node),
                                                                     message.memo_pos)
        if len(entangled_nodes) > 1 :
            # case we are the source node, we need tell remote nodes

            self.logger.info(f"Swap {self.name} -> Swap failed as source nodes, re-entangle the qubits\n"
                             f"Node {self.node.name}\n"
                             f"Entangled Nodes: {entangled_nodes}\n"
                             f"New Edge: {new_edge}\n"
                             f"Source Node: {message.source_node}\n"
                             f"Target Node: {message.target_node}", color="red")
            entangled_node = entangled_nodes[1]
            self.logger.info(f"Swap {self.name} -> Swap failed, sending message to entangle node\n"
                             f"Node {self.node.name}\n"
                             f"To Node: {entangled_node}\n"
                             f"Entangle Nodes {entangled_nodes}", color="yellow")
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                 entangled_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=entangled_node,
                                                     data=SwapFailedMessage(
                                                         source_node=entangled_node,
                                                         target_node=self.node.name,
                                                         memo_pos=message.memo_pos)))
        else:
            entangled_node = new_edge[1]
        # send the re-entangle signal to the lower layer
        self.logger.info(f"Swap {self.name} -> Swap failed sending signal to lower layer\n"
                         f"Node {self.node.name}\n"
                         f"New Edge: {new_edge}\n"
                         f"Entangled Node: {entangled_node}\n"
                         f"Source Node: {message.source_node}\n"
                         f"Target Node: {message.target_node}", color="yellow")
        self.re_entangle_edge[(message.source_node, message.target_node, message.memo_pos)] = True
        self.send_re_entangle(entangled_node, message.memo_pos)

    def get_original_entangled_node(self, edge: tuple, mem_pos):
        """
        find the original entangled node in the stack
        :param edge: the edge of the swapping stack (source, target)
        :param mem_pos: the memory position of the entangled node
        :return:
        """
        edge_copy = (edge[0], edge[1])
        while edge_copy in self.swapping_stack:
            inter_nodes = self.swapping_stack[edge_copy]
            from_node = inter_nodes[0]
            # avoid infinite loop
            if from_node == self.node.name:
                return inter_nodes, edge_copy
            edge_copy = (edge_copy[0], from_node)
            self.logger.info(f"Swap {self.name} -> Swap faild, sending message to everyone\n"
                             f"Node {self.node.name}\n"
                             f"To Node: {from_node}\n", color="cyan")
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED,
                                                 from_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=from_node,
                                                     data=SwapFailedMessage(
                                                         source_node=from_node,
                                                         target_node=self.node.name,
                                                         memo_pos=mem_pos)))
        return edge_copy

    def send_re_entangle(self, entangled_node, memo_pos):
        """
        Send the re-entangle to lower layer
        :param entangled_node: entangled node name
        :param memo_pos: the memory position
        :return:
        """
        if entangled_node in self.entangled_qubits:
            if memo_pos in self.entangled_qubits[entangled_node]:
                self.entangled_qubits[entangled_node].pop(memo_pos)
        if entangled_node in self.re_entangle_paris:
            if memo_pos in self.re_entangle_paris[entangled_node]:
                # skip already entangled mem pos
                return
        self.re_entangle_paris[entangled_node][memo_pos] = None
        self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER,
                                            ReEntangleSignalMessage(entangled_node,
                                                                    [memo_pos]))

    def reset(self):
        self.pending_swap_operation = {}
        self.pending_swap_success_confirmation = {}
        self.pending_swap_request = {}
        self.entangled_qubits = defaultdict(dict)
        self.swapping_stack = defaultdict(list)
        self.re_entangle_paris = defaultdict(dict)
        self.re_entangle_edge = defaultdict(bool)

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
                                          signal_label=MessageType.SWAP_APPLY_CORRECTION_SUCCESS) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_SUCCESS))
        self.logger.info(f"Swap {self.name} -> Start swapping protocol", color="cyan")
        self.start_time = sim_time()
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
                    if result.is_source:
                        self.is_source = True
                    self.logger.info(f"Swap {self.name} -> Entangle signal from {entangle_node}, mem_pos: {mem_pos}",
                                     color="blue")
                    # Here the fid is replaced by target memo_pos. When entangle from lower,
                    # our mem_pos is always matching
                    if type(mem_pos) == list:
                        for pos in mem_pos:
                            self.entangled_qubits[entangle_node][pos] = pos
                    else:
                        self.entangled_qubits[entangle_node][mem_pos] = mem_pos

                    # keep track of the stack of node edge
                    if (result.source_node, result.entangle_node) not in self.swapping_stack:
                        self.swapping_stack[(result.source_node, result.entangle_node)].append(result.source_node)
                        # add additional stack as we are the source node
                        if result.is_source:
                            self.swapping_stack[(result.source_node, result.entangle_node)].append(entangle_node)
                    if type(mem_pos) == list:
                        for pos in mem_pos:
                            if entangle_node in self.re_entangle_paris and pos in self.re_entangle_paris[
                                entangle_node]:
                                self.re_entangle_paris[entangle_node].pop(pos)
                    else:
                        if entangle_node in self.re_entangle_paris and mem_pos in self.re_entangle_paris[entangle_node]:
                            self.re_entangle_paris[entangle_node].pop(mem_pos)
                    # check if the qubits are ready to swap
                    self.check_swap_condition()

            elif expr.second_term.value:
                # case we have any swap signal
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if result.data.timestamp < self.start_time:
                        continue
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

                        yield from self.handle_swap_ready(message)

                    elif ready_signal.label == MessageType.SWAP_APPLY_CORRECTION:
                        # apply the correction
                        message: SwapApplyCorrectionMessage = result.data
                        yield from self.apply_corrections(message)
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
                        self.logger.info(f"Swap {self.name} -> Received Swap Failed\n"
                                         f"From {result.from_node}\n"
                                         f"Source Node: {message.source_node}\n"
                                         f"Target Node: {message.target_node}\n"
                                         f"Mem Pos: {message.memo_pos}",
                                         color="yellow")
                        self.handle_swap_failed(message)
            # check if the qubits are ready to swap
            self.check_swap_condition()
            # case we finish the final entanglement and we are the top layer. We need to stop
            if self.is_top_layer:
                if self.node.name == self.final_entanglement[0] and self.final_entanglement[1] in self.entangled_qubits:
                    if len(self.entangled_qubits[self.final_entanglement[1]]) == self.max_pairs:
                        self.logger.info(f"Swap {self.name} -> Final entanglement finished\n"
                                         f"Source Node: {self.final_entanglement[0]}\n"
                                         f"Target Node: {self.final_entanglement[1]}\n"
                                         f"Sim time: {sim_time()}", color="green")
                        if self.delay_time > 0:
                            yield self.await_timer(self.delay_time)
                        # we finish the final entanglement
                        self.send_signal(MessageType.SWAP_FINISHED,
                                         {self.final_entanglement[1]: self.entangled_qubits[self.final_entanglement[1]]})

                        break
                if self.node.name == self.final_entanglement[1] and self.final_entanglement[0] in self.entangled_qubits:
                    if len(self.entangled_qubits[self.final_entanglement[0]]) == self.max_pairs:
                        # we finish the final entanglement
                        self.logger.info(f"Swap {self.name} -> Final entanglement finished\n"
                                         f"Source Node: {self.final_entanglement[1]}\n"
                                         f"Target Node: {self.final_entanglement[0]}", color="green")
                        self.send_signal(MessageType.SWAP_FINISHED,
                                         {self.final_entanglement[0]: self.entangled_qubits[self.final_entanglement[0]]})
                        break
