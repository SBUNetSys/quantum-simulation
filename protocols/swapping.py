import operator
from functools import reduce

import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
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
from pydynaa import EventExpression
from netsquid.qubits.qubitapi import measure
from netsquid.qubits.operators import CNOT, Z
from netsquid.components.instructions import INSTR_MEASURE
from netsquid.nodes import Node
from netsquid.qubits.qubitapi import fidelity
from messagehandler import *


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


class SwapProtocol(NodeProtocol):
    """
    A protocol that swap qubits between two nodes.
    Swapping Logic:

    A -> B -> C

    """

    def __init__(self, node, name, swapping_tree, qubit_input_signals, cc_message_handler, final_entanglement):
        super().__init__(node=node, name=name)
        self.swapping_tree = swapping_tree
        self.swap_index = 0
        # qubits that are entangled node_name -> memory position
        self.entangled_qubits = {}
        # qubits that are temporarily stored waiting for confirmation from entangled node
        self.temp_qubits = {}
        # store the classical message, where the case that when the remote send the confirmation entangled message
        # but the local node has not received the entangled message yet
        self.entangle_message_queue = []
        # store the swap need signal that arrived by next level node but the local node still in previous level
        self.swap_need_message_queue = []
        # a node will have multiple swapping request during the swapping process
        # example A -> B -> C -> D -> E, B and D will both send SWAPPING_NEED to C.
        self.swapping_queue = {}  # source node -> target node
        self.swapping_qubits = {}
        # qubit input signal, which can be from source or remote node
        await_signals = [self.await_signal(protocol, Signals.SUCCESS) for protocol in qubit_input_signals]
        # have expression to wait for the qubit input signal
        self.qubit_input_signal = reduce(operator.or_, await_signals)
        # classical message handler
        self.cc_message_handler = cc_message_handler
        # add entangle signal so that the protocol can be triggered

        # final entanglement
        self.final_entanglement = final_entanglement
        # keep track of the node name and actual memory name
        # we swapped the qubits however the physical memory name is different from the swapped node name
        self.memory_mapping = {}  # current node name -> memory node name

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
        q1, = q1_qmemory.peek(q1_mem_pos)
        if q2_qmemory.busy:
            yield self.await_program(q2_qmemory)
        q2, = q2_qmemory.peek(q2_mem_pos)
        # print_red(f"Swap {self.name} -> Fidelity before swap: {fidelity([q1, q2], ks.b00)}")
        # apply CNOT on both qubit
        qapi.operate(qubits=[q1, q2], operator=operators.CNOT)
        # apply Hadamard on q1
        qapi.operate(qubits=[q1], operator=operators.H)
        # measure the qubits so we get m1 and m2
        m1, _ = qapi.measure(q1)
        m2, _ = qapi.measure(q2)
        return True, m1, m2

    def apply_corrections(self, m1, m2, node_name, qmem_pos, target_node):
        print_yellow(f"Swap {self.name} -> Apply correction for {target_node}, pos {qmem_pos}"
                     f" by {self.node.name}, with qmem_name: {self.get_qmemory_from_mapping(node_name)}")

        qmemory = self.get_qmemory(f"{self.get_qmemory_from_mapping(node_name)}_qmemory")

        if m1 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            print_green(f"Sawp {self.name} -> Apply correction Z on qubit {qmem_pos} with qmem_name: {qmemory.name}")
            qmemory.execute_instruction(INSTR_Z, [qmem_pos])
        if m2 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            print_green(f"Sawp {self.name} -> Apply correction X on qubit {qmem_pos} with qmem_name: {qmemory.name}")
            qmemory.execute_instruction(INSTR_X, [qmem_pos])

        # send the success signal to the target node
        self.cc_message_handler.send_message(MessageType.CORRECTION_SUCCESS,
                                             target_node,
                                             {"from": self.node.name,
                                              "to": target_node})

    def get_qmemory_from_mapping(self, node_name):
        while node_name in self.memory_mapping:
            node_name = self.memory_mapping[node_name]
        return node_name

    def handle_swapping(self, swap_node):
        """
        Handle the swapping operation
        :param swap_node:
        :return:
        """
        if swap_node is None:
            return
        if not self.swap_need_sent[(swap_node, self.swap_index)][swap_node.left] or \
                not self.swap_need_sent[(swap_node, self.swap_index)][swap_node.right]:
            return
        left_node = swap_node.left
        right_node = swap_node.right

        q1_mem_name = f"{self.get_qmemory_from_mapping(left_node)}_qmemory"
        q2_mem_name = f"{self.get_qmemory_from_mapping(right_node)}_qmemory"
        q1_mem_pos = self.entangled_qubits[swap_node.left]
        q2_mem_pos = self.entangled_qubits[swap_node.right]
        print_cyan(f"Swap {self.name} -> Perform swap between {left_node} and {right_node} by {self.node.name}")
        success, m1, m2 = yield from self.perform_swap(q1_mem_pos, q2_mem_pos, q1_mem_name, q2_mem_name)
        if success:
            # send the apply correction message to the left and right
            # yield self.apply_corrections(m1, m2, parent_mem_pos, parent_mem_name)
            # remove the qubits from the entangled qubits
            self.entangled_qubits.pop(swap_node.left)
            self.entangled_qubits.pop(swap_node.right)

            # send the message to the right node to perform the correction
            self.cc_message_handler.send_message(MessageType.SWAP_RESULT,
                                                 swap_node.right,
                                                 {"result": (True, m1, m2),
                                                  "from": self.node.name,
                                                  "to": swap_node.right})
            self.swap_index += 1
        else:
            print_red(f"Swap {self.name} -> Swap failed, re-entangle the qubits")
            self.entangle_reset()

            # free the left qubit and send the re-entangle signal to the left node
            self.send_signal(f"entangle_{self.node.name}->{swap_node.left}",
                             {"mem_pos": q1_mem_pos,
                              "qmemory_name": f"{swap_node.left}_qmemory"})
            self.cc_message_handler.send_message(MessageType.RE_ENTANGLE,
                                                 swap_node.left,
                                                 {"from": self.node.name,
                                                  "to": swap_node.left,
                                                  "mem_pos": q1_mem_pos})

            # send the re-entangle signal to the right node, let the right node free the qubit
            self.cc_message_handler.send_message(MessageType.RE_ENTANGLE,
                                                 swap_node.right,
                                                 {"from": self.node.name,
                                                  "to": swap_node.right,
                                                  "mem_pos": q2_mem_pos})
            # we need to wait for the right node to clear up its memory position and re-entangle.
            # TODO: should we make sure via classical message that the right node has cleared up the memory position?
            yield self.await_timer(1000)
            # entangle the qubits again, we only send to the right as we are the source to gen the qubit
            self.send_signal(f"entangle_{self.node.name}->{swap_node.right}",
                             {"mem_pos": q2_mem_pos,
                              "qmemory_name": f"{swap_node.right}_qmemory"})

    def process_entangle_message(self):
        """
        Process the entangle message if the local node has not received the entangle message yet
        :return:
        """
        temp = self.entangle_message_queue
        self.entangle_message_queue = []
        for message in temp:
            if message["from"] in self.temp_qubits:
                self.entangled_qubits[message["from"]] = message["mem_pos"]
                self.temp_qubits.pop(message["from"])
                # check if the qubits are ready to swap
                self.check_swap_ready()
            else:
                self.entangle_message_queue.pop(message)

    def process_swap_need_message(self):
        """
        Process the swap need message if the local node is not in the same level
        :return:
        """
        temp = self.swap_need_message_queue
        self.swap_need_message_queue = []
        for message in temp:
            self.handle_swap_need(message)

    def check_swap_ready(self):
        """
        Check if the qubits are ready to swap, if so send the message to the swapping node
        :return:
        """
        if len(self.swapping_queue) == 0:
            return

        for source, target in self.swapping_queue.items():
            if source in self.entangled_qubits:
                # temporarily store the qubit that we are swapping
                # key = intermediate node, value = (target node, qubit_memory_position)
                self.swapping_qubits[source] = (target, self.entangled_qubits[source])
                self.entangled_qubits.pop(source)
                # send the message to the swap target
                self.cc_message_handler.send_message(MessageType.SWAP_READY,
                                                     source,
                                                     {"from": self.node.name,
                                                      "to": source,
                                                      "swap_index": self.swap_index})
            else:
                # case the source qubit is not ready yet
                pass

    def handle_swap_success(self, source):
        """
        Handle the swap success message and update the entangled qubits
        :return:
        """
        target_node, mem_pos = self.swapping_qubits[source]
        self.entangled_qubits[target_node] = mem_pos
        # record the memory mapping
        self.memory_mapping[target_node] = source
        # remove the qubits from the swapping qubits
        self.swapping_qubits.pop(source)
        # dequeue the source from the swapping queue
        self.swapping_queue.pop(source)
        # TODO: we wait here for the right side swap
        # The case of A -> B -> C -> D -> E
        # A and C are swapping, C and E are swapping, we need to wait for the B and D to finish
        # or A and C to finish
        if len(self.swapping_qubits) == 0:
            self.swap_index += 1

    def handle_swap_need(self, result):
        """
        Handle the swap need signal
        :param result: classical message result
        :return:
        """
        # case of not in the same level, we need to store the message
        if self.swap_index == result["swap_index"]:
            # case we are in the same level
            self.swapping_queue[result["source"]] = result["target"]
            # check if the qubits are ready to swap
            self.check_swap_ready()
        else:
            # we check if we have previous performed swapping or not
            if len(self.swapping_queue) == 0:
                # case we are not in the same level, but we are the node supposed to swap
                # example A -> B -> C -> D, A and D are swapping after A and C finish.
                # A and C is at level 1, but D is at level 0, however it is the proper node to swap
                # the unique is that D will have source and target as None
                self.swapping_queue[result["source"]] = result["target"]
                # we update the index to match the swap index
                self.swap_index = result["swap_index"]
                self.check_swap_ready()
            else:
                # we are not in the same level, and we have previous swap source and target
                # we need to store the swap need signal
                print_yellow(f"Swap {self.name} -> Swap need signal from "
                             f"{result['source']} to "
                             f"{result['target']} but not in the same level, store the message")
                self.swap_need_message_queue.append(result)

    def entangle_reset(self):
        """
        Reset the entangled qubits
        :return:
        """
        self.entangled_qubits = {}
        self.temp_qubits = {}
        self.entangle_message_queue = []
        self.swap_need_message_queue = []
        self.swapping_qubits = {}
        self.swap_source = None
        self.swap_target = None
        self.swap_index = 0
        self.swap_need_sent = {}
        self.memory_mapping = {}

    def reset(self):
        self.entangle_reset()
        super().reset()

    def run(self):
        """
        Run the protocol
        We will swap the qubits between the nodes if the qubits are entangled
        1. we check the swapping tree to see if we are the swapping node
        2. if we are the swapping node, we perform the swap operation if we have both qubits entangled
        3. if we are not the swapping node, we wait for the measurement results to apply corrections
        4. if measurement failed, we re-entangle the qubits, send a message to the both nodes to re-entangle
        5. if the measurement is successful, we apply the corrections
        6. we continue the process until we finish the final swapping goal
        :return:
        """

        swap_signals = (self.await_signal(self.cc_message_handler, signal_label=MessageType.ENTANGLED) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_NEED) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_RESULT) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_READY) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.RE_ENTANGLE) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.CORRECTION_SUCCESS))
        # store the swap need signal that we have sent and also the ready signal from the leaf node
        self.swap_need_sent = {}
        while True:
            # try to check if we are the swapping node
            if self.swap_index < len(self.swapping_tree):
                swap_node = None
                swap_level = self.swapping_tree[self.swap_index]
                for swap in swap_level:
                    if swap.parent == self.node.name:
                        # we are the swapping node
                        swap_node = swap
                        break
                if swap_node is not None and (swap_node, self.swap_index) not in self.swap_need_sent:
                    print_orange(f"Swap {self.name} -> Found swap node: {swap_node.parent} at index {self.swap_index}\n"
                                 f"\tSend swap need signal to {swap_node.left} and {swap_node.right}")
                    self.swap_need_sent[(swap_node, self.swap_index)] = {swap_node.left: False, swap_node.right: False}
                    self.cc_message_handler.send_message(MessageType.SWAP_NEED,
                                                         swap_node.left,
                                                         {"from": self.node.name,
                                                          "to": swap_node.left,
                                                          "source": self.node.name,
                                                          "target": swap_node.right,
                                                          "swap_index": self.swap_index})
                    self.cc_message_handler.send_message(MessageType.SWAP_NEED,
                                                         swap_node.right,
                                                         {"from": self.node.name,
                                                          "to": swap_node.right,
                                                          "source": self.node.name,
                                                          "target": swap_node.left,
                                                          "swap_index": self.swap_index})
            # handle other operations first then we try to perform the swap operation
            expr = yield (self.qubit_input_signal | swap_signals
                          )
            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    mem_pos = result["mem_pos"]
                    self.is_source = result["is_source"]
                    qmemory_name = result["qmemory"]
                    entangle_node = result["entangle_node"]
                    if self.is_source:
                        # store the qubit in the temporary qubits
                        self.temp_qubits[entangle_node] = mem_pos
                        print_blue(f"Swap {self.name} -> Entangle signal from QSource, mem_pos: {mem_pos}")
                    else:
                        # add the qubit to the entangled qubits
                        print_blue(f"Swap {self.name} -> Entangle signal from {entangle_node}, mem_pos: {mem_pos}")
                        self.entangled_qubits[entangle_node] = mem_pos
                        # send the entangled signal to the source node
                        self.cc_message_handler.send_message(MessageType.ENTANGLED, entangle_node, {
                            "from": self.node.name,
                            "to": entangle_node,
                            "mem_pos": mem_pos})
                        # check if the qubits are ready to swap
                        self.check_swap_ready()

            elif expr.second_term.value:
                # case we have any swap signal
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if ready_signal.label == MessageType.SWAP_NEED:
                        # the swap node tell the leaf node that they need to swap the qubits to target through source
                        print_yellow(f"Swap {self.name} -> Swap need signal from "
                                     f"{result['source']} to "
                                     f"{result['target']}")

                        self.handle_swap_need(result)

                    elif ready_signal.label == MessageType.SWAP_READY:
                        # the swap node knows that leaf node is ready to swap
                        print_purple(f"Swap {self.name} -> Swap ready signal from {result['from']}")
                        self.swap_need_sent[(swap_node, self.swap_index)][result["from"]] = True

                    elif ready_signal.label == MessageType.SWAP_RESULT:
                        # apply the correction
                        success, m1, m2 = result["result"]
                        print_yellow(f"Swap {self.name} -> Swap result from {result['from']} -> {result['to']}\n"
                                     f"\tSuccess: {success}, m1: {m1}, m2: {m2}")
                        if success:
                            # apply corrections

                            yield from self.apply_corrections(m1, m2,
                                                              result['from'],
                                                              self.swapping_qubits[result['from']][1],
                                                              self.swapping_queue[result['from']])
                            # add the qubits to the entangled qubits
                            print_green(f"Swap {self.name} -> Correction Applied successful by {self.node.name}")
                            self.handle_swap_success(result['from'])
                        else:
                            # re-entangle the qubits
                            _, mem_pos = self.swapping_qubits[self.swap_source]

                            self.send_signal(f"entangle_{self.node.name}",
                                             {"mem_pos": mem_pos,
                                              "qmemory_name": f"{result['from']}_qmemory"})
                            # reset the swap source and target
                            self.entangle_reset()
                    elif ready_signal.label == MessageType.ENTANGLED:
                        print_blue(f"Swap {self.name} -> Entangled signal from {result['from']}")
                        # add the qubit to the entangled qubits
                        if result["from"] in self.temp_qubits:
                            self.entangled_qubits[result["from"]] = result["mem_pos"]
                            # check if the qubits are ready to swap
                            self.check_swap_ready()
                        else:
                            self.entangle_message_queue.append(result)
                    elif ready_signal.label == MessageType.CORRECTION_SUCCESS:
                        # the correction is successful and message send by the target node
                        print_green(f"Swap {self.name} -> Correction successful from {result['from']}")
                        # since this is from right node directly to the left node,
                        # we need to find the source node
                        send_source = None
                        for source, target in self.swapping_queue.items():
                            if target == result['from']:
                                send_source = source
                        if send_source is not None:
                            self.handle_swap_success(send_source)
                        else:
                            print_red(f"Swap {self.name} -> Correction successful but not found the source node")
                    elif ready_signal.label == MessageType.RE_ENTANGLE:
                        # re-entangle the qubits
                        print_red(f"Swap {self.name} -> Re-entangle signal from {result['from']} to {result['to']}")
                        self.entangle_reset()
                        mem_pos = result["mem_pos"]
                        self.send_signal(f"entangle_{self.node.name}->{result['from']}",
                                         {"mem_pos": mem_pos,
                                          "qmemory_name": f"{result['from']}_qmemory"})

            # process the entangle message
            self.process_swap_need_message()
            self.process_entangle_message()
            yield from self.handle_swapping(swap_node)
            # case we finish the final entanglement
            if self.node.name == self.final_entanglement[0] and self.final_entanglement[1] in self.entangled_qubits:
                # we finish the final entanglement
                self.send_signal(Signals.SUCCESS,
                                 {self.final_entanglement[1]: self.entangled_qubits[self.final_entanglement[1]]})
                break
            if self.node.name == self.final_entanglement[1] and self.final_entanglement[0] in self.entangled_qubits:
                # we finish the final entanglement
                self.send_signal(Signals.SUCCESS,
                                 {self.final_entanglement[0]: self.entangled_qubits[self.final_entanglement[0]]})
                break
