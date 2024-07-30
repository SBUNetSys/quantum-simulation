import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops, ketstates
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


class SwapProtocol(NodeProtocol):
    """
    A protocol that swap qubits between two nodes.
    Swapping Logic:

    A -> B -> C

    """

    def __init__(self, node, swapping_tree, qubit_input_signal_handler, cc_message_handler):

        super().__init__()
        self.node = node
        self.swapping_tree = swapping_tree
        self.swap_index = 0
        # qubits that are entangled node_name -> memory position
        self.entangled_qubits = {}
        self.swapping_qubits = {}
        # qubit input signal, which can be from source or remote node
        self.qubit_input_signal = self.await_signal(sender=qubit_input_signal_handler,
                                                    signal_label=Signals.SUCCESS)
        self.swap_ready = False
        # classical message handler
        self.cc_message_handler = cc_message_handler
        # add entangle signal so that the protocol can be triggered
        self.add_signal("entangle")

        # record the swapping source and target if any
        self.swap_source = None
        self.swap_target = None

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
        success_probability = 0.5  # 50% success rate for Bell state measurement
        if np.random.random() > success_probability:
            return False, None, None

        q1_qmemory = self.get_qmemory(q1_mem_name)
        q2_qmemory = self.get_qmemory(q2_mem_name)
        if q1_qmemory.busy:
            yield self.await_program(q1_qmemory)
        q1 = q1_qmemory.pop(q1_mem_pos)
        if q2_qmemory.busy:
            yield self.await_program(q2_qmemory)
        q2 = q2_qmemory.pop(q2_mem_pos)
        # apply CNOT on both qubit
        qapi.operate([q1, q2], CNOT)

        # measure the qubits so we get m1 and m2
        m1 = qapi.measure(q1)
        m2 = qapi.measure(q2)
        return True, m1, m2

    def apply_corrections(self, m1, m2, qmem_pos, qmem_name):
        qmemory = self.get_qmemory(qmem_name)

        if m1 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            qmemory.execute_instruction(INSTR_Z, qmem_pos)
        if m2 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            qmemory.execute_instruction(INSTR_X, qmem_pos)

    def handle_swapping(self, swap_node):
        """
        Handle the swapping operation
        :param swap_node:
        :return:
        """
        if swap_node is None:
            return
        q1_mem_name = f"{swap_node.left}_qmemory"
        q2_mem_name = f"{swap_node.right}_qmemory"
        q1_mem_pos = self.entangled_qubits[q1_mem_name]
        q2_mem_pos = self.entangled_qubits[q2_mem_name]

        success, m1, m2 = yield self.perform_swap(q1_mem_pos, q2_mem_pos, q1_mem_name, q2_mem_name)
        if success:
            # send the apply correction message to the left and right
            # yield self.apply_corrections(m1, m2, parent_mem_pos, parent_mem_name)
            # remove the qubits from the entangled qubits
            self.entangled_qubits.pop(q1_mem_name)
            self.entangled_qubits.pop(q2_mem_name)

            # send the message to the parent node
            self.cc_message_handler.send_message(MessageType.MEASUREMENT_RESULT,
                                                 {"result": (True, m1, m2)})
            self.swap_index += 1
        else:
            # send the message to the parent node to re-entangle
            self.cc_message_handler.send_message(MessageType.SWAP_FAILED, {"node": self.node.name})

    def check_swap_ready(self):
        """
        Check if the qubits are ready to swap, if so send the message to the swapping node
        :return:
        """
        if self.swap_source is None or self.swap_target is None:
            return
        if self.swap_source in self.entangled_qubits:
            # send the message to the swap target
            self.cc_message_handler.send_message(MessageType.SWAP_READY,
                                                 {"source": self.node})
            # temporarily store the qubit that we are swapping
            # key = intermediate node, value = (target node, qubit_memory_position)
            self.swapping_qubits[self.swap_source] = (self.swap_target, self.entangled_qubits[self.swap_source])
            self.entangled_qubits.pop(self.swap_source)
            self.swap_ready = True
        else:
            pass

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

        swap_signals = (self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_NEED) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_RESULT) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.SWAP_READY))
        while True:
            # try to check if we are the swapping node
            swap_node = None
            swap_level = self.swapping_tree[self.swap_index]
            for swap in swap_level:
                if swap.parent == self.node.name:
                    # we are the swapping node
                    swap_node = swap
                    break
            # handle other operations first then we try to perform the swap operation
            expr = yield (self.qubit_input_signal | swap_signals
                          )
            if expr.first_term.value:
                # case we have qubit input signal
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                result = ready_signal.result
                mem_pos = result["mem_pos"]
                self.is_source = result["is_source"]
                qmemory_name = result["qmemory_name"]
                self.entangled_qubits["entangle_node"] = mem_pos
                # check if the qubits are ready to swap
                self.check_swap_ready()

            elif expr.second_term.value:
                # case we have any swap signal
                for event in expr.second_term.triggered_events:
                    source_protocol = expr.second_term.atomic_source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=expr.second_term.triggered_events[0], receiver=self)
                    result = ready_signal.result
                    if event.signal_label == MessageType.SWAP_NEED:
                        # the swap node tell the leaf node that they need to swap the qubits to target through source
                        self.swap_source = result["source"]
                        self.swap_target = result["target"]
                        # check if the qubits are ready to swap
                        self.check_swap_ready()

                    elif event.signal_label == MessageType.SWAP_RESULT:
                        # apply the correction

                        success, m1, m2 = result["result"]
                        if success:
                            # apply corrections
                            yield self.apply_corrections(m1, m2, self.node.name, self.node.name)
                            # add the qubits to the entangled qubits
                            if self.swap_source in self.swapping_qubits:
                                target_node, mem_pos = self.swapping_qubits[self.swap_source]
                                self.entangled_qubits[target_node] = mem_pos
                                # remove the qubits from the swapping qubits
                                self.swapping_qubits.pop(self.swap_source)
                                self.swap_index += 1

                        else:
                            # re-entangle the qubits
                            _, mem_pos = self.swapping_qubits[self.swap_source]

                            self.send_signal("entangle", {"mem_pos": mem_pos,
                                                          "qmemory_name": f"{self.node.name}_qmemory"})
                        # reset the swap source and target
                        self.swap_source = None
                        self.swap_target = None
                        self.swap_ready = False

                # apply the correction
                # source_protocol = expr.second_term.atomic_source
                # ready_signal = source_protocol.get_signal_by_event(
                #     event=expr.second_term.triggered_events[0], receiver=self)
                # result = ready_signal.result
                # success, m1, m2 = result["result"]
                # if success:
                #     # apply corrections
                #     yield self.apply_corrections(m1, m2, self.node.name, self.node.name)
                #     # add the qubits to the entangled qubits
                #
                #     self.swap_index += 1
                # else:
                #     # re-entangle the qubits
                #     self.send_signal("entangle", {"mem_pos": 0,
                #                                   "qmemory_name": f"{self.node.name}_qmemory"})
            yield from self.handle_swapping(swap_node)
