import operator
from functools import reduce
from collections import defaultdict

import numpy as np
from netsquid import sim_time
from netsquid.qubits import operators
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
import netsquid.qubits.operators as ops
from netsquid.components.instructions import INSTR_Z, INSTR_X

from protocols.MessageHandler import MessageType
from utils import Logging
from utils.ClassicalMessages import ClassicalMessage
from utils.SignalMessages import *


class Security(NodeProtocol):
    """
    security node protocol will do purification then verification and finally teleportation

    """

    def __init__(self, node,
                 name,
                 qubit_ready_protocols,
                 entangled_node,
                 source,
                 destination,
                 cc_message_handler,
                 logger,
                 transport_signal_protocol,
                 verification_signal_protocol,
                 is_top_layer=False):
        super().__init__(node=node, name=name)
        self.is_source_node = False
        self.is_destination_node = False
        if self.node.name == source:
            self.is_source_node = True
        if self.node.name == destination:
            self.is_destination_node = True
        # store next hop node name
        self.entangled_node = entangled_node
        # qubit input signal from lower layers, can be purification or verification
        await_signals = [self.await_signal(protocol, Signals.SUCCESS) for protocol in qubit_ready_protocols]
        # have expression to wait for ANY qubit input signal
        self.qubit_input_signal = reduce(operator.or_, await_signals)
        # classical message handler
        self.cc_message_handler = cc_message_handler
        self.entangled_qubits = {} # key = entangled_node val = [mem_pos]
        self.is_source = False
        if self.is_source_node:
            # wait the verification has been done
            self.verification_signal_protocol = verification_signal_protocol
            self.is_verified = False
            self.verify_need_count = 4
        self.verified_count = 0
        self.start_transport = False
        self.transport_signal_protocol = transport_signal_protocol
        self.start_time = None

        self.logger = logger
        # addd signal so we can send
        self.add_signal(MessageType.SECURITY_TRANSPORT_START)
        self.add_signal(MessageType.SECURITY_TRANSPORT_QUBIT)

    def run(self):

        self.logger.info(f"Security {self.name} -> Start transport protocol", color="cyan")
        self.start_time = sim_time()
        while True:
            if self.is_source_node:
                # case sending node
                expr = yield (self.qubit_input_signal |
                              self.await_signal(self.verification_signal_protocol, Signals.SUCCESS))
            else:
                expr = yield (self.qubit_input_signal |
                              self.await_signal(self.transport_signal_protocol, MessageType.SECURITY_TRANSPORT_START))
            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                    except Exception as e:
                        self.logger.info(f"Security {self.name} -> Failed to parse signal\n"
                                         f"Node: {self.node.name}\n"
                                         f"Error: {e}", color="red")
                        continue
                    result = ready_signal.result
                    mem_pos = result.mem_pos
                    entangle_node = result.entangle_node
                    if result.is_source:
                        self.is_source = True
                    self.logger.info(f"Security {self.name} -> Successful entangle signal\n"
                                f"Node: {self.node.name}\n"
                                f"Entangle node: {entangle_node}\n"
                                f"Mem: {mem_pos}\n", color="blue")
                    if entangle_node not in self.entangled_qubits:
                        self.entangled_qubits[entangle_node] = []
                    if type(mem_pos) == list:
                        for pos in mem_pos:
                            self.entangled_qubits[entangle_node].append(pos)
                    else:
                        self.entangled_qubits[entangle_node].append(mem_pos)
                    # TODO process entangle info
                    self.process_entangle_signal()
            elif expr.second_term.value:
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    if self.is_source_node:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                        result = ready_signal.result
                        if (isinstance(result, VerificationSuccessSignalMessage) and
                                result.timestamp < self.start_time):
                            continue
                    else:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self.transport_signal_protocol)
                    if ready_signal.label == MessageType.SECURITY_TRANSPORT_START:
                        # case of non start node
                        self.start_transport = True
                    else:
                        # case of verification
                        self.process_verification_signal(result)
            # process remaining info
            self.process_entangle_signal()

    def process_entangle_signal(self):
        """
        process entangle information from verification layer.
        if transport is not ready, we send signal.success. otherwise we send SECURITY_TRANSPORT_QUBIT.
        :return:
        """
        for entangled_node in self.entangled_qubits.keys():
            if len(self.entangled_qubits[entangled_node]) == 0:
                continue
            qubit_pos = self.entangled_qubits[entangled_node]
            self.entangled_qubits[entangled_node] = []
            # self.logger.info(f"Security {self.name} -> Sending signal to upper layer\n"
            #                  f"Node: {self.node.name}\n"
            #                  f"Entangle node: {entangled_node}\n"
            #                  f"Mems: {qubit_pos}\n", color="green")
            message = SecuritySuccessSignalMessage(self.node.name, entangled_node, self.is_source, qubit_pos)
            if not self.start_transport:
                # this should be caught by end to end
                self.send_signal(Signals.SUCCESS, message)
            else:
                # this should be caught by transport
                self.send_signal(MessageType.SECURITY_TRANSPORT_QUBIT, message)

    def process_verification_signal(self, message:VerificationSuccessSignalMessage):
        """
        process verification signal after security layer
        """
        self.logger.info(f"Security {self.name} -> Successful verification signal\n"
                         f"Node: {self.node.name}\n", color="green")
        self.verified_count += len(message.mem_pos)
        if self.verified_count == self.verify_need_count:
            # finished the verification process
            self.start_transport = True
            # send signal such that all other node can start
            self.send_signal(MessageType.SECURITY_TRANSPORT_START, True)

    def reset(self):
        self.start_transport = False
        self.entangled_qubits = {}
        self.verified_count = 0
        super().reset()