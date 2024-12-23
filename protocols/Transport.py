import operator
from functools import reduce
from collections import defaultdict

import numpy as np
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


class TransportOperation:
    def __init__(self, source_node, source_mem_pos, target_node, target_mem_pos):
        self.source_node = source_node
        self.source_mem_pos = source_mem_pos
        self.target_node = target_node
        self.target_mem_pos = target_mem_pos


class Transportation(NodeProtocol):
    """
    A protocol that send qubits in hop by hop setting.

    For example
    A -> B -> C

    A will teleport qubit in to B
    B will teleport qubit in to C
    """
    """
        Initialize the protocol
        :param node: the node that the protocol is attached to
        :param name: the name of the protocol
        :param qubit_ready_protocols: the lower layers that will send the qubit ready signal
         (i.e purfication or verification)
        :param entangled_node: the out going edge for this node, i.e A -> B -> C, A entangled B, B entangled C
        :param source: the source node name
        :param destination: the destination node name
        :param cc_message_handler: the classical message handler
        :param transmitting_qubit_size: the number of qubits needs to be transmitted
        :param logger: the logger
        :param is_top_layer: if we are the top layer of the simulation
    """

    def __init__(self, node,
                 name,
                 qubit_ready_protocols,
                 entangled_node,
                 source,
                 destination,
                 cc_message_handler,
                 transmitting_qubit_size,
                 logger,
                 is_top_layer=False):
        super().__init__(node=node, name=name)
        # identify role of the node
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
        # keep track of the node name and actual memory name
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger
        self.transmitting_qubit_size = int(transmitting_qubit_size)
        self.is_top_layer = is_top_layer

        # variable for transport use
        self.entangled_qubits = defaultdict(dict)  # key: entangled_node(in_node), value = {mem_pos: fid}
        # qubits that needs to be transport to next hop
        self.pending_transmission_qubits = defaultdict(dict)  # key: entangled_node(in_node), value = {mem_pos: fid}
        self.pending_re_entangled_qubits = defaultdict(dict)  # key: entangled_node, value = {mem_pos:true}
        self.pending_transmission_operations = {}  # key: operation_key: TransportOperation
        # pending apply correct tion success ops
        self.pending_confirmation_operations = {}  # key: operation_key: TransportOperation
        self.sent_qubit_count = 0
        self.transport_need_queue = {}  # key(entangle_node, mem_pos): TransportRequestMessage

    def run(self):
        """
        Run the protocol
        :return:
        """
        transport_signal = (self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_REQUEST) |
                            self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_READY) |
                            self.await_signal(self.cc_message_handler,
                                              signal_label=MessageType.TRANSPORT_APPLY_CORRECTION) |
                            self.await_signal(self.cc_message_handler,
                                              signal_label=MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS)
                            )
        self.logger.info(f"Transport {self.name} -> Start transport protocol", color="cyan")
        while True:
            # handle other operations first then we try to perform the transport operation
            expr = yield self.qubit_input_signal | transport_signal

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
                    self.logger.info(f"Transport {self.name} -> Qubit Ready signal from {entangle_node}\n"
                                     f"entangle_node {entangle_node}\n"
                                     f"mem_pos: {mem_pos}",
                                     color="blue")
                    # this is used incase we have verification, they come in batches
                    if type(mem_pos) == list:
                        for pos in mem_pos:
                            self.entangled_qubits[entangle_node][pos] = True
                    else:
                        self.entangled_qubits[entangle_node][mem_pos] = True
                    self.check_transport_ready()
            elif expr.second_term.value:
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if ready_signal.label == MessageType.TRANSPORT_REQUEST:
                        # the left node need to start transport request to the next hop
                        message: TransportRequestMessage = result.data
                        self.logger.info(f"Transport {self.name} -> Transport need signal\n"
                                         f"From: {result.from_node}\n"
                                         f"Source: {message.source_node}\n"
                                         f"Target: {message.target_node}\n"
                                         f"Mem Pos: {message.target_memo_pos}", color="yellow")
                        self.handle_transport_need(message)
                    elif ready_signal.label == MessageType.TRANSPORT_READY:
                        # we are ready to teleport
                        message: TransportResponseMessage = result.data
                        self.logger.info(f"Transport {self.name} -> Transport ready signal\n"
                                         f"From: {result.from_node}\n"
                                         f"Operation Key: {message.operation_key}\n", color="green")
                        yield from self.handle_transport_ready(message)
                    elif ready_signal.label == MessageType.TRANSPORT_APPLY_CORRECTION:
                        message: TransportApplyCorrectionMessage = result.data
                        self.logger.info(f"Transport {self.name} -> Transport Apply Correction\n"
                                         f"From: {result.from_node}\n"
                                         f"Source: {message.source_node}\n"
                                         f"Target: {message.target_node}\n"
                                         f"Mem Pos: {message.target_memo_pos}\n", color="yellow")
                        yield from self.handle_apply_correction(message)
                    elif ready_signal.label == MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS:
                        message: TransportApplySuccessMessage = result.data
                        self.logger.info(f"Transport {self.name} -> Transport Apply Correction Success\n"
                                         f"From: {result.from_node}\n"
                                         f"Operation Key: {message.operation_key}\n",
                                         color="green")
                        # TODO finish the re-entangle logic
                        self.handle_apply_correction_success(message)

            self.check_transport_ready()

            if self.is_destination_node and \
                len(self.pending_transmission_qubits[self.entangled_node]) == self.transmitting_qubit_size:
                # we finish the final entanglement
                self.send_signal(Signals.SUCCESS,
                                 {"entangle_node": self.entangled_node,
                                  "mem_poses":[x for x in self.pending_transmission_qubits[self.entangled_node]]})
                break

    def check_transport_ready(self):
        """
        check if we can transport qubits
        if we are the sending node, we will start a transport operation.
        after we received the confirmation from remote node, we will generate a qubit and teleport to the node

        if we are not the sending node, i.e forwarding nodes. We will check if we have entangled pairs ready
        if we do, we start a transport operation. After receiving the confirmation from remote node, we will
        teleport the qubit from pending_transmission_qubits
        :return:
        """
        # destination node does not forward anything
        if self.is_destination_node:
            return
        # case we are the sending node
        if self.is_source_node and self.sent_qubit_count < self.transmitting_qubit_size:
            entangled_qubits = len(self.entangled_qubits[self.entangled_node])
            for _ in range(entangled_qubits):
                mem_pos, _ = self.entangled_qubits[self.entangled_node].popitem()
                op = TransportOperation(self.node.name, mem_pos, self.entangled_node, mem_pos)
                op_key = (self.node.name, mem_pos, self.entangled_node, mem_pos)
                self.pending_transmission_operations[op_key] = op
                self.cc_message_handler.send_message(MessageType.TRANSPORT_REQUEST,
                                                     self.entangled_node,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=self.entangled_node,
                                                         data=TransportRequestMessage(
                                                             source_node=self.entangled_node,
                                                             target_node=self.node.name,
                                                             target_memo_pos=mem_pos,
                                                             operation_key=op_key,
                                                         )
                                                     ))
                self.sent_qubit_count += 1
        # case we are the middle node
        for key in self.pending_transmission_qubits.keys():
            if len(self.pending_transmission_qubits[key]) > 0:
                need_transmission_qubit_size = len(self.pending_transmission_qubits[key])
                ready_qubit_size = len(self.entangled_qubits[self.entangled_node])
                # send all possible qubit request
                for _ in range(min(need_transmission_qubit_size, ready_qubit_size)):
                    target_mem_pos, _ = self.entangled_qubits[self.entangled_node].popitem()
                    source_mem_pos, _ = self.pending_transmission_qubits[key].popitem()
                    op = TransportOperation(key, source_mem_pos, self.entangled_node, target_mem_pos)
                    op_key = (key, source_mem_pos, self.entangled_node, target_mem_pos)
                    self.pending_transmission_operations[op_key] = op
                    self.cc_message_handler.send_message(MessageType.TRANSPORT_REQUEST,
                                                         self.entangled_node,
                                                         ClassicalMessage(
                                                             from_node=self.node.name,
                                                             to_node=self.entangled_node,
                                                             data=TransportRequestMessage(
                                                                 source_node=self.entangled_node,
                                                                 target_node=self.node.name,
                                                                 target_memo_pos=target_mem_pos,
                                                                 operation_key=op_key,
                                                             )))

        # now we need to check if we have pending request or not, if we do we need to send the ready signal
        self.process_transmit_queue()

    def process_transmit_queue(self):
        for key in self.entangled_qubits[self.entangled_node].keys():
            if (self.entangled_node, key) in self.transport_need_queue:
                message = self.transport_need_queue[(self.entangled_node, key)]
                self.cc_message_handler.send_message(MessageType.TRANSPORT_READY,
                                                     message.target_node,
                                                     ClassicalMessage(
                                                         from_node=self.node.name,
                                                         to_node=message.target_node,
                                                         data=TransportResponseMessage(
                                                             operation_key=message.operation_key
                                                         )
                                                     ))
                del self.transport_need_queue[(self.entangled_node, key)]

    def handle_transport_need(self, message: TransportRequestMessage):
        """
        handle transport need signal from source node
        :param message:  TransportRequestMessage
        :return:
        """
        if message.target_node in self.entangled_qubits and \
                message.target_memo_pos in self.entangled_qubits[message.target_node]:
            # case we are ready, we need to send the ready signal
            self.cc_message_handler.send_message(MessageType.TRANSPORT_READY,
                                                 message.target_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=message.target_node,
                                                     data=TransportResponseMessage(
                                                         operation_key=message.operation_key
                                                     )
                                                 ))
        else:
            # append to queue for later process when
            self.transport_need_queue[(message.target_node, message.target_memo_pos)] = message

    def handle_transport_ready(self, message: TransportResponseMessage):
        """
        handle transport response from remote node saying ready to teleport.
        based on the operation key, we will either generate a qubit or teleport from pending transmission qubit.
        then we will send APPLY_CORRECTION to remote node.
        :param message:  TransportResponseMessage
        :return:
        """
        op: TransportOperation = self.pending_transmission_operations[message.operation_key]
        if self.is_source_node:
            # case we are the source node, we need generate a qubit to perform teleportation
            teleport_qubit = qapi.create_qubits(1)[0]
            # turn in to y0 state
            qapi.operate(teleport_qubit, ops.H)
            qapi.operate(teleport_qubit, ops.S)
        else:
            qmemory = self.get_qmemory(f"{op.source_node}_qmemory")
            if qmemory.busy:
                yield self.await_program(qmemory)
            teleport_qubit, = qmemory.pop(op.source_mem_pos, skip_noise=False)

        qmemory_a = self.get_qmemory(f"{op.target_node}_qmemory")
        if qmemory_a.busy:
            yield self.await_program(qmemory_a)
        qubit_a, = qmemory_a.pop(op.target_mem_pos, skip_noise=False)

        # perform teleport measurement
        qapi.operate(qubits=[teleport_qubit, qubit_a], operator=ops.CNOT)
        qapi.operate(teleport_qubit, ops.H)
        m1, _ = qapi.measure(teleport_qubit)
        m2, _ = qapi.measure(qubit_a)
        # add to pending confirmation stack
        self.pending_confirmation_operations[message.operation_key] = op
        # remove from pending transmission stack
        del self.pending_transmission_operations[message.operation_key]
        # now send the measurement to next hop to apply correction
        self.cc_message_handler.send_message(MessageType.TRANSPORT_APPLY_CORRECTION,
                                             op.target_node,
                                             ClassicalMessage(
                                                 from_node=self.node.name,
                                                 to_node=op.target_node,
                                                 data=TransportApplyCorrectionMessage(
                                                     source_node=op.target_node,
                                                     target_node=self.node.name,
                                                     target_memo_pos=op.target_mem_pos,
                                                     m1=m1,
                                                     m2=m2,
                                                     operation_key=message.operation_key
                                                 )
                                             ))

    def handle_apply_correction(self, message: TransportApplyCorrectionMessage):
        """
        handle apply correction from source node to finish the teleport operation.
        :param message: TransportApplyCorrectionMessage
        :return:
        """
        self.logger.info(
            f"Transport {self.name} received apply correction\n"
            f"Operation: {message.operation_key}\n"
            f"Source node: {message.source_node}\n"
            f"Target node: {message.target_node}\n"
            f"Target memo pos: {message.target_memo_pos}\n", color="yellow"
        )
        qmemory = self.get_qmemory(f"{message.target_node}_qmemory")

        if message.m1 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            self.logger.info(
                f"Transport {self.name} -> Apply correction Z\n"
                f"Qubit {message.target_memo_pos} with qmem_name: {qmemory.name}",
                color="green")
            qmemory.execute_instruction(INSTR_Z, [message.target_memo_pos])
        if message.m2 == 1:
            if qmemory.busy:
                yield self.await_program(qmemory)
            self.logger.info(
                f"Transport {self.name} -> Apply correction X\n"
                f"Qubit {message.target_memo_pos} with qmem_name: {qmemory.name}",
                color="green")
            qmemory.execute_instruction(INSTR_X, [message.target_memo_pos])

        self.cc_message_handler.send_message(MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS,
                                             message.target_node,
                                             ClassicalMessage(
                                                 from_node=self.node.name,
                                                 to_node=message.target_node,
                                                 data=TransportApplySuccessMessage(
                                                     operation_key=message.operation_key
                                                 )
                                             ))
        # add the qubit to pending transmission
        self.pending_transmission_qubits[message.target_node][message.target_memo_pos] = True

    def handle_apply_correction_success(self, message: TransportApplySuccessMessage):
        """
        handle apply success from source node to finish the teleport operation.
        TODO: regenerate the qubits
        :param message: TransportApplySuccessMessage
        :return:
        """
        del self.pending_confirmation_operations[message.operation_key]
        # TODO add logic for re-entangle the old pair as we are done with teleportation

    def get_qmemory(self, memory_name):
        """
        Get the quantum memory of the node
        :param memory_name:
        :return:
        """
        return self.node.subcomponents[memory_name]
    def reset(self):
        self.entangled_qubits = defaultdict(dict)  # key: entangled_node(in_node), value = {mem_pos: fid}
        # qubits that needs to be transport to next hop
        self.pending_transmission_qubits = defaultdict(dict)  # key: entangled_node(in_node), value = {mem_pos: fid}
        self.pending_re_entangled_qubits = defaultdict(dict)  # key: entangled_node, value = {mem_pos:true}
        self.pending_transmission_operations = {}  # key: operation_key: TransportOperation
        # pending apply correct tion success ops
        self.pending_confirmation_operations = {}  # key: operation_key: TransportOperation
        self.sent_qubit_count = 0
        self.transport_need_queue = {}  # key(entangle_node, mem_pos): TransportRequestMessage
        super().reset()