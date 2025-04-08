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


class TransportOperation:
    def __init__(self, source_node, source_mem_pos, target_node, target_mem_pos, op_key):
        self.source_node = source_node
        self.source_mem_pos = source_mem_pos
        self.target_node = target_node
        self.target_mem_pos = target_mem_pos
        self.operation_key = op_key


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
                 is_top_layer=False,
                 is_after_security=False):
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
        if is_after_security:
            # if we have security we need to await different signal
            await_signals = [self.await_signal(protocol, MessageType.SECURITY_TRANSPORT_QUBIT)
                             for protocol in qubit_ready_protocols]
        else:
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
        self.final_result = defaultdict(dict) # key:entangle_node, value = {mem_pos: fid}
        # pending apply correction success ops
        self.pending_confirmation_operations = {}  # key: operation_key: TransportOperation
        # pending apply correction i.e transport start
        self.pending_need_apply_queue = {}  # key(entangle_node, mem_pos): TransportApplyCorrectionMessage
        self.sent_qubit_count = 0
        self.memory_mapping = {} # key entangled_node value: actual entangled_name
        self.memory_pos_mapping = {} # key entangle_node, value = {mem_pos: remote_memo_pos)}
        self.start_time = sim_time()

        self.add_signal(MessageType.TRANSPORT_FINISHED)
        self.add_signal(MessageType.TRANSPORT_SUCCESS)

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
        self.start_time = sim_time()
        while True:
            # handle other operations first then we try to perform the transport operation
            expr = yield self.qubit_input_signal | transport_signal

            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    except Exception as e:
                        self.logger.info(f"Transport {self.name} -> Failed to parse signal\n"
                                         f"Node: {self.node.name}\n"
                                         f"Error: {e}", color="red")
                        continue
                    # result -> EntangleSignalMessage (Maybe Purification or Verification)
                    result = ready_signal.result
                    mem_pos = result.mem_pos
                    entangle_node = result.entangle_node
                    if type(result) is SwapEntangledSuccess:
                        self.logger.info(f"Transport {self.name} -> Qubit Ready signal from e2e {entangle_node}\n"
                                         f"entangle_node {entangle_node}\n"
                                         f"mem_pos: {mem_pos}\n"
                                         f"actual_node: {result.actual_entangle_node}\n"
                                         f"remote_mem_pos: {result.target_memo_pos}\n"
                                         f"time: {sim_time()}",
                                         color="blue")
                        self.memory_mapping[entangle_node] = result.actual_entangle_node
                        if entangle_node not in self.memory_pos_mapping:
                            self.memory_pos_mapping[entangle_node] = {}
                        self.memory_pos_mapping[entangle_node][mem_pos] = result.target_memo_pos
                    # this is used incase we have verification, they come in batches
                    qmemory = self.get_qmemory(entangle_node)
                    if type(mem_pos) == list:
                        for pos in mem_pos:
                            self.entangled_qubits[entangle_node][pos] = True
                            qubit_a, = qmemory.peek(pos, skip_noise=False)
                            self.logger.info(
                                f"Transport {self.name} Received Entangled Qubit\n"
                                f"Target node: {entangle_node}\n"
                                f"Target memo pos: {pos}\n"
                                f"Source QState: {qubit_a.qstate}", color="green"
                            )
                    else:
                        self.entangled_qubits[entangle_node][mem_pos] = True
                        qubit_a, = qmemory.peek(mem_pos, skip_noise=False)
                        self.logger.info(
                            f"Transport {self.name} Received Entangled Qubit\n"
                            f"Target node: {entangle_node}\n"
                            f"Target memo pos: {mem_pos}\n"
                            f"Source QState: {qubit_a.qstate}", color="green"
                        )
                    yield from self.check_transport_ready()
            elif expr.second_term.value:
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if result.data.timestamp < self.start_time:
                        continue
                    if ready_signal.label == MessageType.TRANSPORT_APPLY_CORRECTION:
                        message: TransportApplyCorrectionMessageList = result.data
                        self.logger.info(f"Transport {self.name} -> Transport Apply Correction\n"
                                         f"From: {result.from_node}\n"
                                         f"Source: {message.source_node}\n"
                                         f"Target: {message.target_node}\n"
                                         f"Ops Count: {len(message.operations)}\n", color="yellow")
                        yield from self.handle_apply_correction_batch(message.operations)
                    elif ready_signal.label == MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS:
                        message: TransportApplySuccessMessage = result.data
                        self.logger.info(f"Transport {self.name} -> Transport Apply Correction Success\n"
                                         f"From: {result.from_node}\n"
                                         f"Operation Keys: {message.operation_keys}\n",
                                         color="green")
                        # TODO finish the re-entangle logic
                        self.handle_apply_correction_success(message)

            yield from self.check_transport_ready()

            if self.is_destination_node and \
                len(self.final_result[self.entangled_node]) == self.transmitting_qubit_size:
                # we finish the final entanglement
                entangle_node = self.entangled_node
                if entangle_node in self.memory_mapping:
                    entangle_node = self.memory_mapping[entangle_node]
                self.send_signal(MessageType.TRANSPORT_FINISHED,
                                 {"entangle_node": entangle_node,
                                  "results":self.final_result[self.entangled_node],})
                break
            # if self.is_source_node and self.sent_qubit_count == self.transmitting_qubit_size:
            #     # we finished all transmission, and we wait X times then sent signa
            #     # this will allow us to capture the transmission success rate
            #     yield self.await_timer(65000)
            #     self.send_signal(MessageType.TRANSPORT_FINISHED, {"finished_time": sim_time()})
            #     break

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
            self.process_transmit_queue()
            return
        # operations we need to process
        transmission_ops = []
        # case we are the sending node
        if self.is_source_node and self.sent_qubit_count < self.transmitting_qubit_size:
            entangled_qubits_size = len(self.entangled_qubits[self.entangled_node])
            qubits_need_size = self.transmitting_qubit_size - self.sent_qubit_count
            # we only can do min of the condition, ether we have more qubits entangled than we will need
            # to transmit. Or vice versa
            for _ in range(min(entangled_qubits_size, qubits_need_size)):
                mem_pos, _ = self.entangled_qubits[self.entangled_node].popitem()
                if (self.entangled_node in self.memory_pos_mapping and
                        mem_pos in self.memory_pos_mapping[self.entangled_node]):
                    target_memo_pos = self.memory_pos_mapping[self.entangled_node][mem_pos]
                else:
                    target_memo_pos = mem_pos
                op_key = (self.node.name, mem_pos, self.entangled_node, target_memo_pos)
                op = TransportOperation(self.node.name, mem_pos, self.entangled_node, target_memo_pos, op_key)
                transmission_ops.append(op)
                self.sent_qubit_count += 1
                self.logger.info(f"Transport {self.name} sending qubits\n"
                                 f"Qubit Sent: {self.sent_qubit_count}\n"
                                 f"Qubit Need Transmission: {self.transmitting_qubit_size}\n",
                                 color="cyan")
        # case we are the middle node
        for key in self.pending_transmission_qubits.keys():
            if len(self.pending_transmission_qubits[key]) > 0:
                need_transmission_qubit_size = len(self.pending_transmission_qubits[key])
                ready_qubit_size = len(self.entangled_qubits[self.entangled_node])
                # send all possible qubit request
                for _ in range(min(need_transmission_qubit_size, ready_qubit_size)):
                    target_mem_pos, _ = self.entangled_qubits[self.entangled_node].popitem()
                    source_mem_pos, _ = self.pending_transmission_qubits[key].popitem()
                    op_key = (key, source_mem_pos, self.entangled_node, target_mem_pos)
                    op = TransportOperation(key, source_mem_pos, self.entangled_node, target_mem_pos,op_key)
                    transmission_ops.append(op)
        # we do batch measurement, avoid sending multiple classical message
        yield from self.start_transport_measurement(transmission_ops)
        # now we need to check if we have pending apply correction if we do, we need process them
        process_correction = self.process_transmit_queue()
        if len(process_correction) > 0:
            yield from self.handle_apply_correction_batch(process_correction)

    def process_transmit_queue(self):
        remove_message = []
        correction_message = []
        for key in self.pending_need_apply_queue.keys():
            node = key[0]
            mem_pos = key[1]
            if node in self.entangled_qubits and mem_pos in self.entangled_qubits[node]:
                message = self.pending_need_apply_queue[key]
                correction_message.append(message)
                remove_message.append(key)
        for key in remove_message:
            self.pending_need_apply_queue.pop(key)
        return correction_message


    def start_transport_measurement(self, transmission_ops: list):
        """
        handle transport response from remote node saying ready to teleport.
        based on the operation key, we will either generate a qubit or teleport from pending transmission qubit.
        then we will send APPLY_CORRECTION to remote node.
        :param transmission_ops:  list of transmission operations
        :return:
        """
        measure_result_messages = defaultdict(list)
        for op in transmission_ops:
            if self.is_source_node:
                # case we are the source node, we need generate a qubit to perform teleportation
                teleport_qubit = qapi.create_qubits(1)[0]
                # turn in to y0 state
                qapi.operate(teleport_qubit, ops.H)
                qapi.operate(teleport_qubit, ops.S)

                qmemory_a = self.get_qmemory(op.target_node)
                if qmemory_a.busy:
                    yield self.await_program(qmemory_a)
                qubit_a, = qmemory_a.pop(op.source_mem_pos, skip_noise=False)
                self.logger.info(
                    f"Transport {self.name} Start Teleporting Qubit as Sender\n"
                    f"Operation: {op.operation_key}\n"
                    f"Target node: {op.target_node}\n"
                    f"Target memo pos: {op.source_mem_pos}\n"
                    # f"QState: {qubit_a.qstate}"
                    , color="green"
                )
            else:
                # qmemory = self.get_qmemory(op.source_node)
                qmemory = self.node.subcomponents[f"{self.node.name}_transport_qmemory"]
                if qmemory.busy:
                    yield self.await_program(qmemory)
                teleport_qubit, = qmemory.pop(op.source_mem_pos, skip_noise=False)

                qmemory_a = self.get_qmemory(op.target_node)
                if qmemory_a.busy:
                    yield self.await_program(qmemory_a)
                qubit_a, = qmemory_a.pop(op.target_mem_pos, skip_noise=False)
                self.logger.info(
                    f"Transport {self.name} Start Teleporting Qubit Measurement as Middle Node\n"
                    f"Operation: {op.operation_key}\n"
                    f"Source Node: {op.source_node}\n"
                    f"Source memo pos: {op.source_mem_pos}\n"
                    f"Target node: {op.target_node}\n"
                    f"Target memo pos: {op.target_mem_pos}\n"
                    # f"QState: {qubit_a.qstate}"
                    , color="green"
                )
            # perform teleport measurement
            qapi.operate(qubits=[teleport_qubit, qubit_a], operator=ops.CNOT)
            qapi.operate(teleport_qubit, ops.H)
            m1, _ = qapi.measure(teleport_qubit)
            m2, _ = qapi.measure(qubit_a)
            # add to pending confirmation stack
            self.pending_confirmation_operations[op.operation_key] = op
            measure_result_messages[op.target_node].append(TransportApplyCorrectionMessage(
                                                     source_node=op.target_node,
                                                     target_node=self.node.name,
                                                     target_memo_pos=op.target_mem_pos,
                                                     m1=m1,
                                                     m2=m2,
                                                     operation_key=op.operation_key
                                                 ))

        # now send the measurement to next hop to apply correction
        # we did target nodes to have feature compatibility incase we have more connected node
        for target_node in measure_result_messages.keys():
            self.cc_message_handler.send_message(MessageType.TRANSPORT_APPLY_CORRECTION,
                                                 target_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=target_node,
                                                     data=TransportApplyCorrectionMessageList(
                                                         source_node=target_node,
                                                         target_node=self.node.name,
                                                         operations=measure_result_messages[target_node],
                                                     )
                                                 ))

    def handle_apply_correction_batch(self, messages: list):
        """
        handle apply correction from source node to finish the teleport operation.
        :param messages: list of TransportApplyCorrectionMessage
        :return:
        """
        success_ops = defaultdict(list)
        for message in messages:

            # check if we have the position or not
            if message.target_node not in self.entangled_qubits or \
                message.target_memo_pos not in self.entangled_qubits[message.target_node]:
                self.logger.info(
                    f"Transport {self.name} received apply correction, node is not ready\n"
                    f"Operation: {message.operation_key}\n"
                    f"Source node: {message.source_node}\n"
                    f"Target node: {message.target_node}\n"
                    f"Target memo pos: {message.target_memo_pos}\n", color="blue"
                )
                # append to pending apply correction q
                self.pending_need_apply_queue[(message.target_node, message.target_memo_pos)].append(message)
                continue
            self.logger.info(
                f"Transport {self.name} received apply correction\n"
                f"Operation: {message.operation_key}\n"
                f"Source node: {message.source_node}\n"
                f"Target node: {message.target_node}\n"
                f"Target memo pos: {message.target_memo_pos}\n", color="yellow"
            )

            qmemory = self.get_qmemory(message.target_node)
            if qmemory.busy:
                yield self.await_program(qmemory)
            qubit, = qmemory.pop(message.target_memo_pos, skip_noise=False)
            if message.m1 == 1:
                self.logger.info(
                    f"Transport {self.name} -> Apply correction Z\n"
                    f"Qubit {message.target_memo_pos} with qmem_name: {qmemory.name}",
                    color="green")
                qapi.operate(qubit, ops.Z)
            if message.m2 == 1:
                self.logger.info(
                    f"Transport {self.name} -> Apply correction X\n"
                    f"Qubit {message.target_memo_pos} with qmem_name: {qmemory.name}",
                    color="green")
                qapi.operate(qubit, ops.X)
            success_ops[message.target_node].append(message.operation_key)

            # if message.m1 == 1:
            #     if qmemory.busy:
            #         yield self.await_program(qmemory)
            #     self.logger.info(
            #         f"Transport {self.name} -> Apply correction Z\n"
            #         f"Qubit {message.target_memo_pos} with qmem_name: {qmemory.name}",
            #         color="green")
            #     qmemory.execute_instruction(INSTR_Z, [message.target_memo_pos])
            # if message.m2 == 1:
            #     if qmemory.busy:
            #         yield self.await_program(qmemory)
            #     self.logger.info(
            #         f"Transport {self.name} -> Apply correction X\n"
            #         f"Qubit {message.target_memo_pos} with qmem_name: {qmemory.name}",
            #         color="green")
            #     qmemory.execute_instruction(INSTR_X, [message.target_memo_pos])

            # add the qubit to pending transmission
            self.pending_transmission_qubits[message.target_node][message.target_memo_pos] = True
            if self.is_destination_node:
                # pop and compare the result
                # qubit, = qmemory.pop(message.target_memo_pos, skip_noise=False)
                fid = qapi.fidelity(qubit, ns.y0)
                self.final_result[message.target_node][message.target_memo_pos] = fid
                # if 0 < fid < 0.99:
                #     print("Hi")
                self.logger.info(f"Transport {self.name} -> received Qubits\n"
                                 f"Current Qubits Received {len(self.final_result[self.entangled_node])}\n"
                                 f"Target Qubits Needed {self.transmitting_qubit_size}\n", color="green")
                self.send_signal(MessageType.TRANSPORT_SUCCESS,
                                 {"results": self.final_result[self.entangled_node],})
            transport_memory = self.node.subcomponents[f"{self.node.name}_transport_qmemory"]
            transport_memory.put(qubit, message.target_memo_pos)

        # send success message feedback to the correct target node
        for target_node in success_ops.keys():
            self.cc_message_handler.send_message(MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS,
                                                 target_node,
                                                 ClassicalMessage(
                                                     from_node=self.node.name,
                                                     to_node=target_node,
                                                     data=TransportApplySuccessMessage(
                                                        operation_keys=success_ops[target_node]
                                                     )
                                                 ))
    def handle_apply_correction_success(self, message: TransportApplySuccessMessage):
        """
        handle apply success from source node to finish the teleport operation.
        TODO: regenerate the qubits
        :param message: TransportApplySuccessMessage
        :return:
        """
        for op_key in message.operation_keys:
            if op_key not in self.pending_confirmation_operations:
                continue
            del self.pending_confirmation_operations[op_key]
        # TODO add logic for re-entangle the old pair as we are done with teleportation

    def get_qmemory(self, node_name):
        """
        Get the quantum memory of the node
        :param node_name: the name of the node we want to get the quantum memory
        :return:
        """
        if node_name in self.memory_mapping:
            node_name = self.memory_mapping[node_name]
        memory_name = f"{node_name}_qmemory"
        return self.node.subcomponents[memory_name]

    def reset(self):
        # broadcast the verification finished signal to lower layer
        self.logger.info(f"Transport {self.name} -> Resetting Transport, sending message", color="red")

        self.entangled_qubits = defaultdict(dict)  # key: entangled_node(in_node), value = {mem_pos: fid}
        # qubits that needs to be transport to next hop
        self.pending_transmission_qubits = defaultdict(dict)  # key: entangled_node(in_node), value = {mem_pos: fid}
        self.pending_re_entangled_qubits = defaultdict(dict)  # key: entangled_node, value = {mem_pos:true}
        self.pending_transmission_operations = {}  # key: operation_key: TransportOperation
        # pending apply correct tion success ops
        self.pending_confirmation_operations = {}  # key: operation_key: TransportOperation
        self.sent_qubit_count = 0
        self.pending_need_apply_queue = {}  # key(entangle_node, mem_pos): TransportRequestMessage
        self.memory_mapping = {}
        self.final_result = defaultdict(dict)
        super().reset()

    def stop(self):
        self.logger.info(f"Transport {self.name} -> Stopping Transport, sending message", color="red")
        super().stop()