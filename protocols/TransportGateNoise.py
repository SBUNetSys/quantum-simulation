import operator
from functools import reduce
from collections import defaultdict

import numpy as np
from netsquid import sim_time
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
import netsquid.qubits.operators as ops
import netsquid as ns
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.instructions import INSTR_CNOT, INSTR_H, INSTR_S, INSTR_X, INSTR_Z, INSTR_MEASURE

from protocols.MessageHandler import MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent
from utils import Logging
from utils.ClassicalMessages import ClassicalMessage
from protocols.Purification import Purification
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
    Hop-by-hop transport where each node applies corrections locally before forwarding.
    Gate operations (CNOT, H, S, X, Z, measure) run through gate_processor so that
    physical gate duration and depolarizing noise are correctly simulated.

    Same protocol logic as Transport.py; only the gate execution mechanism differs.
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
        self.is_source_node = node.name == source
        self.is_destination_node = node.name == destination
        self.entangled_node = entangled_node

        if is_after_security:
            await_signals = [self.await_signal(protocol, MessageType.SECURITY_TRANSPORT_QUBIT)
                             for protocol in qubit_ready_protocols]
        else:
            await_signals = []
            for protocol in qubit_ready_protocols:
                if type(protocol) is Purification:
                    await_signals.append(self.await_signal(protocol, MessageType.PURIFICATION_SUCCESS))
                elif type(protocol) is EntanglementHandlerConcurrent:
                    await_signals.append(self.await_signal(protocol, MessageType.ENTANGLED_SUCCESS))
                else:
                    await_signals.append(self.await_signal(protocol, Signals.SUCCESS))

        self.qubit_input_signal = reduce(operator.or_, await_signals)
        self.cc_message_handler = cc_message_handler
        self.logger = logger if logger is not None else Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        self.transmitting_qubit_size = int(transmitting_qubit_size)
        self.is_top_layer = is_top_layer

        self.entangled_qubits = defaultdict(dict)
        self.pending_transmission_qubits = defaultdict(dict)
        self.pending_re_entangled_qubits = defaultdict(dict)
        self.pending_transmission_operations = {}
        self.final_result = defaultdict(dict)
        self.pending_confirmation_operations = {}
        self.pending_need_apply_queue = defaultdict(list)
        self.sent_qubit_count = 0
        self.memory_mapping = {}
        self.memory_pos_mapping = {}
        self.start_time = sim_time()

        self.add_signal(MessageType.TRANSPORT_FINISHED)
        self.add_signal(MessageType.TRANSPORT_SUCCESS)

    def run(self):
        transport_signal = (
            self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_APPLY_CORRECTION) |
            self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS)
        )
        self.start_time = sim_time()
        while True:
            expr = yield self.qubit_input_signal | transport_signal

            if expr.first_term.value:
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(event=event, receiver=self)
                    except Exception as e:
                        self.logger.info(f"TransportGateNoise {self.name} -> Failed to parse signal: {e}", color="red")
                        continue
                    result = ready_signal.result
                    mem_pos = result.mem_pos
                    entangle_node = result.entangle_node
                    if type(result) is SwapEntangledSuccess:
                        self.memory_mapping[entangle_node] = result.actual_entangle_node
                        if entangle_node not in self.memory_pos_mapping:
                            self.memory_pos_mapping[entangle_node] = {}
                        self.memory_pos_mapping[entangle_node][mem_pos] = result.target_memo_pos
                    qmemory = self.get_qmemory(entangle_node)
                    if type(mem_pos) == list:
                        for pos in mem_pos:
                            self.entangled_qubits[entangle_node][pos] = True
                    else:
                        self.entangled_qubits[entangle_node][mem_pos] = True
                    yield from self.check_transport_ready()

            elif expr.second_term.value:
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(event=event, receiver=self)
                    result = ready_signal.result
                    if result.data.timestamp < self.start_time:
                        continue
                    if ready_signal.label == MessageType.TRANSPORT_APPLY_CORRECTION:
                        message: TransportApplyCorrectionMessageList = result.data
                        yield from self.handle_apply_correction_batch(message.operations)
                    elif ready_signal.label == MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS:
                        message: TransportApplySuccessMessage = result.data
                        self.handle_apply_correction_success(message)

            yield from self.check_transport_ready()

            if self.is_destination_node and \
                    len(self.final_result[self.entangled_node]) == self.transmitting_qubit_size:
                entangle_node = self.entangled_node
                if entangle_node in self.memory_mapping:
                    entangle_node = self.memory_mapping[entangle_node]
                self.send_signal(MessageType.TRANSPORT_FINISHED,
                                 {"entangle_node": entangle_node,
                                  "results": self.final_result[self.entangled_node]})
                break

    def check_transport_ready(self):
        if self.is_destination_node:
            self.process_transmit_queue()
            return
        transmission_ops = []
        if self.is_source_node and self.sent_qubit_count < self.transmitting_qubit_size:
            entangled_qubits_size = len(self.entangled_qubits[self.entangled_node])
            qubits_need_size = self.transmitting_qubit_size - self.sent_qubit_count
            for _ in range(min(entangled_qubits_size, qubits_need_size)):
                mem_pos, _ = self.entangled_qubits[self.entangled_node].popitem()
                if (self.entangled_node in self.memory_pos_mapping and
                        mem_pos in self.memory_pos_mapping[self.entangled_node]):
                    target_memo_pos = self.memory_pos_mapping[self.entangled_node][mem_pos]
                else:
                    target_memo_pos = mem_pos
                op_key = (self.node.name, mem_pos, self.entangled_node, target_memo_pos)
                transmission_ops.append(TransportOperation(
                    self.node.name, mem_pos, self.entangled_node, target_memo_pos, op_key))
                self.sent_qubit_count += 1
        for key in self.pending_transmission_qubits.keys():
            if len(self.pending_transmission_qubits[key]) > 0:
                need_size = len(self.pending_transmission_qubits[key])
                ready_size = len(self.entangled_qubits[self.entangled_node])
                for _ in range(min(need_size, ready_size)):
                    target_mem_pos, _ = self.entangled_qubits[self.entangled_node].popitem()
                    source_mem_pos, _ = self.pending_transmission_qubits[key].popitem()
                    op_key = (key, source_mem_pos, self.entangled_node, target_mem_pos)
                    transmission_ops.append(TransportOperation(
                        key, source_mem_pos, self.entangled_node, target_mem_pos, op_key))
        yield from self.start_transport_measurement(transmission_ops)
        pending = self.process_transmit_queue()
        if pending:
            yield from self.handle_apply_correction_batch(pending)

    def process_transmit_queue(self):
        ready = []
        remove_keys = []
        for key, messages in self.pending_need_apply_queue.items():
            node, mem_pos = key
            if node in self.entangled_qubits and mem_pos in self.entangled_qubits[node]:
                ready.extend(messages)
                remove_keys.append(key)
        for key in remove_keys:
            del self.pending_need_apply_queue[key]
        return ready

    def start_transport_measurement(self, transmission_ops: list):
        """
        Bell measurement for teleportation, routing gate operations through gate_processor
        so that physical instruction noise and delay are applied.
        """
        gate_proc = self.node.subcomponents[f"gate_processor_{self.node.name}"]
        measure_result_messages = defaultdict(list)

        for op in transmission_ops:
            if gate_proc.busy:
                yield self.await_program(gate_proc)

            if self.is_source_node:
                teleport_qubit = qapi.create_qubits(1)[0]
                qmemory_a = self.get_qmemory(op.target_node)
                if qmemory_a.busy:
                    yield self.await_program(qmemory_a)
                qubit_a, = qmemory_a.pop(op.source_mem_pos, skip_noise=False)

                gate_proc.put([teleport_qubit, qubit_a], positions=[0, 1])
                prog = QuantumProgram(num_qubits=2)
                q0, q1 = prog.get_qubit_indices(2)
                prog.apply(INSTR_H, [q0])
                prog.apply(INSTR_S, [q0])
                prog.apply(INSTR_CNOT, [q0, q1])
                prog.apply(INSTR_H, [q0])
                prog.apply(INSTR_MEASURE, [q0], output_key='m1')
                prog.apply(INSTR_MEASURE, [q1], output_key='m2')
                gate_proc.execute_program(prog)
                yield self.await_program(gate_proc)
                m1 = prog.output['m1'][0]
                m2 = prog.output['m2'][0]
                gate_proc.pop(0)
                gate_proc.pop(1)
            else:
                transport_mem = self.node.subcomponents[f"{self.node.name}_transport_qmemory"]
                if transport_mem.busy:
                    yield self.await_program(transport_mem)
                teleport_qubit, = transport_mem.pop(op.source_mem_pos, skip_noise=False)

                qmemory_a = self.get_qmemory(op.target_node)
                if qmemory_a.busy:
                    yield self.await_program(qmemory_a)
                qubit_a, = qmemory_a.pop(op.target_mem_pos, skip_noise=False)

                gate_proc.put([teleport_qubit, qubit_a], positions=[0, 1])
                prog = QuantumProgram(num_qubits=2)
                q0, q1 = prog.get_qubit_indices(2)
                prog.apply(INSTR_CNOT, [q0, q1])
                prog.apply(INSTR_H, [q0])
                prog.apply(INSTR_MEASURE, [q0], output_key='m1')
                prog.apply(INSTR_MEASURE, [q1], output_key='m2')
                gate_proc.execute_program(prog)
                yield self.await_program(gate_proc)
                m1 = prog.output['m1'][0]
                m2 = prog.output['m2'][0]
                gate_proc.pop(0)
                gate_proc.pop(1)

            self.pending_confirmation_operations[op.operation_key] = op
            measure_result_messages[op.target_node].append(TransportApplyCorrectionMessage(
                source_node=op.target_node,
                target_node=self.node.name,
                target_memo_pos=op.target_mem_pos,
                m1=m1,
                m2=m2,
                operation_key=op.operation_key
            ))

        for target_node, ops_list in measure_result_messages.items():
            self.cc_message_handler.send_message(
                MessageType.TRANSPORT_APPLY_CORRECTION,
                target_node,
                ClassicalMessage(
                    from_node=self.node.name,
                    to_node=target_node,
                    data=TransportApplyCorrectionMessageList(
                        source_node=target_node,
                        target_node=self.node.name,
                        operations=ops_list,
                    )
                ))

    def handle_apply_correction_batch(self, messages: list):
        """
        Apply Z/X corrections via gate_processor (with physical noise and delay),
        then store the corrected qubit in transport_qmemory for the next hop.
        """
        gate_proc = self.node.subcomponents[f"gate_processor_{self.node.name}"]
        success_ops = defaultdict(list)

        for message in messages:
            if (message.target_node not in self.entangled_qubits or
                    message.target_memo_pos not in self.entangled_qubits[message.target_node]):
                self.pending_need_apply_queue[(message.target_node, message.target_memo_pos)].append(message)
                continue

            qmemory = self.get_qmemory(message.target_node)
            if qmemory.busy:
                yield self.await_program(qmemory)
            qubit, = qmemory.pop(message.target_memo_pos, skip_noise=False)

            if message.m1 == 1 or message.m2 == 1:
                if gate_proc.busy:
                    yield self.await_program(gate_proc)
                gate_proc.put([qubit], positions=[0])
                corr_prog = QuantumProgram(num_qubits=1)
                q, = corr_prog.get_qubit_indices(1)
                if message.m1 == 1:
                    corr_prog.apply(INSTR_Z, [q])
                if message.m2 == 1:
                    corr_prog.apply(INSTR_X, [q])
                gate_proc.execute_program(corr_prog)
                yield self.await_program(gate_proc)
                qubit, = gate_proc.pop(0)

            success_ops[message.target_node].append(message.operation_key)
            self.pending_transmission_qubits[message.target_node][message.target_memo_pos] = True

            if self.is_destination_node:
                fid = qapi.fidelity(qubit, ns.y0)
                self.final_result[message.target_node][message.target_memo_pos] = fid
                self.logger.info(f"TransportGateNoise {self.name} -> received qubit, fid={fid:.4f}", color="blue")
                self.send_signal(MessageType.TRANSPORT_SUCCESS, {"results": fid})

            transport_memory = self.node.subcomponents[f"{self.node.name}_transport_qmemory"]
            transport_memory.put([qubit], positions=[message.target_memo_pos])

        for target_node, op_keys in success_ops.items():
            self.cc_message_handler.send_message(
                MessageType.TRANSPORT_APPLY_CORRECTION_SUCCESS,
                target_node,
                ClassicalMessage(
                    from_node=self.node.name,
                    to_node=target_node,
                    data=TransportApplySuccessMessage(operation_keys=op_keys)
                ))

    def handle_apply_correction_success(self, message: TransportApplySuccessMessage):
        for op_key in message.operation_keys:
            self.pending_confirmation_operations.pop(op_key, None)

    def get_qmemory(self, node_name):
        if node_name in self.memory_mapping:
            node_name = self.memory_mapping[node_name]
        return self.node.subcomponents[f"{node_name}_qmemory"]

    def reset(self):
        self.entangled_qubits = defaultdict(dict)
        self.pending_transmission_qubits = defaultdict(dict)
        self.pending_re_entangled_qubits = defaultdict(dict)
        self.pending_transmission_operations = {}
        self.pending_confirmation_operations = {}
        self.sent_qubit_count = 0
        self.pending_need_apply_queue = defaultdict(list)
        self.memory_mapping = {}
        self.final_result = defaultdict(dict)
        super().reset()

    def stop(self):
        super().stop()
