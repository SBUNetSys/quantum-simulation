import copy
import gc

import numpy as np
from netsquid.qubits import measure
from netsquid.util.simtools import sim_time
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_CNOT, INSTR_H
import netsquid.qubits.operators as ops
import netsquid.qubits.qubitapi as qapi
from numpy.lib.utils import source

from protocols.MessageHandler import MessageType
from utils import Logging, SignalMessages
from utils.ClassicalMessages import ClassicalMessage
from utils.Gates import controlled_unitary, measure_operator


class SecurityVerification(NodeProtocol):
    """
    Verification build exclusively for Security Protocol
    Since we have end to end as lower layer, we need to make proper adjustment to account
    missmatch with memory pos and memory name
    """

    def __init__(self, node, name,
                 entangled_node,
                 target_node,
                 purification_protocol,
                 cc_message_handler,
                 m_size,
                 batch_size,
                 is_top_layer=False,
                 max_entangled_pairs=2,
                 CU_Gate=None,
                 CCU_Gate=None,
                 measurement_m0=None,
                 measurement_m1=None,
                 logger=None):
        super().__init__(node=node, name=name)

        self.purification_protocol = purification_protocol
        self.cc_message_handler = cc_message_handler
        # mapping of entangled qubits to memory positions key: entangled_node, value:{self mem pos: target_mem_pos}
        self.entangled_qubits = {}
        self.entangled_node = entangled_node
        self.max_entangle_pairs = max_entangled_pairs - 2  # mem_pos 0 always is temp qubit, one from the purification
        # TODO: Can not just use memory positions, because purification does not guarantee the same memory positions
        #  for the same entangled qubits. We need perhaps to use an counter to keep track of the re-entangle qubits
        # self.re_entangle_positions = []
        # m_size is the number of qubits in the register
        self.m_size = m_size
        # batch_size is the number of qubits we start verification process
        self.batch_size = batch_size
        # current verification batches key:(mem_pos, ..., mem_pos_m), value: [mem_pos1, ..., qubit_batch_size]
        self.current_verification_batches = {}
        # current pending response batches key:(mem_pos, ..., mem_pos_m), value: [mem_pos1, ..., mem_pos_m]
        self.pending_verification_batches = {}
        # successful verification batches key:(mem_pos, ..., mem_pos_m), value: [mem_pos1, ..., mem_pos_m]
        self.successful_verification_batches = {}
        # condition for who starts the verification process
        self.is_source = False
        # measurement operators
        self.measurement_m0 = measurement_m0
        self.measurement_m1 = measurement_m1
        # controlled unitary gate
        self.CU_Gate = CU_Gate
        self.CCU_Gate = CCU_Gate

        # classical message queue
        self.cc_message_queue = []

        self.start_time = sim_time()
        # statistics
        self.verification_counter = 0
        self.successful_verification_counter = 0
        self.successful_verification_probability = []

        # Memory mapping
        self.target_node = target_node
        # self.memory_pos_mapping = {}  # key entangle_node, value = {mem_pos: remote_memo_pos)}

        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger
        self.is_top_layer = is_top_layer

        if self.is_top_layer:
            self.add_signal(MessageType.VERIFICATION_FINISHED)

    def handle_entanglement_signal(self, message: SignalMessages.SwapEntangledSuccess):
        """
        Handle the entanglement signal message.
        :param message: SignalMessage.SwapEntangledSuccess
        :return:
        """
        self.is_source = message.is_source
        if message.entangle_node not in self.entangled_qubits:
            self.entangled_qubits[message.entangle_node] = {}
        self.entangled_qubits[message.entangle_node][message.mem_pos] = message.target_memo_pos

        # if message.mem_pos in self.re_entangle_positions:
        #     self.re_entangle_positions.remove(message.mem_pos)

    def run(self):
        self.logger.info(f"{self.name} Verification protocol started with {self.entangled_node} [{self.uid}]")
        entangle_signals = self.await_signal(self.purification_protocol, Signals.SUCCESS)
        verification_signals = (self.await_signal(self.cc_message_handler, MessageType.SECURITY_VERIFICATION_START) |
                                self.await_signal(self.cc_message_handler, MessageType.SECURITY_VERIFICATION_REQUEST) |
                                self.await_signal(self.cc_message_handler, MessageType.SECURITY_VERIFICATION_READY) |
                                self.await_signal(self.cc_message_handler, MessageType.SECURITY_VERIFICATION_RESULT))
        self.start_time = sim_time()
        while True:
            exper = yield entangle_signals | verification_signals
            if exper.first_term.value:
                # handle entanglement signals from purification protocol
                for event in exper.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(event=event, receiver=self)
                    result: SignalMessages.SwapEntangledSuccess = ready_signal.result
                    if result.entangle_node != self.target_node:
                        continue
                    if result.timestamp < self.start_time:
                        continue
                    if ready_signal.label == Signals.SUCCESS:
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received entanglement signal from {source_protocol.name}\n"
                                         f"\tMemory Position: {result.mem_pos}\n"
                                         f"\t{self.entangled_qubits}\n", color="blue")
                        self.handle_entanglement_signal(result)
            if exper.second_term.value:
                for event in exper.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(event=event, receiver=self)
                    result: ClassicalMessage = ready_signal.result
                    if result.from_node != self.target_node:
                        continue
                    if result.data.timestamp < self.start_time:
                        continue
                    if ready_signal.label == MessageType.SECURITY_VERIFICATION_REQUEST:
                        if not isinstance(result.data, SignalMessages.SecurityVerificationSignalMessage):
                            continue
                        message: SignalMessages.SecurityVerificationSignalMessage = result.data
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received verification request signal from {source_protocol.name}\n"
                                         f"\tSourceBatchID: {message.source_verification_batch_id}\n"
                                         f"\tTargetBatchID: {message.target_verification_batch_id}\n"
                                         f"\tSourceBatchPoses: {message.source_verification_batch_poses}\n"
                                         f"\tTargetBatchPoses: {message.target_verification_batch_poses}\n",
                                         color="orange")
                        self.handle_verification_request(message)
                    elif ready_signal.label == MessageType.SECURITY_VERIFICATION_READY:
                        if not isinstance(result.data, SignalMessages.SecurityVerificationSignalMessage):
                            continue
                        message: SignalMessages.SecurityVerificationSignalMessage = result.data
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received verification ready signal from {source_protocol.name}\n"
                                         f"\tSourceBatchID: {message.source_verification_batch_id}\n"
                                         f"\tTargetBatchID: {message.target_verification_batch_id}\n"
                                         f"\tSourceBatchPoses: {message.source_verification_batch_poses}\n"
                                         f"\tTargetBatchPoses: {message.target_verification_batch_poses}\n",
                                         color="yellow")
                        yield from self.handle_verification_ready(message)
                    elif ready_signal.label == MessageType.SECURITY_VERIFICATION_START:
                        if not isinstance(result.data, SignalMessages.SecurityVerificationStartSignalMessage):
                            continue
                        message: SignalMessages.SecurityVerificationStartSignalMessage = result.data
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received verification start signal from {source_protocol.name}\n"
                                         f"\tSourceBatchID: {message.source_verification_batch_id}\n"
                                         f"\tTargetBatchID: {message.target_verification_batch_id}\n"
                                         f"\tSourceBatchPoses: {message.source_verification_batch_poses}\n"
                                         f"\tTargetBatchPoses: {message.target_verification_batch_poses}\n"
                                         f"\tMeasurement: \n\t\t{message.verif_teleport_measurement}",
                                         color="yellow")
                        yield from self.handle_verification_start(message)
                    elif ready_signal.label == MessageType.SECURITY_VERIFICATION_RESULT:
                        if not isinstance(result.data, SignalMessages.SecurityVerificationResultSignalMessage):
                            continue
                        message: SignalMessages.SecurityVerificationResultSignalMessage = result.data
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received verification result signal from {source_protocol.name}",
                                         color="yellow")
                        self.handle_verification_result(message)
            self.process_cc_message_queue()
            if self.is_source:
                self.check_available_verification()
            if self.is_top_layer:
                if self.check_end_condition():
                    self.logger.info(f"{self.name} -> {self.node.name} "
                                     f"verification protocol finished \n"
                                     f"Entangled Pairs {self.entangled_qubits}", color="green")

                    # broadcast the verification finished signal to lower layer
                    self.cc_message_handler.send_signal(MessageType.SECURITY_VERIFICATION_FINISHED,
                                                        SignalMessages.ProtocolFinishedSignalMessage(
                                                            from_protocol=self,
                                                            from_node=self.node.name,
                                                            entangle_node=self.entangled_node
                                                        ))

                    self.send_signal(MessageType.SECURITY_VERIFICATION_FINISHED,
                                     {"verification_probability": self.successful_verification_probability,
                                      "verification_success_count": self.successful_verification_counter,
                                      "verification_total_count": self.verification_counter,
                                      "verification_batches": self.successful_verification_batches})

                    break

    def process_cc_message_queue(self):
        temp = self.cc_message_queue
        self.cc_message_queue = []
        for message in temp:
            if isinstance(message, SignalMessages.SecurityVerificationSignalMessage):
                self.handle_verification_request(message)

    def handle_verification_request(self, message):
        """
        Handle the verification request signal message from source node
        :param message: SignalMessages.SecurityVerificationStartSignalMessage
        :return:
        """
        verification_batch_id = message.source_verification_batch_id
        verification_batch_positions = message.source_verification_batch_poses
        teleport_positions = list(verification_batch_id)
        for pos in verification_batch_positions:
            if pos not in self.entangled_qubits[self.target_node]:
                self.logger.info(f"{self.name} -> {self.entangled_node} "
                                 f"verification batch positions are entangled yet", color="red")
                self.cc_message_queue.append(message)
                return
        for pos in teleport_positions:
            if pos not in self.entangled_qubits[self.target_node]:
                self.logger.info(f"{self.name} -> {self.entangled_node} "
                                 f"verification batch positions are entangled yet", color="red")
                self.cc_message_queue.append(message)
                return
        # now we have all qubits for the verification process
        self.current_verification_batches[verification_batch_id] = verification_batch_positions
        for pos in verification_batch_positions:
            del self.entangled_qubits[self.target_node][pos]
        for pos in teleport_positions:
            del self.entangled_qubits[self.target_node][pos]
        # respond to the source node
        res = SignalMessages.SecurityVerificationSignalMessage(
            entangle_node=self.node.name,
            source_verification_batch_id=message.target_verification_batch_id,
            source_verification_batch_poses=message.target_verification_batch_poses,
            target_verification_batch_id=verification_batch_id,
            target_verification_batch_poses=verification_batch_positions, )
        self.cc_message_handler.send_message(MessageType.SECURITY_VERIFICATION_READY,
                                             self.target_node,
                                             ClassicalMessage(self.node.name,
                                                              self.target_node,
                                                              res))

    def handle_verification_ready(self, message):
        """
        Handle the verification ready signal message from source node
        :param message: SignalMessages.SecurityVerificationSignalMessage
        :return:
        """
        verification_batch_id = message.source_verification_batch_id
        verification_batch_positions = message.source_verification_batch_poses
        teleport_positions = list(verification_batch_id)
        # we need copy the verification batch positions to avoid manipulation of the original list
        self.current_verification_batches[verification_batch_id] = copy.copy(verification_batch_positions)
        del self.pending_verification_batches[verification_batch_id]
        self.verification_counter += 1
        yield from self.start_verification(verification_batch_id, verification_batch_positions, teleport_positions,
                                           message.target_verification_batch_id,
                                           message.target_verification_batch_poses)

    def handle_verification_result(self, message: SignalMessages.SecurityVerificationResultSignalMessage):
        """
        Handle the verification result signal message from remote node
        :param message: SignalMessages.SecurityVerificationResultSignalMessage
        :return:
        """
        if message.verif_result == 0:
            # case we have successful verification
            self.logger.info(f"{self.name} -> {self.node.name} "
                             f"verification successful for batch {message.source_verification_batch_id}\n"
                             f"\t Mem pos:{message.source_verification_batch_poses}", color="green")
            self.successful_verification_batches[message.source_verification_batch_id] = message.source_verification_batch_poses
            del self.current_verification_batches[message.source_verification_batch_id]
            # self.send_signal(Signals.SUCCESS, SignalMessages.VerificationSignalMessage(self.entangled_node,
            #                                                                            message.verif_batch_id,
            #                                                                            message.verif_batch_poses))
            # TODO: should we remove the success batch as we dont have them anymore?
            self.send_signal(Signals.SUCCESS, SignalMessages.VerificationSuccessSignalMessage(
                source_node=self.node.name,
                entangle_node=self.entangled_node,
                is_source=self.is_source,
                verification_batch_poses=message.source_verification_batch_poses
            ))
            if not self.is_top_layer:
                # we remove this as we sent the information to the upper layer. If we are the top we keep this
                del self.successful_verification_batches[message.source_verification_batch_id]
            # send to lower layer to generate the teleportation qubits
            # TODO temp removed re-entangle function
            # self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER,
            #                                     SignalMessages.ReEntangleSignalMessage(self.entangled_node,
            #                                                                            list(message.source_verification_batch_id)))
            self.successful_verification_counter += 1
            self.successful_verification_probability.append(message.result_probability)
        else:
            # we have failed the verification process, re-entangle the qubits
            self.handle_verification_failed(message.source_verification_batch_id)

    def check_available_verification(self):
        """
        Check if we can start the verification process.
        :return:
        """
        if len(self.entangled_qubits[self.target_node]) >= self.batch_size + self.m_size:
            self.logger.info(f"{self.name} -> {self.target_node} "
                             f"starting verification process with {self.target_node}", color="yellow")
            # get the first batch of qubits
            using_positions = list(self.entangled_qubits[self.target_node].keys())[:self.batch_size + self.m_size]
            # remove the positions from the entangled pairs to avoid next batch to have the same positions
            remote_poses = []
            for pos in using_positions:
                remote_poses.append(self.entangled_qubits[self.target_node][pos])
                del self.entangled_qubits[self.target_node][pos]
            teleport_positions = using_positions[:self.m_size]
            teleport_positions_remote = remote_poses[:self.m_size]
            verification_batch_positions = using_positions[self.m_size:]
            verification_batch_positions_remote = remote_poses[self.m_size:]
            # create the verification batch
            verification_batch_id = tuple(teleport_positions)
            self.logger.info(f"{self.name} -> {self.target_node} "
                             f"Starting verification process with {self.target_node}\n"
                             f"\tBatchID: {verification_batch_id}\n"
                             f"\tBatchPoses: {verification_batch_positions}\n"
                             f"\tRemoteBatchID: {teleport_positions_remote}\n"
                             f"\tRemotePoses: {verification_batch_positions_remote}\n", color="yellow")
            self.pending_verification_batches[verification_batch_id] = verification_batch_positions
            # send the verification request to the entangled node
            sig_message = (SignalMessages.
            SecurityVerificationSignalMessage(
                self.node.name,
                source_verification_batch_id=tuple(teleport_positions_remote),
                target_verification_batch_id=verification_batch_id,
                source_verification_batch_poses=verification_batch_positions_remote,
                target_verification_batch_poses=verification_batch_positions,
            ))
            self.cc_message_handler.send_message(MessageType.SECURITY_VERIFICATION_REQUEST,
                                                 self.target_node,
                                                 ClassicalMessage(self.node.name,
                                                                  self.target_node,
                                                                  sig_message))

    def start_verification(self, verification_batch_id, verification_batch_positions, teleport_positions,
                           target_verification_batch_id, target_verification_batch_poses
                           ):
        """
        Start the verification process.
        :param verification_batch_id: verification batch id
        :param verification_batch_positions: verification batch positions
        :param teleport_positions: teleport positions
        :param target_verification_batch_id: target verification batch id
        :param target_verification_batch_poses: target verification batch poses
        :return:
        """

        # measure our result m1's and send them to the entangled node
        # Step 1: Alice prepares register a
        uniform_qubits = self.create_uniform_superposition()

        # Step 2: Alice applies W to a ⊗ L
        uniform_qubits = yield from self.apply_W_operator(uniform_qubits, verification_batch_positions)

        # Teleport register_a to Bob
        measurement_result = yield from self.prepare_teleport_qubit(uniform_qubits, teleport_positions)

        # send the measurement results to the entangled node
        self.cc_message_handler.send_message(MessageType.SECURITY_VERIFICATION_START,
                                             self.target_node,
                                             ClassicalMessage(self.node.name,
                                                              self.target_node,
                                                              SignalMessages.SecurityVerificationStartSignalMessage(
                                                                  entangle_node=self.node.name,
                                                                  source_verification_batch_id=target_verification_batch_id,
                                                                  source_verification_batch_poses=target_verification_batch_poses,
                                                                  target_verification_batch_id=verification_batch_id,
                                                                  target_verification_batch_poses=verification_batch_positions,
                                                                  verification_teleport_measurement=measurement_result)))

    def handle_verification_start(self, message):
        """
        Handle the verification start signal message from source node
        We need to apply the W* operator and perform projective measurement and send the
        results back to the source node
        :param message: SignalMessages.VerificationStartSignalMessage
        :return:
        """
        if message.source_verification_batch_id not in self.current_verification_batches:
            self.logger.error(f"{self.name} -> {self.node.name} Batch ID {message.verif_batch_id}"
                              f" not found in the current verification batches", color="red")
            return
        verification_batch_id = message.source_verification_batch_id
        verification_batch_positions = message.source_verification_batch_poses
        teleport_measurement = message.verif_teleport_measurement
        teleport_positions = list(verification_batch_id)
        self.verification_counter += 1

        teleported_qubits = yield from self.correct_teleportation(teleport_measurement)
        # Step 3: Bob applies W*
        teleported_qubits = yield from self.apply_W_star_operator(teleported_qubits, verification_batch_positions)
        # Step 4: Bob performs projective measurement
        result, p = self.projective_measurement(teleported_qubits)
        # send the result back to the source node
        self.cc_message_handler.send_message(MessageType.SECURITY_VERIFICATION_RESULT,
                                             self.target_node,
                                             ClassicalMessage(self.node.name,
                                                              self.target_node,
                                                              SignalMessages.SecurityVerificationResultSignalMessage(
                                                                  entangle_node=self.node.name,
                                                                  source_verification_batch_id=message.target_verification_batch_id,
                                                                  source_verification_batch_poses=message.target_verification_batch_poses,
                                                                  target_verification_batch_id=verification_batch_id,
                                                                  target_verification_batch_poses=verification_batch_positions,
                                                                  verification_result=result,
                                                                  result_probability=p)))
        # we have failed the verification process, re-entangle the qubits
        if result == 1:
            self.handle_verification_failed(verification_batch_id)
        # we have successful verification process
        if result == 0:
            self.logger.info(f"{self.name} -> {self.node.name} "
                             f"verification successful for batch {verification_batch_id}\n"
                             f"\t Mem pos:{verification_batch_positions}", color="green")
            self.successful_verification_counter += 1
            self.successful_verification_probability.append(p)
            self.successful_verification_batches[verification_batch_id] = verification_batch_positions
            del self.current_verification_batches[verification_batch_id]
            # self.send_signal(Signals.SUCCESS, SignalMessages.VerificationSignalMessage(self.entangled_node,
            #                                                                            verification_batch_id,
            #                                                                            verification_batch_positions))
            # TODO: should we remove the success batch as we dont have them anymore?
            self.send_signal(Signals.SUCCESS, SignalMessages.VerificationSuccessSignalMessage(
                source_node=self.node.name,
                entangle_node=self.entangled_node,
                is_source=self.is_source,
                verification_batch_poses=verification_batch_positions
            ))
            if not self.is_top_layer:
                # remove this information as we sent to top layer already
                del self.successful_verification_batches[verification_batch_id]
            # send to lower layer to generate the teleportation qubits
            # TODO temp remove re-entangle function
            # self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER,
            #                                     SignalMessages.ReEntangleSignalMessage(self.entangled_node,
            #                                                                            teleport_positions))

    def create_uniform_superposition(self):
        """Create a uniform superposition state of m qubits."""
        qubits = qapi.create_qubits(self.m_size)
        for qubit in qubits:
            qapi.operate(qubit, ops.H)
        return qubits

    def apply_W_operator(self, register_qubits, verification_batch_positions):
        """
        Apply the W operator to registered qubits (sigma) and the verify batche qubits.
        :return: list of m qubits in the state of sigma
        """
        qmemory = self.node.subcomponents[f"{self.entangled_node}_qmemory"]
        # get the qubits from the memory positions
        for pos in verification_batch_positions:
            # TODO: should we apply memory noise here during pop?
            #       for now we apply the noise during the pop
            if qmemory.busy:
                yield self.await_program(qmemory)
            qubit, = qmemory.pop(pos, skip_noise=False)
            register_qubits.append(qubit)
        # Apply the W operator
        qapi.operate(register_qubits, self.CU_Gate)
        # put the qubits back to the memory
        for pos, qubit in zip(verification_batch_positions, register_qubits[self.m_size:]):
            if qmemory.busy:
                yield self.await_program(qmemory)
            qmemory.put(qubit, pos)
        return register_qubits[:self.m_size]

    def apply_W_star_operator(self, register_qubits, verification_batch_positions):
        """
        Apply the W* operator to teleported_qubits and batch qubits.
        :return: list of m qubits in the state of sigma
        """
        # TODO: This is a placeholder. In practice, you'd define specific U_i operators for W*
        #       For simplicity, we'll just apply CNOT gates
        qmemory = self.node.subcomponents[f"{self.entangled_node}_qmemory"]
        # get the qubits from the memory positions
        for pos in verification_batch_positions:
            # TODO: should we apply memory noise here during pop?
            if qmemory.busy:
                yield self.await_program(qmemory)
            qubit, = qmemory.pop(pos, skip_noise=False)
            register_qubits.append(qubit)
        # Apply the W* operator
        qapi.operate(register_qubits, self.CCU_Gate)
        # put the qubits back to the memory
        for pos, qubit in zip(verification_batch_positions, register_qubits[self.m_size:]):
            if qmemory.busy:
                yield self.await_program(qmemory)
            qmemory.put(qubit, pos)
        return register_qubits[:self.m_size]

    def projective_measurement(self, register_qubits):
        """
        Perform projective measurement on the register qubits.
        :return: list of measurement results
        """
        # need to apply Hadamard gate before the measurement
        for qubit in register_qubits:
            qapi.operate(qubit, ops.H)
        result, p = qapi.gmeasure(register_qubits, [self.measurement_m0, self.measurement_m1])
        self.logger.info(f"{self.name} -> {self.entangled_node} "
                         f"projective measurement result: {result}\n"
                         f"\tProbability: {p}", color="yellow")
        return result, p

    def correct_teleportation(self, measurement_results):
        """
        Correct the teleportation based on the measurement results.
        @param measurement_results: dictionary of measurement results from the teleportation
        @return: list of corrected entangled qubits
        """
        entangled_qubits = []
        qmemory = self.node.subcomponents[f"{self.entangled_node}_qmemory"]
        for mem_pos, (m1, m2) in measurement_results.items():
            if qmemory.busy:
                yield self.await_program(qmemory)
            # TODO: should we apply memory noise here during pop?
            qubit, = qmemory.pop(mem_pos, skip_noise=False)
            # Correct the teleportation based on the measurement results
            if m1 == 1:
                qapi.operate(qubit, ops.Z)

            if m2 == 1:
                qapi.operate(qubit, ops.X)

            entangled_qubits.append(qubit)
        return entangled_qubits

    def prepare_teleport_qubit(self, qubit_to_send, teleport_memo_poses):
        """Teleport a qubit using an EPR pair."""
        measurement_results = {}
        qmemory = self.node.subcomponents[f"{self.entangled_node}_qmemory"]
        for qubit_a, mem_pos in zip(qubit_to_send, teleport_memo_poses):
            if qmemory.busy:
                yield self.await_program(qmemory)
            # TODO: should we apply memory noise here during pop?
            qubit_b, = qmemory.pop(mem_pos, skip_noise=False)
            qapi.operate([qubit_a, qubit_b], ops.CNOT)
            qapi.operate(qubit_a, ops.H)
            m1, _ = qapi.measure(qubit_a)
            m2, _ = qapi.measure(qubit_b)
            self.logger.info(f"{self.name} -> {self.node.name} measuring qubit {qubit_a} and {qubit_b}\n"
                             f"\tMemory Position: {mem_pos}\n"
                             f"\tM1: {m1}\n"
                             f"\tM2: {m2}", color="yellow")
            measurement_results[mem_pos] = (m1, m2)
        # need send the measurement results to the entangled node
        return measurement_results

    def handle_verification_failed(self, batch_id):
        """
        Handle the rejection of the verification process.
        :return:
        """
        re_entangle_positions = self.current_verification_batches[batch_id]
        re_entangle_positions += list(batch_id)
        # self.re_entangle_positions += re_entangle_positions
        self.logger.info(f"{self.name} -> {self.entangled_node} "
                         f"verification failed for batch {batch_id}. Re-entangling qubits\n"
                         f"\tPoses {re_entangle_positions}", color="red")
        del self.current_verification_batches[batch_id]
        # TODO temp commit re-entangle
        # self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER,
        #                                     SignalMessages.ReEntangleSignalMessage(self.entangled_node,
        #                                                                            re_entangle_positions))

    def check_end_condition(self):
        """
        Check the end condition of the protocol if we are the top layer.
        :return: True if the protocol is finished, False otherwise
        """
        current_entangled_count = len(self.entangled_qubits)
        success_verification_pairs = 0
        for batch_id, batch_poses in self.successful_verification_batches.items():
            success_verification_pairs += len(batch_poses)
        done_entangle = current_entangled_count + success_verification_pairs == self.max_entangle_pairs
        if done_entangle is True and current_entangled_count < self.batch_size + self.m_size:
            return True
        # if self.max_entangle_pairs - success_verification_pairs < self.batch_size + self.m_size and \
        #     len(self.re_entangle_positions) == 0:
        #     return True
        return False

    def reset(self):
        self.entangled_qubits = {}
        self.current_verification_batches = {}
        self.pending_verification_batches = {}
        self.successful_verification_batches = {}
        self.cc_message_queue = []
        self.verification_counter = 0
        self.successful_verification_counter = 0
        self.successful_verification_probability = []
        super().reset()

    def stop(self):
        super().stop()

    def clean_gates(self):
        del self.CCU_Gate
        del self.CU_Gate
        del self.measurement_m0
        del self.measurement_m1
        gc.collect()
