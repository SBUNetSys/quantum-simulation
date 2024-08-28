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


class Verification(NodeProtocol):
    def __init__(self, node, name, entangled_node, purification_protocol, cc_message_handler,
                 m_size,
                 batch_size,
                 logger=None,
                 is_top_layer=False):
        super().__init__(node, name)

        self.purification_protocol = purification_protocol
        self.cc_message_handler = cc_message_handler
        # mapping of entangled qubits to memory positions key: memory position, value: fidelity
        self.entangled_pairs = {}
        self.entangled_node = entangled_node

        # m_size is the number of qubits in the register
        self.m_size = m_size
        # batch_size is the number of qubits we start verification process
        self.batch_size = batch_size
        # current verification batches key:(mem_pos, ..., mem_pos_m), value: [qubit1, ..., qubit_batch_size]
        self.current_verification_batches = {}
        # # current teleportation batches key:(mem_pos, ..., mem_pos_m), value: [mem_pos1, ..., mem_pos_m]
        # self.current_teleportation_batches = []
        # condition for who starts the verification process
        self.is_source = False

        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger
        self.is_top_layer = is_top_layer

        if self.is_top_layer:
            self.add_signal(MessageType.PROTOCOL_FINISHED)

    def handle_entanglement_signal(self, message):
        """
        Handle the entanglement signal message.
        :param message: SignalMessage.PurifySuccessSignalMessage
        :return:
        """
        self.is_source = message.is_source
        self.entangled_pairs[message.mem_pos] = message.fidelity

    def run(self):
        self.logger.info(f"{self.node.name} Verification protocol started with {self.entangled_node.name}")
        entangle_signals = self.await_signal(self.purification_protocol, Signals.SUCCESS)
        verification_signals = (self.await_signal(self.cc_message_handler, MessageType.VERIFICATION_START) |
                                self.await_signal(self.cc_message_handler, MessageType.VERIFICATION_RESULT))

        while True:
            exper = yield entangle_signals | verification_signals
            if exper.first_term.value:
                # handle entanglement signals from purification protocol
                for event in exper.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(event=event, receiver=self)
                    result: SignalMessages.PurifySuccessSignalMessage = ready_signal.result
                    if ready_signal.label == Signals.SUCCESS:
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received entanglement signal from {source_protocol.name}", color="blue")
                        self.handle_entanglement_signal(result)
            if exper.second_term.value:
                for event in exper.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(event=event, receiver=self)
                    result: ClassicalMessage = ready_signal.result
                    if ready_signal.label == MessageType.VERIFICATION_START:
                        message: SignalMessages.VerificationStartSignalMessage = result.data
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received verification start signal from {source_protocol.name}\n"
                                         f"\tBatchID: {message.verif_batch_id}\n"
                                         f"\tBatchPoses: {message.verif_batch_poses}\n"
                                         f"\tMeasurement: \n\t\t{message.verif_teleport_measurement}",
                                         color="yellow")
                        self.handle_verification_start(message)
                    elif ready_signal.label == MessageType.VERIFICATION_RESULT:
                        message = result.data
                        self.logger.info(f"{self.name} -> {self.node.name} "
                                         f"received verification result signal from {source_protocol.name}",
                                         color="yellow")
                        self.handle_verification_result(message)

    def start_verification(self):
        """
        Start the verification process.
        :return:
        """
        if len(self.entangled_pairs) >= self.batch_size + self.m_size:
            self.logger.info(f"{self.name} -> {self.node.name} "
                             f"starting verification process with {self.entangled_node.name}", color="blue")
            # get the first batch of qubits
            verification_batch_positions = list(self.entangled_pairs.keys())[:self.batch_size + self.m_size]
            # remove the positions from the entangled pairs to avoid next batch to have the same positions
            for pos in verification_batch_positions:
                del self.entangled_pairs[pos]
            teleport_positions = verification_batch_positions[:self.m_size]
            verification_batch_positions = verification_batch_positions[self.m_size:]
            # create the verification batch
            verification_batch_id = tuple(teleport_positions)
            self.current_verification_batches[verification_batch_id] = verification_batch_positions
            self.logger.info(f"{self.name} -> {self.node.name} "
                             f"Starting verification process with {self.entangled_node.name}\n"
                             f"\tBatchID: {verification_batch_id}\n"
                             f"\tBatchPoses: {verification_batch_positions}", color="yellow")
            # measure our result m1's and send them to the entangled node
            # Step 1: Alice prepares register a
            uniform_qubits = self.create_uniform_superposition()

            # Step 2: Alice applies W to a ⊗ L
            teleport_qubits = self.apply_W_operator(uniform_qubits, teleport_positions)

            # Teleport register_a to Bob
            measurement_result = self.prepare_teleport_qubit(uniform_qubits, teleport_qubits, teleport_positions)

            # send the measurement results to the entangled node
            self.cc_message_handler.send_message(MessageType.VERIFICATION_START,
                                                 self.entangled_node.name,
                                                 ClassicalMessage(self.node.name,
                                                                  self.entangled_node.name,
                                                                  SignalMessages.VerificationStartSignalMessage(
                                                                      self.node.name,
                                                                      verification_batch_id,
                                                                      verification_batch_positions,
                                                                      measurement_result)))

            # # Step 3: Bob applies W* (simplified as W again for this example)
            # self.apply_W_operator(register_a, qubit_bob)
            #
            # # Step 4: Bob performs projective measurement
            # target_state = self.create_uniform_superposition(m)
            # outcome, prob = self.projective_measurement(register_a, target_state)
            #
            # # Check if the outcome is close to the expected state
            # return prob > 1 - epsilon

    def handle_verification_start(self, message):
        """
        Handle the verification start signal message from source node
        We need to apply the W* operator and perform projective measurement and send the
        results back to the source node
        :param message: SignalMessages.VerificationStartSignalMessage
        :return:
        """
        verification_batch_id = message.verif_batch_id
        verification_batch_positions = message.verif_batch_poses
        teleport_measurement = message.verif_teleport_measurement
        self.current_verification_batches[verification_batch_id] = verification_batch_positions
        for pos in verification_batch_positions:
            del self.entangled_pairs[pos]
        teleport_positions = list(verification_batch_id)
        for pos in teleport_positions:
            del self.entangled_pairs[pos]
        teleported_qubits = self.correct_teleportation(teleport_measurement)
        # Step 3: Bob applies W*
        uniform_qubits = self.create_uniform_superposition()
        self.apply_W_star_operator(uniform_qubits, teleported_qubits)
        # Step 4: Bob performs projective measurement
        result = self.projective_measurement(teleported_qubits, uniform_qubits)
        # send the result back to the source node
        self.cc_message_handler.send_message(MessageType.VERIFICATION_RESULT,
                                             self.entangled_node.name,
                                             ClassicalMessage(self.node.name,
                                                              self.entangled_node.name,
                                                              result))
    def create_uniform_superposition(self):
        """Create a uniform superposition state of m qubits."""
        qubits = qapi.create_qubits(self.m_size)
        for qubit in qubits:
            qapi.operate(qubit, ops.H)
        return qubits

    def apply_W_operator(self, register_qubits, teleport_qubit_poses):
        """
        Apply the W operator to registered qubits (sigma) and teleport_qubits (m).
        :return: list of m qubits in the state of sigma
        """
        qmemory = self.node.subcomponents[f"{self.entangled_node}_qmemory"]
        # get the qubits from the memory positions
        m_qubits = []
        for pos in teleport_qubit_poses:
            # TODO: should we apply memory noise here during pop?
            if qmemory.busy:
                yield self.await_program(qmemory)
            qubit, = qmemory.pop(pos)
            m_qubits.append(qubit)
        # TODO: This is a placeholder. In practice, you'd define specific U_i operators
        #       For simplicity, we'll just apply CNOT gates
        for qubit_a, qubit_b in zip(register_qubits, m_qubits):
            qapi.operate([qubit_a, qubit_b], ops.CNOT)
        # TODO: should we put the qubits back to the memory?
        # for pos, qubit in zip(teleport_qubit_poses, m_qubits):
        #     if qmemory.busy:
        #         yield self.await_program(qmemory)
        #     qmemory.put(pos, qubit)
        return m_qubits

    @staticmethod
    def apply_W_star_operator(register_qubits, teleported_qubits):
        """
        Apply the W* operator to registered qubits (sigma) and teleported_qubits from source node.
        :return: list of m qubits in the state of sigma
        """
        # TODO: This is a placeholder. In practice, you'd define specific U_i operators for W*
        #       For simplicity, we'll just apply CNOT gates
        for qubit_a, qubit_b in zip(register_qubits, teleported_qubits):
            qapi.operate([qubit_a, qubit_b], ops.CNOT)
        return register_qubits

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
            qubit, = qmemory.pop(mem_pos)
            # Correct the teleportation based on the measurement results
            if m2:
                qapi.operate(qubit, ops.X)
            if m1:
                qapi.operate(qubit, ops.Z)
            entangled_qubits.append(qubit)
        return entangled_qubits

    @staticmethod
    def prepare_teleport_qubit(qubit_to_send, teleport_qubits, teleport_memo_poses):
        """Teleport a qubit using an EPR pair."""
        measurement_results = {}
        for qubit_a, qubit_b, mem_pos in zip(qubit_to_send, teleport_qubits, teleport_memo_poses):
            qapi.operate([qubit_a, qubit_b], ops.CNOT)
            qapi.operate(qubit_a, ops.H)
            m1, _ = qapi.measure(qubit_a)
            m2, _ = qapi.measure(qubit_b)
            measurement_results[mem_pos] = (m1, m2)
        # need send the measurement results to the entangled node
        return measurement_results

    @staticmethod
    def projective_measurement(qubits, target_state):
        """Perform a projective measurement."""
        # TODO: This is a simplified version. In practice, you'd construct a proper projector
        fidelity = qapi.fidelity(qubits, target_state)
        # Simulate measurement outcome based on fidelity
        outcome = np.random.random() < fidelity
        return outcome, fidelity

    def verify_epr_pair(self, qubit_alice, qubit_bob, epsilon=0.01):
        """Perform high-fidelity EPR verification."""
        # Step 1: Alice prepares register a
        register_a = self.create_uniform_superposition()

        # Step 2: Alice applies W to a ⊗ L
        self.apply_W_operator(register_a, qubit_alice)

        # Teleport register_a to Bob
        for qubit in register_a:
            qubit = self.teleport_qubit(qubit, qubit_alice, qubit_bob)

        # Step 3: Bob applies W* (simplified as W again for this example)
        self.apply_W_operator(register_a, qubit_bob)

        # Step 4: Bob performs projective measurement
        target_state = self.create_uniform_superposition(self.m)
        outcome, prob = self.projective_measurement(register_a, target_state)

        # Check if the outcome is close to the expected state
        return prob > 1 - epsilon
