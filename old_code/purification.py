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
from netsquid.components.instructions import INSTR_MEASURE, INSTR_CNOT, IGate, INSTR_Z, INSTR_SWAP, INSTR_H
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


class Purification(NodeProtocol):
    """
    Protocol for entanglement purification.

    This is done in combination with another node.

    """

    def __init__(self, node, left_port=None, right_port=None, start_expression=None, msg_header="purification",
                 name=None, target_fidelity=0.9):
        """
        Initialization of purification protocol
        :param node: node on which the protocol is running
        :param left_port: classical port to communicate with the left neighbour
        :param right_port: classical port to communicate with the right neighbour
        :param start_expression: start expression
        :param msg_header: purification message header
        :param name: name
        :param target_fidelity: purification target fidelity rate
        """
        if left_port is None and right_port is None:
            raise ValueError("At least one of left_port or right_port must be specified.")
        if left_port is not None and not isinstance(left_port, Port):
            raise TypeError("left_port should be a {}, not a {}".format(Port, type(left_port)))
        if right_port is not None and not isinstance(right_port, Port):
            raise TypeError("right_port should be a {}, not a {}".format(Port, type(right_port)))
        name = name if name else ("Purification({}, left_port:{}, right_port:{})"
                                  .format(node.name, left_port.name, right_port.name))
        super().__init__(node, name)
        self.add_signal("entangle")
        self.left_port = left_port
        self.right_port = right_port

        self.left_memory = None
        self.right_memory = None
        if left_port is not None:
            self.left_memory = self.node.subcomponents["left_qmemory"]
        if right_port is not None:
            self.right_memory = self.node.subcomponents["right_qmemory"]
        self.target_fidelity = target_fidelity

        # map of left neighbour's entangled pairs with their memory positions and fidelity
        self.left_entangled_pairs = {}
        # map of le neighbour's entangled pairs with higher fidelity
        self.left_satisfied_pairs = {}
        # map of right neighbour's entangled pairs with their memory positions and fidelity
        self.right_entangled_pairs = {}
        # map of right neighbour's entangled pairs with higher fidelity
        self.right_satisfied_pairs = {}
        # store temporary pairs until remote node is ready
        self.temporary_pairs = {}
        # store message from remote node
        self.remote_message = []
        # store header
        self.header = msg_header
        # currently purifying paris, store the memory position and fidelity
        self.purifying_paris = {}  # (p1, p2) -> (f1, f2)
        self.purifying_results = {}  # (p1, p2) -> (M1, M2)
        # TODO rename this expression to 'qubit input'
        self.start_expression = start_expression

        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))

    def run(self):
        if self.left_port is not None and self.right_port is not None:
            cchannel_ready = self.await_port_input(self.left_port) | self.await_port_input(self.right_port)
        elif self.left_port is not None:
            cchannel_ready = self.await_port_input(self.left_port)
        elif self.right_port is not None:
            cchannel_ready = self.await_port_input(self.right_port)
        # cchannel_ready_left = self.await_port_input(self.port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # if expr.triggered_events:  # Ensure there are triggered events to process
            #     for event in expr.triggered_events:
            #         # if event.source == cchannel_ready:
            #         #     for e in cchannel_ready.triggered_events:
            #         #         if e.source == self.left_port:
            #         #             classical_message = self.left_port.rx_input(header=self.header)
            #         #         elif e.source == self.right_port:
            #         #             classical_message = self.right_port.rx_input(header=self.header)
            #         #         if classical_message:
            #         #             messages = classical_message.items
            #         #             for msg in messages:
            #         #                 print(f"Purification -> Node {self.node.name} received classical message:"
            #         #                       f" {msg}")
            #         #                 yield from self._handle_cchannel_rx(msg)
            #         classical_message = None
            #         if event.source == self.left_port:
            #             classical_message = self.left_port.rx_input(header=self.header)
            #         elif event.source == self.right_port:
            #             classical_message = self.right_port.rx_input(header=self.header)
            #         if classical_message:
            #             messages = classical_message.items
            #             for msg in messages:
            #                 print(f"Purification -> Node {self.node.name} received classical message: {msg}")
            #                 yield from self._handle_cchannel_rx(msg)
            #         else:
            #             source_protocol = event.source
            #             ready_signal = source_protocol.get_signal_by_event(
            #                 event=expr.second_term.triggered_events[0], receiver=self)
            #             print(f"Purification ->Node {self.node.name} received qubit signal: {ready_signal.result}")
            #             result = ready_signal.result
            #             if "left_mem_pos" in result:
            #                 # qmemory = self.node.subcomponents['left_qmemory']
            #                 mem_pos = result["left_mem_pos"]
            #                 yield from self._handle_qubit_rx(self.left_memory, "left", mem_pos)
            #
            #             elif "right_mem_pos" in result:
            #                 mem_pos = result["right_mem_pos"]
            #                 # qmemory = self.node.subcomponents['right_qmemory']
            #                 yield from self._handle_qubit_rx(self.right_memory, "right", mem_pos)
            # self.send_signal(Signals.BUSY)
            if expr.first_term:
                for event in expr.first_term.triggered_events:
                    if event.source == self.left_port:
                        classical_message = self.left_port.rx_input(header=self.header)
                    elif event.source == self.right_port:
                        classical_message = self.right_port.rx_input(header=self.header)
                    if classical_message:
                        messages = classical_message.items
                        for msg in messages:
                            print(f"Purification -> Node {self.node.name} received classical message:"
                                  f" {msg}")
                            yield from self._handle_cchannel_rx(msg)
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                print(f"Purification ->Node {self.node.name} received qubit signal: {ready_signal.result}")
                result = ready_signal.result
                if "left_mem_pos" in result:
                    # qmemory = self.node.subcomponents['left_qmemory']
                    mem_pos = result["left_mem_pos"]
                    yield from self._handle_qubit_rx(self.left_memory, "left", mem_pos)

                elif "right_mem_pos" in result:
                    mem_pos = result["right_mem_pos"]
                    # qmemory = self.node.subcomponents['right_qmemory']
                    yield from self._handle_qubit_rx(self.right_memory, "right", mem_pos)
            yield from self.process_messages()
            # if self.right_entangled_pairs is None or len(self.right_entangled_pairs) >= 2:
            #     yield from self.start_purification()
            # print latest status
            self.print_status()

    def _handle_qubit_rx(self, qmemory, neighbour, memory_pos):
        # Handle incoming Qubit on this node.
        if qmemory.busy:
            yield self.await_program(qmemory)

        if neighbour == "left":
            qubit_fidelity = 0.1
            print(
                f"Purification -> Node {self.node.name} measured {neighbour} qubit at "
                f"position {memory_pos}: {qubit_fidelity}")
            # TODO: do we care about the fidelity of the qubit for left memory?
            # all purification will be done with the right memory (which in this case is the source node)
            if qubit_fidelity > self.target_fidelity:
                self.left_satisfied_pairs[memory_pos] = qubit_fidelity
            else:
                self.left_entangled_pairs[memory_pos] = qubit_fidelity
            self.print_status()

            # we are the remote node for left neighbour, we need to send the confirmation via classical channel
            print(f"Purification -> Node {self.node.name} sending entangled qubit confirmation to left neighbour"
                  f" memory pos {memory_pos}")
            self.left_port.tx_output(Message({"entangle": memory_pos}, header=self.header))
        else:
            # Right Qubit are from local QSource therefore we need to wait for the remote node to be ready
            self.temporary_pairs[memory_pos] = None
            print(f"Purification -> Node {self.node.name} temporary pairs:\n"
                  f"{self.paris_to_string(self.temporary_pairs)}\n"
                  f"remote message: {self.remote_message}")

    def print_status(self):
        print(f"Purification -> Node {self.node.name} entangled pairs:\n"
              f"\tLeft Nodes:\n"
              f"\t\t Entangled Pairs:\n"
              f"{self.paris_to_string(self.left_entangled_pairs)}\n"
              f"\t\t Satisfied Pairs:\n"
              f"{self.paris_to_string(self.left_satisfied_pairs)}\n"
              f"\tRight Nodes:\n"
              f"\t\t Entangled Pairs:\n"
              f"{self.paris_to_string(self.right_entangled_pairs)}\n"
              f"\t\t Satisfied Pairs:\n"
              f"{self.paris_to_string(self.right_satisfied_pairs)}\n"
              f"\tTemporary Pairs:\n"
              f"{self.paris_to_string(self.temporary_pairs)}\n"
              f"\tPurifying Pairs:\n"
              f"\t\t\t{self.purifying_paris}\n"
              f"\tPurifying Results:\n"
              f"\t\t\t{self.purifying_results}\n"
              f"\tRemote Message:\n"
              f"\t\t\t{self.remote_message}\n")

    def estimate_fidelity_theoretical(self, initial_fidelity, noise_params, channel_length):
        """Estimate fidelity based on noise parameters and channel length."""
        depolar_rate = noise_params['depolar_rate']
        dephase_rate = noise_params['dephase_rate']

        # Depolarizing effect
        p_depolar = 1 - np.exp(-depolar_rate * channel_length)
        f_depolar = (1 - p_depolar) + p_depolar / 4

        # Dephasing effect
        p_dephase = 1 - np.exp(-dephase_rate * channel_length)
        f_dephase = 1 - p_dephase / 2

        # Combine effects (assuming independent noise processes)
        final_fidelity = initial_fidelity * f_depolar * f_dephase

        return final_fidelity

    def paris_to_string(self, pairs):
        return "\n".join([f"\t\t\tMemory Position: {k}, Fidelity: {v}" for k, v in pairs.items()])

    def _handle_cchannel_rx(self, message):

        # Handle incoming classical message from sister node.
        if "entangle" in message:
            # Remote node is ready to entangle with usa
            print(f"Purification -> Node {self.node.name} received entangled qubit from right neighbour: "
                  f"memo pos {message['entangle']}")
            self.print_status()
            if len(self.temporary_pairs) > 0 and message["entangle"] in self.temporary_pairs:
                self.print_status()
                mem_pos = message["entangle"]
                mem_fidelity = 0.5
                if mem_fidelity > self.target_fidelity:
                    # we have a pair with higher fidelity
                    self.right_satisfied_pairs[mem_pos] = mem_fidelity
                else:
                    self.right_entangled_pairs[mem_pos] = mem_fidelity
                # remove the temporary pair
                del self.temporary_pairs[message["entangle"]]
                print(f"Purification -> Node {self.node.name} removing temporary pair {mem_pos}")
                self.print_status()
                # TODO: Start entanglement protocol with remote node
                # always start purification with the right memory paris
                #  A -> B -> C
                #  A will start purification with B
                #  B will start purification with C
                #  if no right memory then no need to start purification
            else:
                # we have no temporary pairs to entangle
                # TODO: race condition, we might get the message before we have the temporary pairs
                # add the message to the remote message queue
                self.remote_message.append(message)
                print(f"Purification -> Node {self.node.name} remote message queue: {self.remote_message}")
                self.print_status()
        elif "purify_start" in message:
            # A ->(purify start) B
            # B ->(purify measurement) A
            # A ->(purify result) B
            pair = message["purify_start"]
            print(f"Purification -> Node {self.node.name} received purification start message: {pair}")
            self.print_status()
            # start purification
            # we are using left memory to purify the pairs as we are the remote node
            m2 = yield from self.purify_measurement(pair[0], pair[1], self.left_memory)
            # send the measurement result to the remote node
            print(f"Purification -> Node {self.node.name} sending purification measurement to right neighbour"
                  f" memory pos {pair}, measurement: {m2}")
            self.left_port.tx_output(Message({"purify_measurement": (pair, m2)}, header=self.header))
        elif "purify_measurement" in message:
            # only operation with right memory will receive this message
            # A ->(purify start) B
            # B ->(purify measurement) A
            # A ->(purify result) B
            pair, m2 = message["purify_measurement"]
            m1 = self.purifying_results[pair][0]
            if m1 == m2:
                # check if the fidelity is higher than the target fidelity
                new_fidelity = qapi.fidelity(self.right_memory.peek(pair[0]), ks.b00)
                # purification is successful
                print(f"Purification -> Node {self.node.name} Purification successful for pair {pair}\n"
                      f"\tNew Fidelity: {new_fidelity}\n"
                      f"\tOld Fidelity: {self.purifying_paris[pair][0]}\n"
                      f"\tTarget Fidelity: {self.target_fidelity}")
                if new_fidelity > self.target_fidelity:
                    self.right_satisfied_pairs[pair[0]] = new_fidelity
                    print(f"Purification -> Node {self.node.name} Pair {pair} is satisfied, "
                          f"Add {pair[0]} to satisfied pairs")
                else:
                    self.right_entangled_pairs[pair[0]] = self.purifying_paris[pair][0]
                    print(f"Purification -> Node {self.node.name} Pair {pair} is not satisfied, "
                          f"Add {pair[0]} back to entangled pairs")
                    self.print_status()

                # send the message to the right neighbour the result
                self.right_port.tx_output(Message({"purify_result": (pair, True)}, header=self.header))
            else:
                # case of purification failure
                print(f"Purification -> Node {self.node.name} Purification failed for pair {pair}, "
                      f"Add {pair[0]} back to entangled pairs")
                self.right_entangled_pairs[pair[0]] = self.purifying_paris[pair][0]
                self.print_status()
                # send the message to the right neighbour the result
                self.right_port.tx_output(Message({"purify_result": (pair, False)}, header=self.header))
            # remove the pair from the purifying paris
            print(f"Purification -> Node {self.node.name} removing pair {pair} from purifying paris")
            del self.purifying_paris[pair]
            del self.purifying_results[pair]
            # we dont remove the pair[1] from the entangled pairs as we are the source, has been removed initially
            self.print_status()
            print(f"Purification -> Node {self.node.name} sending re-entangled signal to right neighbour")
            self.send_signal("entangle", {"right_mempos": pair[1]})
            # TODO send generation signal to Entangle protocol to generate new qubits
        elif "purify_result" in message:
            # only operation with left memory will receive this message
            # A ->(purify result) B
            print(f"Purification -> Node {self.node.name} received purification result: {message['purify_result']}")
            self.print_status()
            pair, result = message["purify_result"]
            if result:
                # purification is successful
                new_fidelity = qapi.fidelity(self.left_memory.peek(pair[0]), ks.b00)
                print(f"Purification -> Node {self.node.name} Purification successful for pair {pair}\n"
                      f"\tNew Fidelity: {new_fidelity}\n"
                      f"\tOld Fidelity: {self.left_entangled_pairs[pair[0]]}\n"
                      f"\tTarget Fidelity: {self.target_fidelity}")
                if new_fidelity > self.target_fidelity:
                    self.left_satisfied_pairs[pair[0]] = new_fidelity
                    print(f"Purification -> Node {self.node.name} Pair {pair} is satisfied, "
                          f"Add {pair[0]} to satisfied pairs")
                    self.print_status()

            # remove the pair is being measured
            print(f"Purification -> Node {self.node.name} removing destroyed {pair[1]} from purifying paris")
            del self.left_entangled_pairs[pair[1]]
            self.print_status()
            # TODO wait for new Entangle signal from the left neighbour
            print(f"Purification -> Node {self.node.name} sending re-entangled signal to left neighbour")
            self.send_signal("entangle", {"left_mempos": pair[1]})

    def start_purification(self):
        """
        Start the purification protocol.
        1. Check if we have enough entangled pairs (in right entangled pairs)
        2. Pick 2 pairs randomly (can be change) to start purification process
        3. Send classical message to the right neighbour to start purification process, with the memory positions

        Sequence of events:
        A -(purify start)-> B
            - A will start purification measurement with right memory (q1, q2)
            - Once B receives the message, it will start the purification measurement with left memory (q1, q2)
        B -(purify measurement)-> A
            - B will measure q2 and send the result to A
        A -(purify result)-> B
            - A will compare its measurement result with B's result and send the result (T/F) to B
        A -(entangle)-> B
            - A will regenerate a pair of qubits and send one to B (as the purification destroyed q2)
        :return:
        """
        # print(f"Purification -> Node {self.node.name} Starting purification process")
        # pick 2 pairs randomly
        pairs = list(self.right_entangled_pairs.keys())
        pair1 = pairs[0]
        pair2 = pairs[1]
        print(f"Purification -> Node {self.node.name} Starting purification for pairs: {pair1}, {pair2}")
        self.print_status()
        # store the pairs we are purifying
        self.purifying_paris[(pair1, pair2)] = (self.right_entangled_pairs[pair1], self.right_entangled_pairs[pair2])
        del self.right_entangled_pairs[pair1]
        del self.right_entangled_pairs[pair2]

        # send classical message to the right neighbour
        self.right_port.tx_output(Message({"purify_start": (pair1, pair2)}, header=self.header))
        # start purification meausrement
        m1 = yield from self.purify_measurement(pair1, pair2, self.right_memory)
        self.purifying_results[(pair1, pair2)] = (m1, None)
        # wait for the remote node to be ready

    def purify_measurement(self, q1_pos, q2_pos, qmemory):
        """
        Perform purification measurement on the qubits
        :param q1_pos: qubit 1 memory position
        :param q2_pos: qubit 2 memory position
        :param qmemory: quantum memory
        :return: True if the purification is successful otherwise False
        """
        # Handle incoming Qubit on this node.
        if qmemory.busy:
            yield self.await_program(qmemory)
        # Apply CNOT gate to qubit 1 and qubit 2
        qmemory.execute_instruction(INSTR_CNOT, [q1_pos, q2_pos])
        # Apply Hadamard gate to qubit 1
        if qmemory.busy:
            yield self.await_program(qmemory)
        qmemory.execute_instruction(INSTR_H, [q1_pos])
        # Measure qubit 2
        if qmemory.busy:
            yield self.await_program(qmemory)
        measured_result = qmemory.execute_instruction(INSTR_MEASURE, [q2_pos], output_key="M1")
        # TODO: can we remove q2 from the memory? since we measured it
        return measured_result[0]["M1"]

    def process_messages(self):
        # Process all messages in the message queue
        temp = self.remote_message
        self.remote_message = []
        for message in temp:
            yield from self._handle_cchannel_rx(message)

    def _check_success(self):
        # Check if protocol succeeded after receiving new input (qubit or classical information).
        # Returns true if protocol has succeeded on this node
        if (self.local_qcount > 0 and self.local_qcount == self.remote_qcount and
                self.local_meas_OK and self.remote_meas_OK):
            # SUCCESS!
            print(f"Purification -> Node {self.node.name} SUCCESS, qubit at position {self._qmem_pos} accepted.")
            self.send_signal(Signals.SUCCESS, self._qmem_pos)
        elif self.local_meas_OK and self.local_qcount > self.remote_qcount:
            # Need to wait for latest remote status
            pass
        else:
            # FAILURE
            print(f"Purification -> Node {self.node.name} FAILURE, qubit at position {self._qmem_pos} rejected.")
            self._handle_fail()
            self.send_signal(Signals.FAIL, self.local_qcount)

    def _handle_fail(self):
        if self.node.qmemory.mem_positions[self._qmem_pos].in_use:
            self.node.qmemory.pop(positions=[self._qmem_pos])
