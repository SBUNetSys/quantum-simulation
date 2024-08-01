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


def print_blue(msg):
    print(f"\033[94m{msg}\033[0m")


def print_green(msg):
    print(f"\033[92m{msg}\033[0m")


def print_orange(msg):
    print(f"\033[93m{msg}\033[0m")


def print_red(msg):
    print(f"\033[91m{msg}\033[0m")


class PurifyEntangle(NodeProtocol):
    """
    Protocol for entanglement purification.
    Only care about the entangled node, no need to care about left and right nodes.
    """

    def __init__(self, node,
                 cc_port=None,
                 entangle_node=None,
                 start_expression=None,
                 msg_header="purification",
                 name=None,
                 target_fidelity=0.9):
        """
        Initialization of purification protocol
        :param node: node on which the protocol is running
        :param cc_port: classical port to communicate with the entangle node
        :param start_expression: start expression
        :param msg_header: purification message header
        :param name: name
        :param target_fidelity: purification target fidelity rate
        """
        if cc_port is None:
            raise ValueError("cc_port must be specified.")
        if not isinstance(cc_port, Port):
            raise TypeError("cc_port should be a {}, not a {}".format(Port, type(cc_port)))
        if entangle_node is None:
            raise ValueError("entangle_node must be specified.")

        name = name if name else ("Purification({}, cc_port:{})"
                                  .format(node.name, cc_port.name))
        super().__init__(node, name)
        self.entangled_node = entangle_node
        self.cc_port = cc_port
        self._qmemory_name = f"{entangle_node}_qmemory"
        try:
            self.qmemory = self.node.subcomponents[self._qmemory_name]
        except KeyError:
            raise ValueError(f"Node {node.name} does not have a quantum memory named {self._qmemory_name}")

        self.target_fidelity = target_fidelity
        # map of entangled pairs with their memory positions and fidelity
        self.entangled_pairs = {}
        # map of entangled pairs with higher fidelity
        self.satisfied_pairs = {}
        # store temporary pairs until remote node is ready
        self.temporary_pairs = {}
        # store message from remote node
        self.remote_message = []
        # store header
        self.header = msg_header
        # currently purifying paris, store the memory position and fidelity
        self.purifying_paris = {}  # (p1, p2) -> (f1, f2)
        self.purifying_results = {}  # (p1, p2) -> (M1, M2)
        # this expression is 'qubit input' event
        self.start_expression = start_expression
        # is source node
        self.is_source = None
        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        # count of purification process
        # record how many purification pairs we have done in order to have all the pairs meet the target fidelity
        self.purification_count = 0
        self.purification_success_count = 0

    def add_new_signal(self, signal):
        """
        Add new signal to the protocol
        :param signal: signal name
        :return:
        """
        self.add_signal(signal)

    def run(self):
        cchannel_ready = self.await_port_input(self.cc_port)
        qmemory_ready = self.start_expression
        while True:
            # self.send_signal(Signals.WAITING)
            expr = yield cchannel_ready | qmemory_ready
            # self.send_signal(Signals.BUSY)
            if expr.first_term.value:
                classical_message = self.cc_port.rx_input(header=self.header)
                if classical_message:
                    messages = classical_message.items
                    for msg in messages:
                        print(f"Purify {self.name} -> Node {self.node.name} received classical message:"
                              f" {msg}")
                        yield from self._handle_cchannel_rx(msg)
            elif expr.second_term.value:
                source_protocol = expr.second_term.atomic_source
                ready_signal = source_protocol.get_signal_by_event(
                    event=expr.second_term.triggered_events[0], receiver=self)
                print(f"Purify {self.name} -> Node {self.node.name} received qubit signal: {ready_signal.result} from "
                      f"{source_protocol}")
                result = ready_signal.result
                qmem_name = result["qmemory"]
                if qmem_name == self._qmemory_name:
                    mem_pos = result["mem_pos"]
                    is_source = result["is_source"]
                    if self.is_source is None:
                        self.is_source = is_source
                    initial_fidelity = result["initial_fidelity"]
                    yield from self._handle_qubit_rx(self.qmemory, mem_pos, is_source, initial_fidelity)
                else:
                    # print in blue color for unknown source
                    print(
                        f"\033[94mPurify {self.name} -> Node {self.node.name} received qubit signal "
                        f"from unknown source: "
                        f"\033[0m{result}")
            # process remote messages that are in the queue
            yield from self.process_messages()
            if len(self.entangled_pairs) >= 2 and self.is_source:
                # we have enough entangled pairs to start purification, and we are the source node
                # purify always initiate by the source node
                yield from self.start_purification()
            self.print_status()
            # finished purifying all the pairs
            # TODO: what case we can say we are done? Current end condition is when we have no entangled pairs == 0
            # however, we still have one pair that is not purified which is being sent to re-entangle, but odd number
            # of pairs will always have one pair that is not purified
            if (len(self.entangled_pairs) == 0
                    and len(self.temporary_pairs) == 0
                    and len(self.remote_message) == 0
                    and len(self.purifying_paris) == 0
                    and len(self.satisfied_pairs) > 0):
                self.send_signal(Signals.SUCCESS, {"satisfied_pairs": self.satisfied_pairs,
                                                   "purification_count": self.purification_count,
                                                   "purification_success_count": self.purification_success_count,
                                                   "finish_time": sim_time()})
                break

    def _handle_qubit_rx(self, qmemory, memory_pos, is_source, initial_fidelity):
        # Handle incoming Qubit on this node.
        if qmemory.busy:
            yield self.await_program(qmemory)
        # store the temporary pairs
        self.temporary_pairs[memory_pos] = initial_fidelity
        print(f"Purify {self.name} -> Node {self.node.name} adding temporary pair {(memory_pos, initial_fidelity)}")
        if not is_source and initial_fidelity is None:
            # case of remote node
            # we are the remote node for left neighbour, we need to send the confirmation via classical channel
            print(f"Purify {self.name} -> Node {self.node.name} sending entangled qubit confirmation to source node"
                  f" memory pos {memory_pos}")
            self.cc_port.tx_output(Message({"entangle": memory_pos}, header=self.header))

    def print_status(self):
        print(f"Purify {self.name} -> Node {self.name} entangled pairs:\n"
              f"\tNodes:\n"
              f"\t\tEntangled Pairs:\n"
              f"{self.paris_to_string(self.entangled_pairs)}\n"
              f"\t\tSatisfied Pairs:\n"
              f"{self.paris_to_string(self.satisfied_pairs)}\n"
              f"\tTemporary Pairs:\n"
              f"{self.paris_to_string(self.temporary_pairs)}\n"
              f"\tPurifying Pairs:\n"
              f"\t\t\t{self.purifying_paris}\n"
              f"\tPurifying Results:\n"
              f"\t\t\t{self.purifying_results}\n"
              f"\tRemote Message:\n"
              f"\t\t\t{self.remote_message}\n")

    def estimate_fidelity_theoretical(self, initial_fidelity, depolar_rate=1e-3, channel_length=50):
        """Estimate fidelity based on noise parameters and channel length."""
        # depolar_rate = noise_params['depolar_rate']
        # dephase_rate = noise_params['dephase_rate']

        # Depolarizing effect
        p_depolar = 1 - np.exp(-depolar_rate * channel_length)
        f_depolar = (1 - p_depolar) + (p_depolar / 4)

        # # Dephasing effect
        # p_dephase = 1 - np.exp(-dephase_rate * channel_length)
        # f_dephase = 1 - p_dephase / 2

        # Combine effects (assuming independent noise processes)
        final_fidelity = initial_fidelity * f_depolar

        return final_fidelity

    def paris_to_string(self, pairs):
        return "\n".join([f"\t\t\tMemory Position: {k}, Fidelity: {v}" for k, v in pairs.items()])

    def _handle_cchannel_rx(self, message):

        # Handle incoming classical message from sister node.
        if "entangle" in message:
            # Remote node is ready to entangle with usa
            print(f"Purify {self.name} -> Node {self.node.name} received entangled qubit from entangle node: "
                  f"memo pos {message['entangle']}")
            if len(self.temporary_pairs) > 0 and message["entangle"] in self.temporary_pairs:
                mem_pos = message["entangle"]
                initial_fidelity = self.temporary_pairs[mem_pos]
                # calculate the fidelity
                mem_fidelity = self.estimate_fidelity_theoretical(initial_fidelity)

                if mem_fidelity > self.target_fidelity:
                    # we have a pair with higher fidelity
                    self.satisfied_pairs[mem_pos] = mem_fidelity
                else:
                    self.entangled_pairs[mem_pos] = mem_fidelity
                # remove the temporary pair
                del self.temporary_pairs[message["entangle"]]
                print(f"Purify {self.name} -> Node {self.node.name} removing temporary pair {mem_pos}")
                # if we are the source node, we need to send the result to the remote node
                if self.is_source:
                    print(
                        f"Purify {self.name} -> Node {self.node.name} sending entangled qubit confirmation to"
                        f"entangled node memory pos {mem_pos}")
                    self.cc_port.tx_output(Message({"fidelity": (mem_pos, mem_fidelity)}, header=self.header))
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
                print(f"Purify {self.name} -> Node {self.node.name} received unknown classical message,"
                      f"adding to remote message queue.\n"
                      f"\tCurrent remote message queue: {self.remote_message}")
        elif "fidelity" in message:
            # process the fidelity message from the source node
            print(f"Purify {self.name} -> Node {self.node.name} received fidelity message from source node: {message}")
            mem_pos, mem_fidelity = message["fidelity"]
            if mem_pos in self.temporary_pairs:
                # check if the fidelity is higher than the target fidelity
                if mem_fidelity > self.target_fidelity:
                    # we have a pair with higher fidelity
                    self.satisfied_pairs[mem_pos] = mem_fidelity
                else:
                    self.entangled_pairs[mem_pos] = mem_fidelity
                # remove the temporary pair
                del self.temporary_pairs[mem_pos]
                print(f"Purify {self.name} -> Node {self.node.name} removing temporary pair {mem_pos}")
            else:
                # add the message to the remote message queue
                self.remote_message.append(message)
                print(f"Purify {self.name} -> Node {self.node.name} remote message queue: {self.remote_message}")

        elif "purify_start" in message:
            # A ->(purify start) B
            # B ->(purify measurement) A
            # A ->(purify result) B
            pair = message["purify_start"]
            print_green(f"Purify {self.name} -> Node {self.node.name} received purification start message: {pair}"
                        f"from source node")
            # start purification
            # we are using left memory to purify the pairs as we are the remote node
            m2 = yield from self.purify_measurement(pair[0], pair[1], self.qmemory)
            # send the measurement result to the remote node
            print_green(f"Purify {self.name} -> Node {self.node.name} sending purification measurement to source node"
                        f" memory pos {pair}, measurement: {m2}")
            self.cc_port.tx_output(Message({"purify_measurement": (pair, m2)}, header=self.header))
        elif "purify_measurement" in message:
            # only operation with source node will receive this message
            # A ->(purify start) B
            # B ->(purify measurement) A
            # A ->(purify result) B
            pair, m2 = message["purify_measurement"]
            m1 = self.purifying_results[pair][0]
            if m1 == m2:
                # purification is successful
                self.purification_success_count += 1
                # check if the fidelity is higher than the target fidelity
                new_fidelity = self.calculate_purified_fidelity(self.purifying_paris[pair][0])
                # purification is successful
                print_orange(f"Purify {self.name} -> Node {self.node.name} Purification successful for pair {pair}\n"
                             f"\tNew Fidelity: {new_fidelity}\n"
                             f"\tOld Fidelity: {self.purifying_paris[pair][0]}\n"
                             f"\tTarget Fidelity: {self.target_fidelity}")
                if new_fidelity > self.target_fidelity:
                    self.satisfied_pairs[pair[0]] = new_fidelity
                    print_red(f"Purify {self.name} -> Node {self.node.name} Pair {pair} is satisfied, "
                              f"Add {pair[0]} to satisfied pairs")
                else:
                    self.entangled_pairs[pair[0]] = new_fidelity
                    print_orange(f"Purify {self.name} -> Node {self.node.name} Pair {pair} is not satisfied, "
                                 f"Add {pair[0]} back to entangled pairs")

                # send the message to the right neighbour the result
                print_green(f"Purify {self.name} -> Node {self.node.name} sending purification "
                            f"result to entangled node")
                self.cc_port.tx_output(Message({"purify_result": (pair, True, new_fidelity)}, header=self.header))
            else:
                # case of purification failure
                print_green(f"Purify {self.name} -> Node {self.node.name} Purification failed for pair {pair}\n"
                            f"\tAdd {pair[0]} back to entangled pairs")
                self.entangled_pairs[pair[0]] = self.purifying_paris[pair][0]
                # send the message to the right neighbour the result
                self.cc_port.tx_output(Message({"purify_result": (pair, False, None)}, header=self.header))

            # increase the purification count regardless of the result
            self.purification_count += 1
            # remove the pair from the purifying paris
            print_green(f"Purify {self.name} -> Node {self.node.name} removing pair {pair} from purifying paris")
            del self.purifying_paris[pair]
            del self.purifying_results[pair]
            # we dont remove the pair[1] from the entangled pairs as we are the source, has been removed initially
            print_blue(
                f"Purify {self.name} -> Node {self.node.name} sending re-entangled signal to GenEntangle protocol")
            self.send_signal(f"entangle_{self.node.name}->{self.entangled_node}",
                             {"mem_pos": pair[1], "qmemory_name": self._qmemory_name})
            # TODO send generation signal to Entangle protocol to generate new qubits
        elif "purify_result" in message:
            # only operation with left memory will receive this message
            # A ->(purify result) B
            print_green(
                f"Purify {self.name} -> Node {self.node.name} received purification result\n"
                f"\t{message['purify_result']}")
            self.purification_count += 1
            pair, result, new_fidelity = message["purify_result"]
            if result:
                # purification is successful
                self.purification_success_count += 1
                print_orange(f"Purify {self.name} -> Node {self.node.name} Purification successful for pair {pair}\n"
                             f"\tNew Fidelity: {new_fidelity}\n"
                             f"\tOld Fidelity: {self.entangled_pairs[pair[0]]}\n"
                             f"\tTarget Fidelity: {self.target_fidelity}")
                if new_fidelity > self.target_fidelity:
                    self.satisfied_pairs[pair[0]] = new_fidelity
                    print_red(f"Purify {self.name} -> Node {self.node.name} Pair {pair} is satisfied, "
                              f"Add {pair[0]} to satisfied pairs")
                    del self.entangled_pairs[pair[0]]
                    print_green(f"Purify {self.name} -> Node {self.node.name} removing satisfied {pair[0]} from "
                                f"entangled pairs")
                else:
                    self.entangled_pairs[pair[0]] = new_fidelity
                    print_orange(f"Purify {self.name} -> Node {self.node.name} Pair {pair} is not satisfied, "
                                 f"updating {pair[0]}'s fidelity")
            # remove the pair is being measured
            print_green(
                f"Purify {self.name} -> Node {self.node.name} removing destroyed {pair[1]} from entangled paris")
            del self.entangled_pairs[pair[1]]
            # TODO wait for new Entangle signal from the left neighbour
            print_blue(
                f"Purify {self.name} -> Node {self.node.name} sending re-entangled signal to GenEntangle protocol")
            self.send_signal(f"entangle_{self.node.name}->{self.entangled_node}", {"mem_pos": pair[1], "qmemory_name": self._qmemory_name})

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
        # print(f"Purify {self.name} -> Node {self.node.name} Starting purification process")
        # pick 2 pairs randomly
        # TODO maybe we can pick the pairs with one pair with the lowest fidelity and one with the highest fidelity?
        pairs = list(self.entangled_pairs.keys())
        # shuffle the pairs
        np.random.shuffle(pairs)
        pair1 = pairs[0]
        pair2 = pairs[1]

        print_green(f"Purify {self.name} -> Node {self.node.name} Starting purification for pairs: {pair1}, {pair2}")

        # store the pairs we are purifying
        self.purifying_paris[(pair1, pair2)] = (self.entangled_pairs[pair1], self.entangled_pairs[pair2])
        del self.entangled_pairs[pair1]
        del self.entangled_pairs[pair2]

        # send classical message to the right neighbour
        self.cc_port.tx_output(Message({"purify_start": (pair1, pair2)}, header=self.header))
        # start purification meausrement
        m1 = yield from self.purify_measurement(pair1, pair2, self.qmemory)
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
        measured_result = qmemory.execute_instruction(INSTR_MEASURE, [q2_pos], output_key="M")
        # TODO: can we remove q2 from the memory? since we measured it
        return measured_result[0]["M"]

    def calculate_purified_fidelity(self, initial_fidelity):
        F = initial_fidelity
        numerator = F ** 2 + (1 / 9) * (1 - F) ** 2
        denominator = F ** 2 + (2 / 3) * F * (1 - F) + (5 / 9) * (1 - F) ** 2
        new_fidelity = numerator / denominator
        return new_fidelity

    def process_messages(self):
        # Process all messages in the message queue
        temp = self.remote_message
        self.remote_message = []
        for message in temp:
            yield from self._handle_cchannel_rx(message)

    def reset(self):
        # clean up the pairs
        self.entangled_pairs = {}
        # map of entangled pairs with higher fidelity
        self.satisfied_pairs = {}
        # store temporary pairs until remote node is ready
        self.temporary_pairs = {}
        # store message from remote node
        self.remote_message = []
        super().reset()

    def stop(self):
        super().stop()
