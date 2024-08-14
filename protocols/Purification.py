import numpy as np

from netsquid.util.simtools import sim_time
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_CNOT, INSTR_H
from netsquid.components.component import Message, Port

from pydynaa import EventExpression

from netsquid.components.instructions import INSTR_MEASURE

from protocols.MessageHandler import MessageType
from utils import Logging, SignalMessages


class PurifyEntangle(NodeProtocol):
    """
    Protocol for entanglement purification.
    Only care about the entangled node, no need to care about left and right nodes.
    """

    def __init__(self,
                 node,
                 name,
                 entangled_nodes,
                 entanglement_handler,
                 cc_message_handler,
                 max_entangled_pair=10,
                 target_fidelity=0.9,
                 logger=None,
                 is_top_layer=False):
        """
        Initialization of purification protocol
        :param node: `~netsquid.nodes.node.Node`
                The node which the protocol is running.
        :param name: str
                Name of the protocol.
        :param entangled_nodes: list
                A list of entangled nodes name
        :param entanglement_handler: protocols.EntanglementHandler
                The entanglement handler protocol for the node
        :param cc_message_handler: protocols.MessageHandler
                The classical message handler protocol
        :param max_entangled_pair: int
                maximum number of entangled pairs we could have
        :param target_fidelity: float
                The purification target fidelity rate
        :param logger: utils.Logging.Logger
                The logger for the protocol
        :param is_top_layer: bool
                The flag to indicate if the protocol is the top layer. If True, the protocol will stop
                when all the entangled pairs are purified to the target fidelity.
        """

        super().__init__(node, name)

        # set variables
        self.entangled_node = entangled_nodes
        self.target_fidelity = target_fidelity
        self.cc_message_handler = cc_message_handler
        self.entanglement_handler = entanglement_handler
        # keep track of number of entangled pairs, help for termination condition if needed
        self.max_entangled_pair = max_entangled_pair
        # mapping of entangled qubits to memory positions key: node name, value: {memory position, fidelity}
        self.entangled_pairs = {node_name: {} for node_name in entangled_nodes}
        # mapping of satisfied pairs key: node name, value: {memory position, fidelity}
        self.satisfied_pairs = {node_name: {} for node_name in entangled_nodes}
        # store message from remote node
        # self.remote_message = []
        # currently purifying paris, store the memory position and fidelity
        self.purifying_paris = {}  # (p1, p2) -> (f1, f2)
        # self.purifying_results = {}  # (p1, p2) -> (M1, M2)

        # count of purification process
        # record how many purification pairs we have done in order to have all the pairs meet the target fidelity
        self.purification_count = {node_name: 0 for node_name in entangled_nodes}
        self.purification_success_count = {node_name: 0 for node_name in entangled_nodes}

        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        self.is_top_layer = is_top_layer

    def add_new_signal(self, signal):
        """
        Add new signal to the protocol
        :param signal: signal name
        :return:
        """
        self.add_signal(signal)

    def handle_entangle_signal(self, message):
        """
        Handle the entangle signal message, store the entangled pairs in the entangled_pairs
        :param message: utils.SignalMessages.EntangleSuccessSignalMessage
        :return: None
        """
        # store the entangled pairs
        entangled_node = message.entangled_node
        self.entangled_pairs[entangled_node][message.mem_pos] = message.initial_fidelity
        # TODO: check if we have enough entangled pairs to start purification? Here?

    def handle_purify_start_signal(self, message):
        """
        Handle the purification start signal message. We will have m1 from the source node
        We need to measure the m2 on our side and send the result back to the source node
        :param message: utils.SignalMessages.PurifyStartSignalMessage
        :return:
        """
        m1 = message.m1
        m2 = yield from self.purify_measurement(message.qubit1_pos, message.qubit2_pos, self.qmemory)
        self.purification_count[message.entangle_node] += 1
        # send the measurement result to the source node
        if m1 == m2:
            self.purification_success_count[message.entangle_node] += 1
            self.cc_message_handler.send_message(MessageType.PURIFICATION_RESULT,
                                                 self.entangled_node,
                                                 SignalMessages.PurifyResultSignalMessage(
                                                     entangle_node=self.node.name,
                                                     qubit1_pos=message.qubit1_pos,
                                                     qubit2_pos=message.qubit2_pos,
                                                     m2=m2,
                                                     result=True))
        else:
            self.cc_message_handler.send_message(MessageType.PURIFICATION_RESULT,
                                                 self.entangled_node,
                                                 SignalMessages.PurifyResultSignalMessage(
                                                     entangle_node=self.node.name,
                                                     qubit1_pos=message.qubit1_pos,
                                                     qubit2_pos=message.qubit2_pos,
                                                     m2=m2,
                                                     result=False))
        # remove the second pair from entangled pairs
        del self.entangled_pairs[message.entangle_node][message.qubit2_pos]
        # start re-entangle the second qubit
        self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE, SignalMessages.EntangleSignalMessage(
            entangle_node=message.entangle_node, mem_pos=message.qubit2_pos))

    def handle_purify_result_signal(self, message):
        """
        Handle the purification result signal message. We will have m2 from the source node and the result of the
        purification
        :param message: SignalMessages.PurifyResultSignalMessage
        :return:
        """
        self.purification_count[message.entangle_node] += 1
        if message.result:
            # purification is successful
            self.purification_success_count[message.entangle_node] += 1
            pair = (message.qubit1_pos, message.qubit2_pos)
            new_fidelity = self.calculate_purified_fidelity(self.purifying_paris[message.entangle_node][pair][0])
            # logging
            self.logger.info(f"Purify {self.name} -> Purification successful\n",
                             f"\tPair: {message.entangle_node} -> {pair}\n",
                             f"\tNew Fidelity: {new_fidelity}\n",
                             f"\tOld Fidelity: {self.purifying_paris[message.entangle_node][pair][0]}\n",
                             f"\tTarget Fidelity: {self.target_fidelity}",
                             color="green")
            if new_fidelity > self.target_fidelity:
                self.satisfied_pairs[message.entangle_node][message.qubit1_pos] = new_fidelity
                # send the target met signal to the remote node
                self.cc_message_handler.send_message(MessageType.PURIFICATION_TARGET_MET,
                                                     message.entangle_node,
                                                     SignalMessages.PurifyTargetMetSignalMessage(
                                                         entangle_node=self.node.name,
                                                         mem_pos=message.qubit1_pos,
                                                         new_fidelity=new_fidelity))
                # emit signal to upper layer
                self.send_signal(Signals.SUCCESS, SignalMessages.PurifyTargetMetSignalMessage(
                    entangle_node=message.entangle_node, mem_pos=message.qubit1_pos, new_fidelity=new_fidelity))
            else:
                self.entangled_pairs[message.entangle_node][message.qubit1_pos] = new_fidelity
        else:
            # purification failed
            self.logger.info(f"Purify {self.name} -> Purification failed\n",
                             f"\tPair: {message.entangle_node} -> {(message.qubit1_pos, message.qubit2_pos)}",
                             color="orange")
            self.entangled_pairs[message.entangle_node][message.qubit1_pos] = (
                self.purifying_paris)[message.entangle_node][0]
        # remove the pair from the purifying paris
        del self.purifying_paris[message.entangle_node][(message.qubit1_pos, message.qubit2_pos)]
        # remove the pair from the entangled pairs
        del self.entangled_pairs[message.entangle_node][message.qubit2_pos]
        # send the re-entangle signal to the source node
        self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE, SignalMessages.EntangleSignalMessage(
            entangle_node=message.entangle_node, mem_pos=message.qubit2_pos))

    def run(self):
        entangle_signal = self.await_signal(self.entanglement_handler, signal_label=Signals.SUCCESS)
        cc_message_signal = (self.await_signal(self.cc_message_handler, signal_label=MessageType.PURIFICATION_START) |
                             self.await_signal(self.cc_message_handler, signal_label=MessageType.PURIFICATION_RESULT) |
                             self.await_signal(self.cc_message_handler,
                                               signal_label=MessageType.PURIFICATION_TARGET_MET))
        while True:
            expr = yield entangle_signal | cc_message_signal
            if expr.first_term.value:
                # handle the entangle signal from the entanglement handler
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result: SignalMessages.EntangleSuccessSignalMessage = ready_signal.result
                    if ready_signal.signal_label == Signals.SUCCESS:
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received entangle signal: {result.__str__()}",
                                         color="blue")
                        self.handle_entangle_signal(result)

            elif expr.second_term.value:
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if ready_signal.signal_label == MessageType.PURIFICATION_START:
                        # start purification measurement
                        result: SignalMessages.PurifyStartSignalMessage
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received purification start signal:\n"
                                         f"\tFrom:{result.entangle_node}\n"
                                         f"\tQubit 1: {result.qubit1_pos}\n"
                                         f"\tQubit 2: {result.qubit2_pos}\n"
                                         f"\tM1: {result.m1}",
                                         color="yellow")
                        yield from self.handle_purify_start_signal(result)
                    elif ready_signal.signal_label == MessageType.PURIFICATION_RESULT:
                        # handle the purification result
                        result: SignalMessages.PurifyResultSignalMessage
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received purification result signal:\n"
                                         f"\tFrom:{result.entangle_node}\n"
                                         f"\tQubit 1: {result.qubit1_pos}\n"
                                         f"\tQubit 2: {result.qubit2_pos}\n"
                                         f"\tResult: {result.result}\n",
                                         f"\tM2: {result.m2}",
                                         color="yellow")
                        yield from self.handle_purify_result_signal(result)
                    elif ready_signal.signal_label == MessageType.PURIFICATION_TARGET_MET:
                        # handle the purification target met signal
                        result: SignalMessages.PurifyTargetMetSignalMessage
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received purification target met signal:\n"
                                         f"\tFrom:{result.entangle_node}\n"
                                         f"\tMem pos: {result.mem_pos}\n"
                                         f"\tFidelity: {result.fidelity}",
                                         color="green")
                        self.satisfied_pairs[result.entangle_node][result.mem_pos] = result.fidelity
                        # emit signal to upper layer
                        self.send_signal(Signals.SUCCESS, SignalMessages.PurifyTargetMetSignalMessage(
                            entangle_node=result.entangle_node, mem_pos=result.mem_pos,
                            new_fidelity=result.fidelity))
                        # delete the pair from the entangled pairs
                        del self.entangled_pairs[result.entangle_node][result.mem_pos]
                # check if we have enough entangled pairs to start purification
                for node_name, pairs in self.entangled_pairs.items():
                    if len(pairs) >= 2:
                        yield from self.start_purification(node_name)
                # check if we have enough entangled pairs to start purification
                self.print_status()
                # finished purifying all the pairs
                # TODO: what case we can say we are done? Current end condition is when we have no entangled pairs == 0
                # however, we still have one pair that is not purified which is being sent to re-entangle,
                # but odd number of pairs will always have one pair that is not purified
                if self.is_top_layer:
                    no_entangled_pairs = sum([len(pairs) for pairs in self.entangled_pairs.values()]) <= 1
                    no_purifying_pairs = sum([len(pairs) for pairs in self.purifying_paris.values()]) == 0
                    all_satisfied_pairs = sum([len(pairs) for pairs in self.satisfied_pairs.values()]) == (
                            self.max_entangled_pair - 2) * len(self.entangled_node)
                    if no_entangled_pairs and no_purifying_pairs and all_satisfied_pairs:
                        self.send_signal(Signals.SUCCESS, {"satisfied_pairs": self.satisfied_pairs,
                                                           "purification_count": self.purification_count,
                                                           "purification_success_count": self.purification_success_count,
                                                           "finish_time": sim_time()})
                        break

    def print_status(self):
        self.logger.info(f"Purify {self.name} -> Node {self.name} entangled pairs:\n"
                         f"\tNodes:\n"
                         f"\t\tEntangled Pairs:\n"
                         f"{self.paris_to_string(self.entangled_pairs)}\n"
                         f"\t\tSatisfied Pairs:\n"
                         f"{self.paris_to_string(self.satisfied_pairs)}\n"
                         f"\tPurifying Pairs:\n"
                         f"{self.paris_to_string(self.purifying_paris)}",)

    @staticmethod
    def paris_to_string(pairs):
        pair_str = ""
        for node_name, pairs in pairs.items():
            pair_str += f"\t\t\t{node_name}:\n"
            pair_str += "\n".join([f"\t\t\t\tMemory Position: {k}, Fidelity: {v}" for k, v in pairs.items()])

    def start_purification(self, entangled_node):
        """
        Start the purification protocol.
        1. Check if we have enough entangled pairs (in right entangled pairs)
        2. Pick 2 pairs randomly (can be change) to start purification process
        3. Send classical message to the right neighbour to start purification process, with the memory positions

        Sequence of events:
        A -(purify start)-> B
            - A will start purification measurement with (q1, q2) and send the result to node_B
        B -(purify result)-> A
            - Once B receives the message, it will start the purification measurement with (q1, q2)
             and send the result to A with T/F
            - B will start re-entangle the second qubit as it is sacrificed
            - If T, node_A will store the new fidelity and check if it meets the target fidelity
            - If F, node_A will store the old fidelity and re-add to entangled pairs
            - A will start re-entangle the second qubit as it is sacrificed
        A -(purify target met)-> B
            - If the target fidelity is met, A notifies B to remove the pair from the entangled pairs and
            add to satisfied pairs
        @param entangled_node: str
                Entangled node we should do the purification with
        :return:
        """
        # print(f"Purify {self.name} -> Node {self.node.name} Starting purification process")
        # pick 2 pairs randomly
        # TODO maybe we can pick the pairs with one pair with the lowest fidelity and one with the highest fidelity?
        pairs = list(self.entangled_pairs[entangled_node].keys())
        # shuffle the pairs
        np.random.shuffle(pairs)
        pair1 = pairs[0]
        pair2 = pairs[1]

        self.logger(f"Purify {self.name} -> Node {self.node.name} Starting purification for pairs: {pair1}, {pair2}",
                    color="yellow")

        # store the pairs we are purifying
        self.purifying_paris[entangled_node][(pair1, pair2)] = (self.entangled_pairs[entangled_node][pair1],
                                                                self.entangled_pairs[entangled_node][pair2])
        del self.entangled_pairs[pair1]
        del self.entangled_pairs[pair2]

        # start purification meausrement
        m1 = yield from self.purify_measurement(pair1, pair2, self.qmemory)

        # send classical message to the right neighbour
        self.cc_message_handler.send_message(MessageType.PURIFICATION_START,
                                             entangled_node,
                                             SignalMessages.PurifyStartSignalMessage(
                                                 entangle_node=self.node.name,
                                                 qubit1_pos=pair1,
                                                 qubit2_pos=pair2,
                                                 m1=m1))

    def purify_measurement(self, q1_pos, q2_pos, entangled_node):
        """
        Perform purification measurement on the qubits
        :param q1_pos: qubit 1 memory position
        :param q2_pos: qubit 2 memory position
        :param entangled_node: name of the entangled node to get the qmemory
        :return: measurement result
        """
        qmemory = self.subcomponents[f"{entangled_node}_qmemory"]
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

    @staticmethod
    def calculate_purified_fidelity(initial_fidelity):
        F = initial_fidelity
        numerator = F ** 2 + (1 / 9) * (1 - F) ** 2
        denominator = F ** 2 + (2 / 3) * F * (1 - F) + (5 / 9) * (1 - F) ** 2
        new_fidelity = numerator / denominator
        return new_fidelity

    def reset(self):
        # clean up the pairs
        self.entangled_pairs = {node_name: {} for node_name in self.entangled_node}
        # map of entangled pairs with higher fidelity
        self.satisfied_pairs = {node_name: {} for node_name in self.entangled_node}
        # temporary pairs
        self.purifying_paris = {node_name: {} for node_name in self.entangled_node}
        # count of purification process
        self.purification_count = {node_name: 0 for node_name in self.entangled_node}
        self.purification_success_count = {node_name: 0 for node_name in self.entangled_node}
        super().reset()

    def stop(self):
        super().stop()
