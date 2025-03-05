import copy
import gc

from netsquid.components import INSTR_MEASURE
from netsquid.util.simtools import sim_time
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_CNOT, INSTR_H

from protocols.MessageHandler import MessageType
from utils import Logging, SignalMessages
from utils.ClassicalMessages import ClassicalMessage


class Purification(NodeProtocol):
    """
    Protocol for entanglement purification.
    Only care about the entangled node, no need to care about left and right nodes.
    """

    def __init__(self,
                 node,
                 name,
                 entangled_node,
                 entanglement_handler,
                 cc_message_handler,
                 max_purify_pair=10,
                 target_fidelity=0.9,
                 logger=None,
                 is_top_layer=False):
        """
        Initialization of purification protocol
        :param node: `~netsquid.nodes.node.Node`
                The node which the protocol is running.
        :param name: str
                Name of the protocol.
        :param entangled_node: str
                The name of the entangled nodes we are purifying with
        :param entanglement_handler: protocols.EntanglementHandler
                The entanglement handler protocol for the node
        :param cc_message_handler: protocols.MessageHandler
                The classical message handler protocol
        :param max_purify_pair: int
                maximum number of purified pairs we could have
        :param target_fidelity: float
                The purification target fidelity rate
        :param logger: utils.Logging.Logger
                The logger for the protocol
        :param is_top_layer: bool
                The flag to indicate if the protocol is the top layer. If True, the protocol will stop
                when all the entangled pairs are purified to the target fidelity.
        """

        super().__init__(node=node, name=name)

        # set variables
        self.entangled_node = entangled_node
        self.target_fidelity = target_fidelity
        self.cc_message_handler = cc_message_handler
        self.entanglement_handler = entanglement_handler
        # keep track of number of entangled pairs, help for termination condition if needed
        self.max_purify_pair = max_purify_pair
        # record are we source node or not
        self.is_source_node = False
        # mapping of entangled qubits to memory positions key: memory position, value: fidelity
        self.entangled_pairs = {}
        # mapping of satisfied pairs key: memory position, value: fidelity
        self.satisfied_pairs = {}
        # store message from remote node
        self.classical_messages_queue = []
        # currently purifying paris, store the memory position and fidelity
        self.purifying_paris = {}  # (p1, p2) -> (f1, f2)
        # count of purification process
        # record how many purification pairs we have done in order to have all the pairs meet the target fidelity
        self.purification_count = 0
        self.purification_success_count = 0
        # graceful shutdown from uppler's termination
        self.shutdown = False
        self.start_time = sim_time()
        # logger
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=False)
        else:
            self.logger = logger

        self.is_top_layer = is_top_layer
        # adding necessary signals
        self.add_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER)
        self.add_signal(MessageType.PURIFICATION_FINISHED)
        self.add_signal(MessageType.PURIFICATION_SUCCESS)

    def handle_entangle_signal(self, message):
        """
        Handle the entangle signal message, store the entangled pairs in the entangled_pairs
        :param message: utils.SignalMessages.EntangleSuccessSignalMessage
        :return: None
        """
        # store the entangled pairs
        entangled_node = message.entangle_node
        # if we have the entangled pair satisfied the target fidelity, we will store it in the satisfied pairs
        self.is_source_node = message.is_source
        # account for concurrent entanglement
        purify_success_poses = []
        purify_success_fids= []
        if type(message.mem_pos) is list:
            for pos, fidelity in zip(message.mem_pos, message.fidelity):
                if fidelity > self.target_fidelity:
                    self.satisfied_pairs[pos] = fidelity
                    purify_success_poses.append(pos)
                    purify_success_fids.append(fidelity)
                    if not self.is_top_layer:
                        # remove this info from our record keeping
                        del self.satisfied_pairs[pos]
                else:
                    self.entangled_pairs[pos] = fidelity
        else:
            if message.fidelity > self.target_fidelity:
                self.satisfied_pairs[message.mem_pos] = message.fidelity
                if not self.is_top_layer:
                    # remove this info from our record keeping
                    del self.satisfied_pairs[message.mem_pos]
                purify_success_poses.append(message.mem_pos)
                purify_success_fids.append(message.fidelity)
            else:
                self.entangled_pairs[message.mem_pos] = message.fidelity

        # send signal to upper layer
        if len(purify_success_poses) > 0:
            self.send_signal(MessageType.PURIFICATION_SUCCESS, SignalMessages.PurifySuccessSignalMessage(
                source_node=self.node.name,
                entangle_node=entangled_node, mem_pos=purify_success_poses, new_fidelity=purify_success_fids,
                is_source=message.is_source))


    def handle_purify_start_signal(self, message):
        """
        Handle the purification start signal message. We will have m1 from the source node
        We need to measure the m2 on our side and send the result back to the source node
        :param message: utils.SignalMessages.PurifyStartSignalMessage
        :return:
        """
        # check if we have the qubits in the memory
        if message.qubit1_pos not in self.entangled_pairs or \
                message.qubit2_pos not in self.entangled_pairs:
            self.logger.error(f"Purify {self.name} -> "
                              f"Node {self.node.name} does not have the qubits in the memory for {message.__dict__}",
                              color="red")
            self.classical_messages_queue.append(message)
            return
        m1 = message.m1
        m2 = yield from self.purify_measurement(message.qubit1_pos, message.qubit2_pos, message.entangle_node)
        self.purification_count += 1
        # send the measurement result to the source node
        if m1 == m2:
            self.logger.info(f"Purify {self.name} -> Purification successful\n"
                             f"\tPair: {message.entangle_node} -> {(message.qubit1_pos, message.qubit2_pos)}",
                             color="yellow")
            # Calculate the fidelity
            new_fidelity = self.calculate_purified_fidelity(self.entangled_pairs[message.qubit1_pos],
                                                            self.entangled_pairs[message.qubit2_pos])
            self.entangled_pairs[message.qubit1_pos] = new_fidelity
            if new_fidelity > self.target_fidelity:
                self.satisfied_pairs[message.qubit1_pos] = new_fidelity
                del self.entangled_pairs[message.qubit1_pos]

                self.send_signal(MessageType.PURIFICATION_SUCCESS, SignalMessages.PurifySuccessSignalMessage(
                    source_node=self.node.name,
                    entangle_node=message.entangle_node, mem_pos=message.qubit1_pos,
                    new_fidelity=new_fidelity,
                    is_source=False))
                # remove info for non-top layer
                if not self.is_top_layer:
                    # remove this info from our record keeping
                    del self.satisfied_pairs[message.qubit1_pos]
                self.print_status()

            self.purification_success_count += 1
            self.cc_message_handler.send_message(MessageType.PURIFICATION_RESULT,
                                                 message.entangle_node,
                                                 ClassicalMessage(
                                                     self.node.name,
                                                     message.entangle_node,
                                                     SignalMessages.PurifyResultSignalMessage(
                                                         entangle_node=self.node.name,
                                                         qubit1_pos=message.qubit1_pos,
                                                         qubit2_pos=message.qubit2_pos,
                                                         m2=m2,
                                                         result=True))
                                                 )
            # # remove the second pair from entangled pairs
            # del self.entangled_pairs[message.entangle_node][message.qubit2_pos]
            # # start re-entangle the second qubit
            # self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE, SignalMessages.EntangleSignalMessage(
            #     entangle_node=message.entangle_node, mem_pos=message.qubit2_pos))

            self.re_entangle(message.entangle_node, [message.qubit2_pos])
        else:
            self.logger.info(f"Purify {self.name} -> Purification failed\n"
                             f"\tPair: {message.entangle_node} -> {(message.qubit1_pos, message.qubit2_pos)}",
                             color="red")
            self.cc_message_handler.send_message(MessageType.PURIFICATION_RESULT,
                                                 message.entangle_node,
                                                 ClassicalMessage(
                                                     self.node.name,
                                                     message.entangle_node,
                                                     SignalMessages.PurifyResultSignalMessage(
                                                         entangle_node=self.node.name,
                                                         qubit1_pos=message.qubit1_pos,
                                                         qubit2_pos=message.qubit2_pos,
                                                         m2=m2,
                                                         result=False)
                                                 ))
            # del self.entangled_pairs[message.entangle_node][message.qubit1_pos]
            # self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE, SignalMessages.EntangleSignalMessage(
            #     entangle_node=message.entangle_node, mem_pos=message.qubit1_pos))
            # remove the second pair from entangled pairs
            # del self.entangled_pairs[message.entangle_node][message.qubit2_pos]
            # # start re-entangle the second qubit
            # self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE, SignalMessages.EntangleSignalMessage(
            #     entangle_node=message.entangle_node, mem_pos=message.qubit2_pos))
            # TODO: do we need to re-entangle the first qubit?
            self.re_entangle(message.entangle_node, [message.qubit1_pos, message.qubit2_pos])

    def handle_purify_result_signal(self, message):
        """
        Handle the purification result signal message. We will have m2 from the source node and the result of the
        purification
        :param message: SignalMessages.PurifyResultSignalMessage
        :return:
        """
        self.purification_count += 1
        if message.result:
            # purification is successful
            self.purification_success_count += 1
            pair = (message.qubit1_pos, message.qubit2_pos)
            new_fidelity = self.calculate_purified_fidelity(self.purifying_paris[pair][0],
                                                            self.purifying_paris[pair][1])
            # logging
            self.logger.info(f"Purify {self.name} -> Purification successful\n"
                             f"\tPair: {message.entangle_node} -> {pair}\n"
                             f"\tNew Fidelity: {new_fidelity}\n"
                             f"\tOld Fidelity: {self.purifying_paris[pair][0]}\n"
                             f"\tTarget Fidelity: {self.target_fidelity}",
                             color="green")
            if new_fidelity > self.target_fidelity:
                self.satisfied_pairs[message.qubit1_pos] = new_fidelity
                # send the target met signal to the remote node
                # self.cc_message_handler.send_message(MessageType.PURIFICATION_TARGET_MET,
                #                                      message.entangle_node,
                #                                      ClassicalMessage(
                #                                          self.node.name,
                #                                          message.entangle_node,
                #                                          SignalMessages.PurifyTargetMetSignalMessage(
                #                                              source_node=self.node.name,
                #                                              entangle_node=self.node.name,
                #                                              mem_pos=message.qubit1_pos,
                #                                              new_fidelity=new_fidelity)
                #                                      )
                #                                      )
                # emit signal to upper layer
                self.send_signal(MessageType.PURIFICATION_SUCCESS, SignalMessages.PurifySuccessSignalMessage(
                    source_node=self.node.name,
                    entangle_node=message.entangle_node,
                    mem_pos=message.qubit1_pos,
                    new_fidelity=new_fidelity,
                    is_source=True))
                if not self.is_top_layer:
                    # remove this info from our record keeping
                    del self.satisfied_pairs[message.qubit1_pos]
            else:
                self.entangled_pairs[message.qubit1_pos] = new_fidelity
            # re-entangle the second qubit
            self.re_entangle(self.entangled_node, [message.qubit2_pos])
        else:
            # purification failed
            self.logger.info(f"Purify {self.name} -> Purification failed\n"
                             f"\tPair: {message.entangle_node} -> {(message.qubit1_pos, message.qubit2_pos)}",
                             color="red")
            # TODO: do we need to re-entangle the first qubit?
            self.re_entangle(self.entangled_node, [message.qubit1_pos, message.qubit2_pos])
            # add the pair back to the entangled pairs
            # self.entangled_pairs[message.entangle_node][message.qubit1_pos] \
            #     = self.purifying_paris[message.entangle_node][(message.qubit1_pos, message.qubit2_pos)][0]
        # remove the pair from the purifying paris
        del self.purifying_paris[(message.qubit1_pos, message.qubit2_pos)]

    def process_classical_message(self):
        temp = self.classical_messages_queue
        self.classical_messages_queue = []
        for message in temp:
            if isinstance(message, SignalMessages.PurifyStartSignalMessage):
                yield from self.handle_purify_start_signal(message)
            elif isinstance(message, SignalMessages.PurifyTargetMetSignalMessage):
                self.handle_purify_target_met_signal(message)

    def handle_purify_target_met_signal(self, message):
        """
        Handle the purification target met signal message. We will have the new fidelity from the source node
        :param message: SignalMessages.PurifyTargetMetSignalMessage
        :return:
        """
        if message.mem_pos not in self.entangled_pairs:
            self.logger.error(f"Purify {self.name} -> "
                              f"Node {self.node.name} does not have the qubits in the memory for {message.__dict__}\n"
                              f"Source:{self.is_source_node}"
                              , color="red")
            self.print_status()
            self.classical_messages_queue.append(message)
            return
        self.logger.info(f"Purify {self.name} -> Purification target met processed\n"
                         f"Node{self.node.name} \n"
                         f"Entangled: {message.entangle_node}\n"
                         f"MemPos: {message.mem_pos}\n"
                         f"Fid: {message.fidelity}", color="green")
        self.print_status()
        self.satisfied_pairs[message.mem_pos] = message.fidelity
        # emit signal to upper layer
        self.send_signal(MessageType.PURIFICATION_SUCCESS, SignalMessages.PurifySuccessSignalMessage(
            source_node=self.node.name,
            entangle_node=message.entangle_node, mem_pos=message.mem_pos,
            new_fidelity=message.fidelity,
            is_source=False))
        # delete the pair from the entangled pairs
        del self.entangled_pairs[message.mem_pos]
        # remove info for non-top layer
        if not self.is_top_layer:
            # remove this info from our record keeping
            del self.satisfied_pairs[message.mem_pos]
        self.print_status()

    def re_entangle(self, entangle_node, mem_poses):
        """
        Re-entangle the memory positions
        :param entangle_node: str
                The entangled node name
        :param mem_poses: list
                A List of memory positions to re-entangle
        :return:
        """
        for mem_pos in mem_poses:
            # remove the pair from the entangled pairs
            if mem_pos in self.entangled_pairs:
                del self.entangled_pairs[mem_pos]
            if mem_pos in self.satisfied_pairs:
                del self.satisfied_pairs[mem_pos]
        # re-entangle the memory position
        self.cc_message_handler.send_signal(MessageType.RE_ENTANGLE_FROM_UPPER_LAYER,
                                            SignalMessages.ReEntangleSignalMessage(
            entangle_node=entangle_node, re_entangle_mem_poses=mem_poses, is_source=self.is_source_node))

    def run(self):
        entangle_signal = self.await_signal(self.entanglement_handler, signal_label=MessageType.ENTANGLED_SUCCESS)
        cc_message_signal = (self.await_signal(self.cc_message_handler, signal_label=MessageType.PURIFICATION_START) |
                             self.await_signal(self.cc_message_handler, signal_label=MessageType.PURIFICATION_RESULT) |
                             self.await_signal(self.cc_message_handler,
                                               signal_label=MessageType.PURIFICATION_TARGET_MET)
                             )
        yield self.await_timer(1)
        self.start_time = sim_time()
        while True:
            expr = yield entangle_signal | cc_message_signal
            if expr.first_term.value:
                # handle the entangle signal from the entanglement handler
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                    except Exception as e:
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} failed to get signal: {e}",
                                         color="red")
                        continue
                    result: SignalMessages.EntangleSuccessSignalMessage = ready_signal.result
                    if result.timestamp < self.start_time:
                        continue
                    if ready_signal.label == MessageType.ENTANGLED_SUCCESS:
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received entangle signal: {result.__dict__}",
                                         color="blue")
                        if result.entangle_node != self.entangled_node:
                            continue
                        self.handle_entangle_signal(result)

            elif expr.second_term.value:
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                        result: ClassicalMessage = ready_signal.result
                        if result is None:
                            continue
                    except Exception as e:
                        self.logger.error(f"Purify {self.name} -> "
                                          f"Node {self.node.name} error processing classical message: {e}",
                                          color="red")
                        continue
                    if isinstance(result, ClassicalMessage):
                        if result.from_node != self.entangled_node:
                            # we do not care about the message from other nodes
                            continue
                        if result.data.timestamp < self.start_time:
                            # discard race condition where previous round info just arrived
                            continue
                    if ready_signal.label == MessageType.PURIFICATION_START:
                        # start purification measurement
                        result: SignalMessages.PurifyStartSignalMessage = result.data
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received purification start signal:\n"
                                         f"\tFrom:{result.entangle_node}\n"
                                         f"\tQubit 1: {result.qubit1_pos}\n"
                                         f"\tQubit 2: {result.qubit2_pos}\n"
                                         f"\tM1: {result.m1}",
                                         color="yellow")
                        yield from self.handle_purify_start_signal(result)
                    elif ready_signal.label == MessageType.PURIFICATION_RESULT:
                        # handle the purification result
                        result: SignalMessages.PurifyResultSignalMessage = result.data
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received purification result signal:\n"
                                         f"\tFrom:{result.entangle_node}\n"
                                         f"\tQubit 1: {result.qubit1_pos}\n"
                                         f"\tQubit 2: {result.qubit2_pos}\n"
                                         f"\tResult: {result.result}\n"
                                         f"\tM2: {result.m2}",
                                         color="yellow")
                        self.print_status()
                        self.handle_purify_result_signal(result)
                    elif ready_signal.label == MessageType.PURIFICATION_TARGET_MET:
                        # handle the purification target met signal
                        result: SignalMessages.PurifyTargetMetSignalMessage = result.data
                        self.logger.info(f"Purify {self.name} -> "
                                         f"Node {self.node.name} received purification target met signal:\n"
                                         f"\tFrom:{result.entangle_node}\n"
                                         f"\tMem pos: {result.mem_pos}\n"
                                         f"\tFidelity: {result.fidelity}",
                                         color="green")
                        self.handle_purify_target_met_signal(result)
                    # elif ready_signal.label == MessageType.RE_ENTANGLE_FROM_UPPER_LAYER:
                    #     # handle the re-entangle signal from the upper layer
                    #     result: SignalMessages.ReEntangleSignalMessage = result
                    #     # skip if the re-entangle is not for us
                    #     if result.entangle_node != self.entangled_node:
                    #         continue
                    #     self.logger.info(f"Purify {self.name} -> "
                    #                      f"Node {self.node.name} received re-entangle from upper layer signal:\n"
                    #                      f"\tFrom:{result.entangle_node}\n"
                    #                      f"\tMem pos: {result.re_entangle_mem_poses}",
                    #                      color="green")
                    #     self.re_entangle(result.entangle_node, result.re_entangle_mem_poses)

            # check status
            self.print_status()
            # finished purifying all the pairs
            # TODO: what case we can say we are done? Current end condition is when we have no entangled pairs == 0
            # however, we still have one pair that is not purified which is being sent to re-entangle,
            # but odd number of pairs will always have one pair that is not purified
            if self.is_top_layer and self.check_end_condition():
                self.logger.info(f"Purify {self.name} -> Node {self.name} Finished purification process",
                                 color="green")
                self.print_status(color="green")
                # send signal to local protocol
                self.send_signal(MessageType.PURIFICATION_FINISHED, {"satisfied_pairs": self.satisfied_pairs,
                                                                     "purification_count": self.purification_count,
                                                                     "purification_success_count":
                                                                         self.purification_success_count,
                                                                     "finish_time": sim_time()})
                break
            # start purification process from the source node
            if len(self.entangled_pairs) >= 2 and self.is_source_node:
                # we are checking if we have engouh pairs to start purification
                # AND we are the source node, which means we have the initial fidelity
                if not self.shutdown:
                    yield from self.start_purification(self.entangled_node)
            # clear the classical message queue
            yield from self.process_classical_message()



    def print_status(self, color="cyan"):
        self.logger.info(f"Purify {self.name} -> Node {self.name} entangled pairs:\n"
                         f"\tEntangled Pairs:\n"
                         f"{self.paris_to_string(self.entangled_pairs)}\n"
                         f"\tSatisfied Pairs:\n"
                         f"{self.paris_to_string(self.satisfied_pairs)}\n"
                         f"\tPurifying Pairs:\n"
                         f"{self.paris_to_string(self.purifying_paris)}\n"
                         , color=color)

    def check_end_condition(self):
        """
        Check if the purification protocol has finished
        The end condition as follows:
        1. We have reached target purified pairs

        :return: bool
                True if the protocol has finished, False otherwise
        """
        return len(self.satisfied_pairs) == self.max_purify_pair

    @staticmethod
    def paris_to_string(pairs):
        pair_str = ""
        pair_str += "\n".join([f"\t\t\tMemory Position: {k}, Fidelity: {v}" for k, v in pairs.items()])
        return pair_str

    def start_purification(self, entangled_node):
        """
        Start the purification protocol.
        1. Check if we have enough entangled pairs (in right entangled pairs)
        2. Pick one highest and one lowest to start purification process
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
        # pick 2 pairs one high fidelity and one low fidelity
        pairs = list(self.entangled_pairs.keys())
        # sort the pairs by fidelity
        pairs = sorted(pairs, key=lambda x: self.entangled_pairs[x], reverse=True)
        # f1 is highest
        pair1 = pairs[0]
        # f2 is second highest
        pair2 = pairs[1]

        if pair1 is None or pair2 is None:
            self.logger.info(f"Purify {self.name} -> Node {self.node.name} No pairs to purify", color="green")
            return
        self.logger.info(
            f"Purify {self.name} -> Node {self.node.name} Starting purification for pairs: {pair1},{pair2}"
            f"\n\tFidelity: {self.entangled_pairs[pair1]}, "
            f"{self.entangled_pairs[pair2]}",
            color="yellow")

        # store the pairs we are purifying
        self.purifying_paris[(pair1, pair2)] = (self.entangled_pairs[pair1],
                                                self.entangled_pairs[pair2])
        del self.entangled_pairs[pair1]
        del self.entangled_pairs[pair2]

        # start purification meausrement
        m1 = yield from self.purify_measurement(pair1, pair2, entangled_node)
        self.logger.info(f"Purify {self.name} -> Node {self.node.name} start purification measurement\n"
                         f"\tPair: {entangled_node} -> {(pair1, pair2)}\n"
                         f"\tM1: {m1}",
                         color="yellow")
        self.print_status()
        # send classical message to the remote node to start purification
        self.cc_message_handler.send_message(MessageType.PURIFICATION_START,
                                             entangled_node,
                                             ClassicalMessage(
                                                 self.node.name,
                                                 entangled_node,
                                                 SignalMessages.PurifyStartSignalMessage(
                                                     entangle_node=self.node.name,
                                                     qubit1_pos=pair1,
                                                     qubit2_pos=pair2,
                                                     m1=m1)
                                             ))

    def purify_measurement(self, q1_pos, q2_pos, entangled_node):
        """
        Perform purification measurement on the qubits
        :param q1_pos: qubit 1 memory position
        :param q2_pos: qubit 2 memory position
        :param entangled_node: name of the entangled node to get the qmemory
        :return: measurement result
        """
        qmemory = self.node.subcomponents[f"{entangled_node}_qmemory"]
        # Handle incoming Qubit on this node.
        if qmemory.busy:
            yield self.await_program(qmemory)
        # Apply CNOT gate to qubit 1 and qubit 2
        self.logger.info(
            f"Purify {self.name} -> Node {self.node.name} Applying CNOT gate to qubits {q1_pos} and {q2_pos}\n"
            f"\tTime:{sim_time()}\n"
            f"\tMemName:{qmemory.name}",
            color="yellow")
        try:
            qmemory.execute_instruction(INSTR_CNOT, [q1_pos, q2_pos])
        except Exception as e:
            self.logger.error(e)
        # Apply Hadamard gate to qubit 1
        if qmemory.busy:
            yield self.await_program(qmemory)
        # self.logger.info(f"Purify {self.name} -> Node {self.node.name} Applying Hadamard gate to qubit {q1_pos}",
        #                  color="yellow")
        # qmemory.execute_instruction(INSTR_H, [q1_pos])
        # Measure qubit 2
        self.logger.info(f"Purify {self.name} -> Node {self.node.name} Measuring qubit {q2_pos}",
                         color="yellow")
        if qmemory.busy:
            yield self.await_program(qmemory)
        # measured_result, _ = qmemory.measure(q2_pos)
        measured_result, _ = qmemory.execute_instruction(INSTR_MEASURE, [q2_pos], output_key="M")
        # TODO: can we remove q2 from the memory? since we measured it
        res = measured_result["M"][0]
        return res

    @staticmethod
    def calculate_purified_fidelity(f1, f2):
        """
        Calculate the new fidelity after purification with two pairs of non-identical fidelity
        10*f1*f2 - f1 - f2 + 1 / 8f1f2 - 2f1 - 2f2+5
        """
        # F = initial_fidelity
        # numerator = F ** 2 + (1 / 9) * (1 - F) ** 2
        # denominator = F ** 2 + (2 / 3) * F * (1 - F) + (5 / 9) * (1 - F) ** 2
        # new_fidelity = numerator / denominator
        new_fidelity = (10 * f1 * f2 - f1 - f2 + 1) / (8 * f1 * f2 - 2 * f1 - 2 * f2 + 5)
        return new_fidelity

    def clear_info(self):
        # record are we source node or not
        self.is_source_node = False
        # mapping of entangled qubits to memory positions key: memory position, value: fidelity
        self.entangled_pairs = {}
        # mapping of satisfied pairs key: memory position, value: fidelity
        self.satisfied_pairs = {}
        # store message from remote node
        self.classical_messages_queue = []
        # currently purifying paris, store the memory position and fidelity
        self.purifying_paris = {}  # (p1, p2) -> (f1, f2)
        # count of purification process
        # record how many purification pairs we have done in order to have all the pairs meet the target fidelity
        self.purification_count = 0
        self.purification_success_count = 0
        # reset the shutdown flag and finished flag
        self.shutdown = False

    def reset(self):
        self.logger.info(f"Purify {self.name} -> Node {self.node.name} Resetting purification protocol", color="red")
        self.clear_info()
        super().reset()

    def stop(self):
        self.logger.info(f"Purify {self.name} -> Node {self.node.name} Stop purification protocol", color="red")
        self.clear_info()
        super().stop()
