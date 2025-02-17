import netsquid as ns
from netsquid import sim_time

from netsquid.qubits import ketstates as ks
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import INSTR_SWAP
from netsquid.components.qsource import QSource

from utils import Logging, SignalMessages
from protocols.MessageHandler import MessageType


class GenEntanglement(NodeProtocol):
    """
    Protocol for generating entanglement between two nodes.
    We only care about the at node level and with respect to the qmemory.
    """

    def __init__(self, node, name=None,
                 entangle_handler=None,
                 input_mem_pos=0,
                 total_pairs=2,
                 entangle_node=None,
                 is_source=False,
                 logger=None):
        """
        Initialize the GenEntanglementProtocol.
        @param node: `netsquid.nodes.node.Node`
                    The node that the protocol is attached to
        @param name: str
                    The name of the protocol
        @param entangle_handler: `protocols.EntanglementHandler`
                    The protocol that will send the entangle and re-entangle signal.
        @param input_mem_pos: int
                    The memory position to use as input
        @param total_pairs: int
                    The total number of entangled pairs to generate
        @param entangle_node: str
                    The name of the node to entangle with
        @param is_source: bool
                    If the node is the source of the entanglement, if True, the node will generate qubits
        @param logger: `utils.Logging.Logger`
                    The logger to use for logging
        """
        if entangle_node is None:
            raise ValueError("Entangle node must be specified.")

        name = name if name else ("DirectionalEntangleNode({}, left={}, right={})"
                                  .format(node.name,
                                          self.left,
                                          self.right))

        super().__init__(node=node, name=name)

        if entangle_handler is not None and not isinstance(entangle_handler, NodeProtocol):
            raise TypeError("Start expression should be a {}, not a {}".format(
                NodeProtocol, type(entangle_handler)))
        self.entanglement_handler = entangle_handler
        self.aval_mem_postions = None  # stack of available memory positions
        self.used_mem_positions = None  # stack of used memory positions
        self._total_pairs = total_pairs
        self._input_mem_pos = input_mem_pos
        self.entangle_node = entangle_node
        self._is_source = is_source
        # keep track of when we started the protocol. we can use this avoid process old data
        self.start_time = None
        # re-entangle helper
        # TODO: can we combine with to aval_mem_positions? Since we are now aligning the memory positions
        self.re_entangle_pos = []
        # keep a list of generated fidelity we can pop each when we want to forward it to upper layer
        self.generated_fidelity = []
        # get the qmemory
        try:
            self._qmemory_name = f"{entangle_node}_qmemory"
            self.qmemory = self.node.subcomponents[self._qmemory_name]
        except KeyError:
            raise ValueError("Qmemory {} not found in node {}.".format(self._qmemory_name, node))
        # case of is source we need to add port to send qubits
        if self._is_source:
            self._qport = self.node.subcomponents["internal_qchannel"].ports["send"]

        # Qmemory input port is used to take received qubits from entangle node
        self._qmem_input_port = None
        if self._input_mem_pos is not None:
            self._qmem_input_port = self.qmemory.ports["qin{}".format(self._input_mem_pos)]
            self.qmemory.mem_positions[self._input_mem_pos].in_use = True

        # logger setup
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        # add signal for re-entangle ready
        self.add_signal(MessageType.RE_ENTANGLE_READY)
        self.add_signal(MessageType.GEN_ENTANGLE_READY)
        self.add_signal(MessageType.RE_ENTANGLE_READY_SOURCE)

    def run(self):
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                         f"\tentangle node: {self.entangle_node}\n"
                         f"\tmemory available positions: {self.aval_mem_postions}\n"
                         f"\tmemory used positions: {self.used_mem_positions}"
                         f"\ttotal pairs: {self._total_pairs}"
                         f"\tinput memory position: {self._input_mem_pos}\n"
                         f"\tStarting Time: {ns.sim_time()}"
                         )

        if self.entanglement_handler is None:
            raise ValueError("Re-entangle sender must be specified.")
        self.start_time = ns.sim_time()
        # the EntanglementHandler will send the {self.entangle_node}_re_entangle and we will listen for it
        # the EntanglementHandler will also send the {self.entangle_node}_re_entangle_ready and we will listen for it

        if self._is_source:
            self.signal_watcher = ReEntangleSignalWatcher(self.node, self,
                                                          self.entanglement_handler,
                                                          f"{self.name}_re_entangle_ready",
                                                          f"{self.name}_signal_watcher",
                                                          self.logger)
        else:
            self.signal_watcher = ReEntangleSignalWatcher(self.node, self,
                                                          self.entanglement_handler,
                                                          f"{self.name}_re_entangle",
                                                          f"{self.name}_signal_watcher",
                                                          self.logger)
        self.qubit_watcher = QubitSignalWatcher(self.node, f"{self.name}_watcher",
                                                self._qmemory_name,
                                                self._qmem_input_port, self,
                                                self.logger)
        self.qubit_generator = QubitGenerationProtocol(self.node, f"{self.name}_generator", self, self.logger)

        self.qubit_watcher.start()
        self.signal_watcher.start()
        self.qubit_generator.start()

        yield self.await_timer(5001)
        # start the main logic
        while True:
            # the logic that we generate qubits and send them to the entangle node
            # if is source we generate qubits and send them to the entangle node
            # if not source we receive qubits from the entangle node
            # print(f"GenEntangle {self.name} -> Node {self.node.name} had entangle node {self.entangle_node}.")
            initial_fidelity = None
            if self._is_source:
                if len(self.aval_mem_postions) > 0 or len(self.re_entangle_pos) > 0:
                    self.send_signal(MessageType.GEN_ENTANGLE_READY, None)
                yield self.await_signal(self.entanglement_handler, MessageType.ENTANGLED)
            else:
                yield self.await_signal(self, Signals.SUCCESS)
            # # wait for qubit from entangle node
            # expr = yield self.await_port_input(self._qmem_input_port) | re_entangle
            #
            # for event in expr.triggered_events:
            #     if event.source == self._qmem_input_port:
            #         qubit = self._qmem_input_port.rx_input().items[0]
            #         yield from self.handle_entangle(initial_fidelity)
            #     else:
            #         self.handle_re_entangle(event)

    def handle_entangle(self, init_fidelity):
        # TODO this is a dirty way to get rid of previous round of simulation qubit input
        # if sim_time() - self.start_time < 5001:
        #     self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
        #                      f"Detected Previous Round Data", color="red")
        #     return
        # if the qubit is received from the entangle node
        if not self._is_source:
            init_fidelity = None

        if len(self.aval_mem_postions) > 0:
            mem_pos = self.aval_mem_postions.pop(0)
            if self.qmemory.busy:
                yield self.await_program(self.qmemory)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                             f"Received qubit from entangle node {self.entangle_node}\n"
                             f"\tType: ENTANGLE\n"
                             f"\tTime: {ns.sim_time()}\n"
                             f"\tIs Source: {self._is_source}\n"
                             f"\tQubit State: {self.qmemory.peek(0)[0]}\n"
                             f"\tAvailable Memory positions: {self.aval_mem_postions}\n"
                             f"\tUsed Memory positions: {self.used_mem_positions}\n"
                             f"\tSwapping qubit from position {self._input_mem_pos} to {mem_pos}", color="red")

            if self.qmemory.busy:
                yield self.await_program(self.qmemory)
            self.qmemory.execute_instruction(
                INSTR_SWAP, [self._input_mem_pos, mem_pos])

            self.entangled_pairs += 1
            self.used_mem_positions.append(mem_pos)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                             f"\tSwap instruction executed\n"
                             f"\tEntangled position {mem_pos}\n"
                             f"\tIs Source: {self._is_source}\n"
                             f"\tCurrent entangled pairs: {self.entangled_pairs}\n"
                             f"\tUsed memory positions: {self.used_mem_positions}\n"
                             f"\tAvailable memory positions: {self.aval_mem_postions}", color="red")
            self.send_signal(Signals.SUCCESS,
                             SignalMessages.NewEntanglementSignalMessage(
                                 self.node.name,
                                 self.entangle_node,
                                 mem_pos,
                                 self._qmemory_name,
                                 self._is_source,
                                 init_fidelity))
        elif len(self.re_entangle_pos) > 0:
            # case of we dont have free memory positions but we have re-entangle positions
            # we need to re-entangle the qubits
            mem_pos = self.re_entangle_pos.pop(0)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                             f"Received qubit from entangle node {self.entangle_node}\n"
                             f"\tType: RE-ENTANGLE\n"
                             f"\tTime: {ns.sim_time()}\n"
                             f"\tSwapping qubit from position {self._input_mem_pos} to {mem_pos}"
                             , color="red")
            if self.qmemory.busy:
                yield self.await_program(self.qmemory)
            self.qmemory.execute_instruction(
                INSTR_SWAP, [self._input_mem_pos, mem_pos])

            self.entangled_pairs += 1
            if self.qmemory.busy:
                yield self.await_program(self.qmemory)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                             f"\tEntangled position {mem_pos}\n"
                             f"\tIs Source: {self._is_source}\n"
                             f"\tRe-entangle positions: {self.re_entangle_pos}\n"
                             f"\tQState: {self.qmemory.peek(mem_pos)[0]}", color="red")
            self.send_signal(Signals.SUCCESS,
                             SignalMessages.NewEntanglementSignalMessage(
                                 self.node.name,
                                 self.entangle_node, mem_pos,
                                 self._qmemory_name,
                                 self._is_source,
                                 init_fidelity,
                             ))
        else:
            return

    def handle_qubit_generation(self):
        """
        We prioritize the re-entangle signal over the free memory position.
        After no available memory positions, we process the re-entangle positions.
        if both re-entangle and available memory positions are empty, we wait upper layer for
        re-entanglement.
        """
        if self._is_source and (len(self.aval_mem_postions) > 0 or len(self.re_entangle_pos) > 0):
            qsource = self.node.subcomponents[self._qsource_name]
            # avoid generating qubits if the qsource is busy
            current_time = ns.sim_time()
            if qsource._busy_until > current_time:
                wait_time = max(qsource._busy_until - current_time, 1)
                yield self.await_timer(wait_time)
            qsource.trigger()
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} generating qubit\n"
                             f"\tCurrent entangled pairs {self.entangled_pairs}\n"
                             f"\tAvailable memory positions: {self.aval_mem_postions}\n"
                             f"\tUsed memory positions: {self.used_mem_positions}\n"
                             f"\tRe-entangle positions: {self.re_entangle_pos}\n"
                             f"\tIs source: {self._is_source}", color="red")
            # wait for qsource to generate qubit and make sure both we give enough
            yield self.await_port_output(qsource.ports['qout0'])
            qubit_1, qubit_2 = qsource.ports['qout0'].rx_output().items
            # perform fidelity measurement
            initial_fidelity = qapi.fidelity([qubit_1, qubit_2], ks.b00)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} Generated Qubits\n"
                             f"\tInitial fidelity: {initial_fidelity}\n"
                             f"\tQubit1: {qubit_1}\n"
                             f"\tQubit2: {qubit_2}\n"
                             f"\tQState: {qubit_1.qstate}", color="red")
            # send qubit to our own qmemory
            self._qport.tx_input(qubit_1)
            # send the qubit to the right neighbour
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                             f"\tSending qubit to {self.entangle_node}\n"
                             f"\tQState {qubit_2}", color="red")
            # IMPORTANT! WE NEED TO WAIT HERE, AVOID ERROR DUE TO SENDING TOO FAST
            # NETSQUID WILL THROW ERROR AS IT CANNOT HANDLE FORWARD MESSAGE TOO FAST
            # yield self.await_timer(1)
            self.node.ports[f"qout_{self.entangle_node}"].tx_output(qubit_2)



    def handle_re_entangle(self, event):
        source_protocol = event.source
        try:
            ready_signal = source_protocol.get_signal_by_event(
                event=event, receiver=self)
            result: SignalMessages.ReEntangleSignalMessage = ready_signal.result
            """
            result = {
                "entangle_node": entangle_node
                "mem_pos": mem_pos,
            }
            """

            if ready_signal.label == f"{self.name}_re_entangle":
                self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} received signal: {result.__dict__}",
                                 color="blue")
                if f"{result.entangle_node}_qmemory" == self._qmemory_name:
                    # we free non-source node first to make sure we clear the memory position
                    if not self._is_source:
                        self.re_entangle_position(result.re_entangle_mem_poses)
                        self.logger.info(
                            f"GenEntangle {self.name} -> Node {self.node.name} sending re-entangle ready to"
                            f" {result.entangle_node}\n"
                            f"\tRe-entangle memory positions: {result.re_entangle_mem_poses}",
                            color="green")
                        self.send_signal(MessageType.RE_ENTANGLE_READY,
                                         result)
                    else:
                        # case we are the source node, we need to wait 
                        self.logger.info(
                            f"GenEntangle {self.name} -> Node {self.node.name} waiting re-entangle ready signal "
                            f"from {self.entangle_node}",
                            color="blue")
            elif ready_signal.label == f"{self.name}_re_entangle_ready" and self._is_source:
                self.logger.info(
                    f"GenEntangle {self.name} -> Node {self.node.name} received re-entangle ready signal from "
                    f"{result.entangle_node}, mem_pos: {result.re_entangle_mem_poses}, type:{result.re_entangle_type}",
                    color="cyan")
                # TODO: we need to check if we need to send the re-entangle signal or not
                #       to trigger qubit generation. We cannot send the signal if we have available memory positions
                #       or we already have re-entangle positions which the main loop will take care of it
                # after entangle means re-entangle is being processed during a EPR matched, otherwise normal process
                # this is to deal the edge case of stall after last re-entangle is processed
                if result.re_entangle_type == "after_entangle":
                    send_signal = False
                else:
                    send_signal = len(self.re_entangle_pos) == 0 and len(self.aval_mem_postions) == 0

                # we need to add the memory position to the re-entangle position
                self.re_entangle_position(result.re_entangle_mem_poses)
                if send_signal:
                    self.logger.info(
                        f"GenEntangle {self.name} -> Node {self.node.name} received re-entangle ready "
                        f"sending RE_ENTANGLE_READY_SOURCE signal\n"
                        f"\tEntangle Node {result.entangle_node}\n"
                        f"\tmem_pos: {result.re_entangle_mem_poses}\n"
                        f"\tUsed Mem:{self.used_mem_positions}\n"
                        f"\tRe-entangle positions: {result.re_entangle_mem_poses}",
                        color="cyan")
                    self.send_signal(MessageType.RE_ENTANGLE_READY_SOURCE, result)
            else:
                # case of qubit lost due to timeout signal
                self.logger.info(
                    f"GenEntangle {self.name} -> Node {self.node.name} received qubit loss signal\n"
                    f"\tfrom: {result.entangle_node}\n"
                    f"\tmem_pos: {result.re_entangle_mem_poses}",
                    color="cyan")
                if result.re_entangle_type != "timeout":
                    self.logger.error(
                        f"GenEntangle {self.name} -> Node {self.node.name} Signal timeout not match with message type.",
                        color="red")
                    return
                self.re_entangle_timeout(result.re_entangle_mem_poses)

        except KeyError as e:
            self.logger.error(f"GenEntangle {self.name} -> Node {self.node.name} Signal not found in source protocol.",
                              color="red")

    def re_entangle_position(self, re_entangle_mem_poses):
        for mem_pos in re_entangle_mem_poses:
            self.re_entangle_pos.append(mem_pos)
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} re-entangling signal received\n"
                         f"\tAdding Memory position: {re_entangle_mem_poses}\n"
                         f"\tIs Source: {self._is_source}\n"
                         f"\tCurrent Re-entangle position: {self.re_entangle_pos}", color="red")
    def re_entangle_timeout(self, re_entangle_mem_poses):
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} re-entangling timeout signal received\n"
                         f"\tRe-active mem poses: {re_entangle_mem_poses}\n"
                         f"\tIs Source: {self._is_source}\n"
                         f"\tCurrent Available Poses: {self.aval_mem_postions}\n"
                         f"\tUsed Poses: {self.used_mem_positions}\n"
                         f"\tCurrent Re-entangle position: {self.re_entangle_pos}", color="yellow")
        for mem_pos in re_entangle_mem_poses:
            self.aval_mem_postions.insert(0, mem_pos)
            self.used_mem_positions.remove(mem_pos)
            # have case of retangle lost
            # if len(self.aval_mem_postions) > 0:
            #     self.aval_mem_postions.insert(0, mem_pos)
            #     self.used_mem_positions.remove(mem_pos)
            # else:
            #     self.re_entangle_pos.insert(0, mem_pos)
            self.entangled_pairs -= 1
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} re-entangling timeout signal received\n"
                         f"\tRe-active mem poses: {re_entangle_mem_poses}\n"
                         f"\tIs Source: {self._is_source}\n"
                         f"\tCurrent Available Poses: {self.aval_mem_postions}\n"
                         f"\tUsed Poses: {self.used_mem_positions}\n"
                         f"\tCurrent Re-entangle position: {self.re_entangle_pos}", color="red")
        self.send_signal(MessageType.GEN_ENTANGLE_READY, None)
    def start(self):
        """
        Start the protocol.
        :return:
        """
        self.entangled_pairs = None  # counter for entangled pairs
        # Calculate extra memory positions needed:
        extra_memory = self._total_pairs

        # Claim extra memory positions to use (if any):
        def claim_memory_positions(extra_memories, mem_positions, qmemory):
            if extra_memories > 0:
                unused_positions = qmemory.unused_positions
                if extra_memories > len(unused_positions):
                    raise RuntimeError("{} Not enough unused memory positions available: need {}, have {}"
                                       .format(self.name, extra_memories, len(unused_positions)))
                # we need start with 1 since 0 is the default value for input_mem_pos
                for i in unused_positions[0:extra_memories]:
                    mem_positions.append(i)
                    qmemory.mem_positions[i].in_use = True
                # sort the memory positions to make sure they are in order
                mem_positions.sort()

        if self._input_mem_pos is not None:
            self.entangled_pairs = 0
            # since we are using the input memory position as temporary memory position.
            # we will swap the qubits from input memory position to the available memory positions
            # therefore we need to make sure we do not use the input memory position as available memory position
            self.aval_mem_postions = []
            # check if the input memory position is in use
            if not self.qmemory.mem_positions[self._input_mem_pos].in_use:
                # we need to claim the input memory position
                self.qmemory.mem_positions[self._input_mem_pos].in_use = True
            self.used_mem_positions = [self._input_mem_pos]
            extra_memory -= 1
            claim_memory_positions(extra_memory, self.aval_mem_postions, self.qmemory)

        return super().start()

    def reset_memory_positions(self):
        # unclaim used memory positions again in case of stop was not called
        self.qmemory.reset()
        for i in self.qmemory.used_positions:
            self.qmemory.mem_positions[i].in_use = False
        self.used_mem_positions = []
        self.aval_mem_postions = None
        self.re_entangle_pos = []

    def reset(self):
        # stop the signal watcher and qubit watcher and qubit generator
        self.signal_watcher.stop()
        self.qubit_watcher.stop()
        self.qubit_generator.stop()
        # then clear the memory positions
        self.reset_memory_positions()
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} resetting. Memory positions released.\n"
                         f"\tAvailable memory positions: {self.qmemory.unused_positions}\n"
                         f"\tUsed memory positions: {self.qmemory.used_positions}", color="red")

        # Call parent stop method
        super().reset()

    def stop(self):
        # stop the signal watcher and qubit watcher and qubit generator
        self.signal_watcher.stop()
        self.qubit_watcher.stop()
        self.qubit_generator.stop()
        # then clear the memory positions
        self.reset_memory_positions()
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} stopped. Memory positions released.\n"
                         f"\tAvailable memory positions: {self.qmemory.unused_positions}\n"
                         f"\tUsed memory positions: {self.qmemory.used_positions}", color="red")
        # Call parent stop method
        super().stop()

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        # check if entangle node is present but memory positions are not assigned
        if self.entangle_node is not None and self.aval_mem_postions is None and len(
                self._qmemory.unused_positions) < self._total_pairs + 1:
            return False
        # check if entangle node is present and memory positions are assigned correctly
        # -1 here since we are using the input memory position as temporary memory position
        elif (self.aval_mem_postions is not None and
              len(self.aval_mem_postions) != self._total_pairs - 1):
            return False

        # check if the node is a source and has a QSource
        if self._is_source:
            for name, subcomp in self.node.subcomponents.items():
                if isinstance(subcomp, QSource):
                    self._qsource_name = name
                    break
            else:
                return False
        return True


class QubitGenerationProtocol(NodeProtocol):
    def __init__(self, node, name, main_protocol: GenEntanglement, logger=None):
        super().__init__(node=node, name=name)
        self.main_protocol = main_protocol

        # logger setup
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

    def run(self):
        while self.is_running:
            # yield self.await_signal(self.main_protocol, MessageType.GEN_ENTANGLE_READY)
            # self.logger.info(f"QubitGenerationProtocol {self.name} -> Node {self.node.name} received signal\n"
            #                  f"\tSignal: {MessageType.GEN_ENTANGLE_READY}", color="red")
            # yield from self.main_protocol.handle_qubit_generation()

            # wait signals to generate qubits
            expr = (self.await_signal(self.main_protocol, MessageType.GEN_ENTANGLE_READY) |
                    self.await_signal(self.main_protocol, MessageType.RE_ENTANGLE_READY_SOURCE))
            yield expr
            if sim_time() - self.main_protocol.start_time < 5001:
                continue
            if expr.first_term:
                self.logger.info(f"QubitGenerationProtocol {self.name} -> Node {self.node.name} received signal\n"
                                 f"\tSignal: {expr.triggered_events[0].type}", color="red")
                yield from self.main_protocol.handle_qubit_generation()
            elif expr.second_term:
                self.logger.info(f"QubitGenerationProtocol {self.name} -> Node {self.node.name} received signal\n"
                                 f"\tSignal: {expr.triggered_events[0].type}", color="red")
                # check if we should trigger the re-entangle signal or not
                if len(self.main_protocol.aval_mem_postions) == 0:
                    self.logger.info(f"QubitGenerationProtocol {self.name} -> Node {self.node.name} "
                                     f"No available memory positions\n"
                                     f"\tRe-entangle positions: {self.main_protocol.re_entangle_pos}", color="red")
                    yield from self.main_protocol.handle_qubit_generation()
                    # self.main_protocol.aval_mem_postions = self.main_protocol.re_entangle_pos
                    # self.main_protocol.re_entangle_pos = []

    def stop(self):
        super().stop()


class QubitSignalWatcher(NodeProtocol):
    def __init__(self, node, name, qmemory_name, qport, main_protocol: GenEntanglement, logger=None):
        super().__init__(node=node, name=name)
        self.qmemory_name = qmemory_name
        self.qmemory = node.subcomponents[self.qmemory_name]
        self.qport = qport
        self.gen_protocol = main_protocol

        # logger setup
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

    def run(self):
        # TODO: HOW CAN WE AVOID THIS WAIT?
        #       THIS IS A TEMPORARY FIX TO AVOID THE MISS ALIGNMENT OF QUBITS AS THE REMOTE NODE WILL CATCH
        #       THE QUBIT FROM PREVIOUS ROUND OF EXPERIMENT
        # yield self.await_timer(1)
        while self.is_running:
            # wait for qubit from qport
            yield self.await_port_input(self.qport)
            if sim_time() - self.gen_protocol.start_time < 5001:
                self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                                 f"Detected Previous Round Data", color="red")
                continue
            yield from self.gen_protocol.handle_entangle(1)

    def stop(self):
        # wait any pending qubits
        super().stop()


class ReEntangleSignalWatcher(NodeProtocol):
    def __init__(self, node, main_protocol: GenEntanglement, watch_protocol, watch_signal, name, logger=None):
        super().__init__(node=node, name=name)
        self.main_protocol = main_protocol
        self.watch_protocol = watch_protocol
        self.watch_signal = watch_signal
        # logger setup
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

    def run(self):
        while self.is_running:
            expr = (self.await_signal(self.watch_protocol, signal_label=self.watch_signal) |
                    self.await_signal(self.watch_protocol, signal_label=MessageType.RE_ENTANGLE_QUBIT_LOST))

            yield expr
            if sim_time() - self.main_protocol.start_time < 5001:
                self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                                 f"Detected Previous Round Data", color="red")
                continue
            self.logger.info(f"ReEntangleSignalWatcher {self.name} -> Node {self.node.name} received signal\n"
                             f"\tSignal: {self.watch_signal}", color="red")
            self.main_protocol.handle_re_entangle(expr.triggered_events[0])

    def stop(self):
        super().stop()
