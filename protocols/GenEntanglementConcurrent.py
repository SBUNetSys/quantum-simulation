from turtledemo.forest import start

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


class GenEntanglementConcurrent(NodeProtocol):
    """
    Protocol for generating entanglement between two nodes concurrently.
    We only care about the at node level and with respect to the qmemory.
    """

    def __init__(self, node, name=None,
                 entangle_handler=None,
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
        self.aval_mem_positions = [i for i in range(total_pairs)]  # stack of available memory positions
        self.used_mem_positions = []  # stack of used memory positions
        self._total_pairs = total_pairs
        self.entangle_node = entangle_node
        self._is_source = is_source
        # keep track of when we started the protocol. we can use this avoid process old data
        self.start_time = None
        # get the qmemory
        try:
            self._qmemory_name = f"{entangle_node}_qmemory"
            self.qmemory = self.node.subcomponents[self._qmemory_name]
        except KeyError:
            raise ValueError("Qmemory {} not found in node {}.".format(self._qmemory_name, node))

        # input signal from source node
        if not self._is_source:
            self.input_port = self.node.ports[f"qin_{self.entangle_node}"]
        else:
            self.input_port = None

        # logger setup
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        # add signal for re-entangle ready
        self.add_signal(MessageType.ENTANGLED_QUBIT_LOST)
        self.add_signal(MessageType.GEN_ENTANGLE_SUCCESS)

    def run(self):
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                         f"\tentangle node: {self.entangle_node}\n"
                         f"\tmemory available positions: {self.aval_mem_positions}\n"
                         f"\tmemory used positions: {self.used_mem_positions}"
                         f"\ttotal pairs: {self._total_pairs}"
                         f"\tStarting Time: {ns.sim_time()}"
                         )

        if self.entanglement_handler is None:
            raise ValueError("Re-entangle sender must be specified.")

        yield self.await_timer(1)
        self.start_time = ns.sim_time()
        # start the main logic
        while True:
            """
            The concurrent generation logic is
            1. continues generation until no available memory to use
            2. wait two signal:
                if source:
                    SUCCESS and RE-ENTANGLE. SUCCESS = gen-ok, RE-ENTANGLE = re-entangle or qubit lost
                not source:
                    input_port signale, we do not care re-entangle, 
                    we assume whatever source said and overwrite anything else 
            """
            if self._is_source:
                if len(self.aval_mem_positions) > 0:
                    yield from self.handle_qubit_generation()
                expr = yield (self.await_signal(self, MessageType.GEN_ENTANGLE_SUCCESS) |
                              self.await_signal(self.entanglement_handler, MessageType.RE_ENTANGLE_CONCURRENT))
                if expr.first_term:
                    for event in expr.second_term.triggered_events:
                        source_protocol = event.source
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                        if ready_signal.result.data.timestamp < self.start_time:
                            continue
                        yield from self.handle_qubit_generation()
                elif expr.second_term:
                    for event in expr.second_term.triggered_events:
                        source_protocol = event.source
                        try:
                            ready_signal = source_protocol.get_signal_by_event(
                                event=event, receiver=self)
                            result: SignalMessages.ReEntangleSignalMessage = ready_signal.result
                            if result.timestamp < self.start_time:
                                continue
                            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                                             f"Re-entangle signal\n"
                                             f"\tRe-entangle pos {result.re_entangle_mem_poses}"
                                             f"\tmemory available positions: {self.aval_mem_positions}\n"
                                             f"\tmemory used positions: {self.used_mem_positions}", color="blue")
                            for pos in result.re_entangle_mem_poses:
                                self.used_mem_positions.remove(pos)
                                self.aval_mem_positions.append(pos)
                            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                                             f"Re-entangle position updated\n"
                                             f"\tmemory available positions: {self.aval_mem_positions}\n"
                                             f"\tmemory used positions: {self.used_mem_positions}", color="cyan")
                        except Exception as e:
                            self.logger.error(f"Error: {e}")
                            continue

            else:
                yield self.await_port_input(self.input_port)
                message = self.input_port.rx_input()
                if message is None:
                    self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                                     f"Error: No Message where captured\n"
                                     f"\tentangle node: {self.entangle_node}\n"
                                     f"\tmemory available positions: {self.aval_mem_positions}\n"
                                     f"\tmemory used positions: {self.used_mem_positions}", color="red")
                    continue
                self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                                 f"Handle Entanglement\n"
                                 f"\tType: ENTANGLE\n"
                                 f"\tTime: {ns.sim_time()}\n"
                                 f"\tIs Source: {self._is_source}\n"
                                 f"\tQubit State: {self.qmemory.peek(0)[0]}\n"
                                 f"\tAvailable Memory positions: {self.aval_mem_positions}\n"
                                 f"\tUsed Memory positions: {self.used_mem_positions}\n", color="red")
                lost_mem_poses = []
                success_mem_poses = []
                for item in message.items:
                    pos_info, qubit = item
                    if pos_info[1] < self.start_time:
                        lost_mem_poses = []
                        success_mem_poses = []
                        break
                    pos = pos_info[0]
                    # case we lost the qubit we ignore the rest
                    if len(qubit) != 1:
                        lost_mem_poses.append(pos)
                        continue
                    if pos not in self.used_mem_positions:
                        self.aval_mem_positions.remove(pos)
                        self.used_mem_positions.append(pos)

                    if self.qmemory.busy:
                        yield self.await_program(self.qmemory)
                    # put the qubit in memory
                    try:
                        self.qmemory.put(qubit[0], pos)
                    except Exception as e:
                        self.logger.error(f"Error: {e}")
                    self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                                     f"\tPut executed\n"
                                     f"\tEntangled position {pos}\n"
                                     f"\tIs Source: {self._is_source}\n"
                                     f"\tUsed memory positions: {self.used_mem_positions}\n"
                                     f"\tAvailable memory positions: {self.aval_mem_positions}", color="red")
                    success_mem_poses.append(pos)
                if len(lost_mem_poses) > 0:
                    # send signal message indicating failed
                    self.send_signal(MessageType.ENTANGLED_QUBIT_LOST,
                                     SignalMessages.NewEntanglementSignalMessage(
                                         self.node.name,
                                         self.entangle_node,
                                         lost_mem_poses,
                                         self._qmemory_name,
                                         self._is_source,
                                         None))
                if len(success_mem_poses) > 0:
                    self.send_signal(MessageType.GEN_ENTANGLE_SUCCESS,
                                     SignalMessages.NewEntanglementSignalMessage(
                                         self.node.name,
                                         self.entangle_node,
                                         success_mem_poses,
                                         self._qmemory_name,
                                         self._is_source,
                                         None))



    def handle_qubit_generation(self):
        """
        We prioritize the re-entangle signal over the free memory position.
        After no available memory positions, we process the re-entangle positions.
        if both re-entangle and available memory positions are empty, we wait upper layer for
        re-entanglement.
        """
        if self._is_source and len(self.aval_mem_positions) > 0:
            qsource = self.node.subcomponents[self._qsource_name]
            # avoid generating qubits if the qsource is busy
            current_time = ns.sim_time()
            if qsource._busy_until > current_time:
                wait_time = max(qsource._busy_until - current_time, 1)
                yield self.await_timer(wait_time)
            qsource.trigger()
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} generating qubit\n"
                             f"\tAvailable memory positions: {self.aval_mem_positions}\n"
                             f"\tUsed memory positions: {self.used_mem_positions}\n"
                             f"\tIs source: {self._is_source}", color="red")
            # wait for qsource to generate qubit
            yield self.await_port_output(qsource.ports['qout0'])
            qubit_1, qubit_2 = qsource.ports['qout0'].rx_output().items
            # perform fidelity measurement
            initial_fid = qapi.fidelity([qubit_1, qubit_2], ks.b00)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} Generated Qubits\n"
                             f"\tInitial fidelity: {initial_fid}\n"
                             f"\tQubit1: {qubit_1}\n"
                             f"\tQubit2: {qubit_2}\n"
                             f"\tQState: {qubit_1.qstate}", color="red")
            # put qubit to memory
            mem_pos = self.aval_mem_positions.pop(0)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                             f"Handle Entanglement\n"
                             f"\tType: ENTANGLE\n"
                             f"\tTime: {ns.sim_time()}\n"
                             f"\tIs Source: {self._is_source}\n"
                             f"\tQubit State: {self.qmemory.peek(0)[0]}\n"
                             f"\tAvailable Memory positions: {self.aval_mem_positions}\n"
                             f"\tUsed Memory positions: {self.used_mem_positions}\n", color="red")

            if self.qmemory.busy:
                yield self.await_program(self.qmemory)
            # put the qubit in memory
            self.qmemory.put(qubit_1, mem_pos)
            self.used_mem_positions.append(mem_pos)
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                             f"\tPut executed\n"
                             f"\tEntangled position {mem_pos}\n"
                             f"\tIs Source: {self._is_source}\n"
                             f"\tUsed memory positions: {self.used_mem_positions}\n"
                             f"\tAvailable memory positions: {self.aval_mem_positions}", color="red")
            self.send_signal(MessageType.GEN_ENTANGLE_SUCCESS,
                             SignalMessages.NewEntanglementSignalMessage(
                                 self.node.name,
                                 self.entangle_node,
                                 mem_pos,
                                 self._qmemory_name,
                                 self._is_source,
                                 initial_fid))

            # send the qubit to the right neighbour
            self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                             f"\tSending qubit to {self.entangle_node}\n"
                             f"\tQState {qubit_2}", color="red")
            # IMPORTANT! WE NEED TO WAIT HERE, AVOID ERROR DUE TO SENDING TOO FAST
            # NETSQUID WILL THROW ERROR AS IT CANNOT HANDLE FORWARD MESSAGE TOO FAST
            # yield self.await_timer(1)
            # send qubit to the right node with memory pos to keep reference
            self.node.ports[f"qout_{self.entangle_node}"].tx_output(((mem_pos,sim_time()), qubit_2))


    def reset_memory_positions(self):
        # unclaim used memory positions again in case of stop was not called
        self.qmemory.reset()
        self.used_mem_positions = []
        self.aval_mem_positions = [i for i in range(self._total_pairs)]

    def reset(self):
        # then clear the memory positions
        self.reset_memory_positions()
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} resetting. Memory positions released.\n"
                         f"\tAvailable memory positions: {self.qmemory.unused_positions}\n"
                         f"\tUsed memory positions: {self.qmemory.used_positions}", color="red")

        # Call parent stop method
        super().reset()

    def stop(self):
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
        if self.entangle_node is not None and self.aval_mem_positions is None and len(
                self._qmemory.unused_positions) != self._total_pairs :
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


