import netsquid as ns

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
                 re_entangle_sender=None,
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
        @param re_entangle_sender: `protocols.EntanglementHandler`
                    The protocol that will send the re-entangle signal.
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

        if re_entangle_sender is not None and not isinstance(re_entangle_sender, NodeProtocol):
            raise TypeError("Start expression should be a {}, not a {}".format(
                NodeProtocol, type(re_entangle_sender)))
        self.re_entangle_sender = re_entangle_sender
        self.aval_mem_postions = None  # stack of available memory positions
        self.used_mem_positions = None  # stack of used memory positions
        self._total_pairs = total_pairs
        self._input_mem_pos = input_mem_pos
        self.entangle_node = entangle_node

        self._is_source = is_source
        self._qmemory_name = f"{entangle_node}_qmemory"
        # get the qmemory
        try:
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

    def run(self):

        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                         f"\tentangle node: {self.entangle_node}\n"
                         f"\tmemory available positions: {self.aval_mem_postions}\n"
                         f"\tmemory used positions: {self.used_mem_positions}"
                         f"\ttotal pairs: {self._total_pairs}"
                         f"\tinput memory position: {self._input_mem_pos}"
                         )
        if self.re_entangle_sender is None:
            raise ValueError("Re-entangle sender must be specified.")
        # the EntanglementHandler will send the {self.entangle_node}_re_entangle and we will listen for it
        # the EntanglementHandler will also send the {self.entangle_node}_re_entangle_ready and we will listen for it
        re_entangle = (self.await_signal(self.re_entangle_sender, f"{self.name}_re_entangle") |
                       self.await_signal(self.re_entangle_sender, f"{self.name}_re_entangle_ready"))
        while True:
            # the logic that we generate qubits and send them to the entangle node
            # if is source we generate qubits and send them to the entangle node
            # if not source we receive qubits from the entangle node
            if self._qmem_input_port is not None:
                # print(f"GenEntangle {self.name} -> Node {self.node.name} had entangle node {self.entangle_node}.")
                initial_fidelity = None
                if self._is_source and len(self.aval_mem_postions) > 0:
                    qsource = self.node.subcomponents[self._qsource_name]
                    # avoid generating qubits if the qsource is busy
                    if qsource._busy_until > ns.sim_time():
                        yield self.await_timer(qsource._busy_until - ns.sim_time())
                    qsource.trigger()
                    self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} generating qubit\n"
                                     f"\tCurrent entangled pairs {self.entangled_pairs}")
                    # wait for qsource to generate qubit and make sure both we give enough

                    yield self.await_port_output(qsource.ports['qout0'])
                    qubit_1, qubit_2 = qsource.ports['qout0'].rx_output().items
                    # perform fidelity measurement
                    initial_fidelity = qapi.fidelity([qubit_1, qubit_2], ks.b00)
                    new_fidelity = qapi.fidelity([qubit_1, qubit_2], ks.b00)

                    # apply H gate to qubit_1
                    # print_blue(f"GenEntangle {self.name} -> Node {self.node.name} applying H gate to qubit_1")
                    # qapi.operate(qubits=[qubit_1], operator=operators.H)
                    # qapi.operate(qubits=[qubit_1, qubit_2], operator=operators.CNOT)
                    self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                                     f"initial fidelity: {initial_fidelity}")
                    # send qubit right qmemory
                    self._qport.tx_input(qubit_1)
                    # send the qubit to the right neighbour
                    self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                                     f"sending qubit to {self.entangle_node}")
                    self.node.ports[f"qout_{self.entangle_node}"].tx_output(qubit_2)
                    # yield self.await_timer(duration=10000.0)
                    # time for the qubit to be sent
                # wait for qubit from entangle node
                expr = (yield self.await_port_input(self._qmem_input_port) | re_entangle)
                if expr.triggered_events:  # Ensure there are triggered events to process
                    for event in expr.triggered_events:
                        if event.source == self._qmem_input_port:
                            yield from self.handle_entangle(initial_fidelity)
                        else:
                            self.handle_re_entangle(event)

    def handle_entangle(self, init_fidelity):
        # if the qubit is received from the entangle node
        if len(self.aval_mem_postions) == 0:
            return
        mem_pos = self.aval_mem_postions.pop(0)
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} "
                         f"Received qubit from entangle node {self.entangle_node}\n"
                         f"\tSwapping qubit from position {self._input_mem_pos} to {mem_pos}")

        self.qmemory.execute_instruction(
            INSTR_SWAP, [self._input_mem_pos, mem_pos])
        if self.qmemory.busy:
            yield self.await_program(self.qmemory)

        self.entangled_pairs += 1
        self.used_mem_positions.append(mem_pos)
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name}\n"
                         f"\tCurrent entangled pairs: {self.entangled_pairs}\n"
                         f"\tUsed memory positions: {self.used_mem_positions}\n"
                         f"\tAvailable memory positions: {self.aval_mem_postions}")
        self.send_signal(Signals.SUCCESS,
                         SignalMessages.NewEntanglementSignalMessage(self.entangle_node, mem_pos, self._qmemory_name,
                                                                     self._is_source,
                                                                     init_fidelity,
                                                                     ))

    def handle_re_entangle(self, event):
        source_protocol = event.source
        try:
            ready_signal = source_protocol.get_signal_by_event(
                event=event, receiver=self)
            result: SignalMessages.EntangleSignalMessage = ready_signal.result
            """
            result = {
                "entangle_node": entangle_node
                "mem_pos": mem_pos,
            }
            """
            if ready_signal.label == f"{self.entangle_node}_re_entangle":
                self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} received signal: {result}",
                                 color="blue")
                if (result.mem_pos in self.used_mem_positions and
                        f"{result.entangle_node}_qmemory" == self._qmemory_name):
                    # release the memory position from the used memory positions and add it to the available positions
                    if not self._is_source:
                        # case we are not the source node, we can release the memory position and add it to the available positions
                        # we need send the signal to the upper layer to tell source node to generate new qubit
                        # self.used_mem_positions.remove(result["mem_pos"])
                        # self.aval_mem_postions.append(result["mem_pos"])
                        # self.entangled_pairs -= 1
                        self.free_memory_position(result.mem_pos)
                        self.send_signal(MessageType.RE_ENTANGLE_READY,
                                         SignalMessages.EntangleSignalMessage(self.entangle_node, result.mem_pos))
                    else:
                        # case we are the source node, we need to wait 
                        self.logger.info(
                            f"GenEntangle {self.name} -> Node {self.node.name} waiting re-entangle ready signal from {self.entangle_node}",
                            color="blue")
            elif ready_signal.label == f"{self.entangle_node}_re_entangle_ready" and self._is_source:
                self.logger.info(
                    f"GenEntangle {self.name} -> Node {self.node.name} received re-entangle ready signal from {self.entangle_node}",
                    color="blue")
                # we need to generate new qubit by freeing the memory position
                self.free_memory_position(result.mem_pos)

        except KeyError as e:
            self.logger.error(f"GenEntangle {self.name} -> Node {self.node.name} Signal not found in source protocol.",
                              color="red")

    def free_memory_position(self, mem_pos):
        self.used_mem_positions.remove(mem_pos)
        self.aval_mem_postions.append(mem_pos)
        self.entangled_pairs -= 1
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} freeing memory position {mem_pos}\n"
                         f"\tCurrent entangled pairs: {self.entangled_pairs}\n"
                         f"\tUsed memory positions: {self.used_mem_positions}\n"
                         f"\tAvailable memory positions: {self.aval_mem_postions}", color="blue")

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
        for i in self.qmemory.used_positions:
            self.qmemory.mem_positions[i].in_use = False
        self.used_mem_positions = []
        self.aval_mem_postions = None

    def reset(self):
        self.reset_memory_positions()
        self.logger.info(f"GenEntangle {self.name} -> Node {self.node.name} resetting. Memory positions released.\n"
                         f"\tAvailable memory positions: {self.qmemory.unused_positions}\n"
                         f"\tUsed memory positions: {self.qmemory.used_positions}", color="red")
        # Call parent stop method
        super().reset()

    def stop(self):
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
