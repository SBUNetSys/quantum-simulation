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
class DirectionalEntanglementProtolMultiMem(NodeProtocol):
    """
    Protocol for creating entanglement between two nodes.
    We need to take care of neighbouring nodes and their connections.

    For qubit generation, the node always generates qubits to its right neighbour.



    """

    def __init__(self, node, start_expression=None,
                 left_input_mem_pos=0,
                 right_input_mem_pos=0,
                 left_total_pairs=2,
                 right_total_pairs=2,
                 name=None,
                 left_node=None,
                 right_node=None,
                 ):
        """
        Initialize the DirectionalEntanglementProtocol.
        This protocol is used to create entanglement between two nodes.

        :param node:
        :param start_expression:
        :param left_input_mem_pos:
        :param right_input_mem_pos:
        :param left_total_pairs:
        :param right_total_pairs:
        :param name:
        :param left_node:
        :param right_node:
        """
        if left_node is None and right_node is None:
            raise ValueError("At least one of left_node or right_node must be specified.")
        # if left_cport is None and right_cport is None:
        #     raise ValueError("At least one of left_connection or right_connection must be specified.")
        # if left_cport is not None and not isinstance(left_cport, Port):
        #     raise TypeError("left_connection should be a {}, not a {}".format(Port, type(left_cport)))
        # if right_cport is not None and not isinstance(right_cport, Port):
        #     raise TypeError("right_connection should be a {}, not a {}".format(Port, type(right_cport)))
        # self.left_cport = left_cport
        # self.right_cport = right_cport

        name = name if name else ("DirectionalEntangleNode({}, left={}, right={})"
                                  .format(node.name,
                                          self.left,
                                          self.right))

        super().__init__(node=node, name=name)
        # assign the left and right nodes
        self.left = left_node
        self.right = right_node
        # assign mem pos for input and receiver
        self.left_aval_mem_positions = None  # stack of available memory positions for left neighbour
        self.right_aval_mem_positions = None  # stack of available memory positions for right neighbour
        self.left_used_mem_positions = None  # stack of used memory positions for left neighbour
        self.right_used_mem_positions = None  # stack of used memory positions for right neighbour

        # Now process the case with left and right neighbours
        if left_node is not None and right_node is not None:
            if self.node.subcomponents["left_qmemory"] is None:
                raise ValueError("Node {} does not have a left node quantum memory assigned.".format(self.node))
            if self.node.subcomponents["right_qmemory"] is None:
                raise ValueError("Node {} does not have a right node quantum memory assigned.".format(self.node))
            self._left_total_pairs = left_total_pairs
            self._right_total_pairs = right_total_pairs
            self._left_input_mem_pos = left_input_mem_pos
            self._right_input_mem_pos = right_input_mem_pos
            self._left_qmemory = self.node.subcomponents["left_qmemory"]
            self._right_qmemory = self.node.subcomponents["right_qmemory"]
            self._right_qport = self.node.subcomponents["internal_qchannel"].ports["send"]
            self._is_source = True
        elif left_node is not None:
            if self.node.subcomponents["left_qmemory"] is None:
                raise ValueError("Node {} does not have  a left node quantum memory assigned.".format(self.node))
            self._left_total_pairs = left_total_pairs
            self._right_total_pairs = None
            self._left_input_mem_pos = left_input_mem_pos
            self._right_input_mem_pos = None
            self._left_qmemory = self.node.subcomponents["left_qmemory"]
            self._right_qmemory = None
            # No right neighbour, so this node is a not source
            self._is_source = False
        elif right_node is not None:
            if self.node.subcomponents["right_qmemory"] is None:
                raise ValueError("Node {} does not have a right node quantum memory assigned.".format(self.node))
            self._left_total_pairs = None
            self._right_total_pairs = right_total_pairs
            self._left_input_mem_pos = None
            self._right_input_mem_pos = right_input_mem_pos
            self._right_qmemory = self.node.subcomponents["right_qmemory"]
            self.left_qmemory = None
            self._is_source = True
            self._right_qport = self.node.subcomponents["internal_qchannel"].ports["send"]

        if start_expression is not None and not isinstance(start_expression, EventExpression):
            raise TypeError("Start expression should be a {}, not a {}".format(EventExpression, type(start_expression)))
        self.start_expression = start_expression
        # Qmemory input port left is used to take received qubits from left neighbour
        # Qmemory input port right is used to receive qubits from QSource
        self._qmem_input_port_left = None
        self._qmem_input_port_right = None
        if self._left_input_mem_pos is not None:
            self._qmem_input_port_left = self._left_qmemory.ports["qin{}".format(self._left_input_mem_pos)]
            self._left_qmemory.mem_positions[self._left_input_mem_pos].in_use = True
        if self._right_input_mem_pos is not None:
            self._qmem_input_port_right = self._right_qmemory.ports["qin{}".format(self._right_input_mem_pos)]
            # Claim input memory position is in use:
            self._right_qmemory.mem_positions[self._right_input_mem_pos].in_use = True

    def start(self):
        self.left_entangled_pairs = None  # counter for left neighbour
        self.right_entangled_pairs = None  # counter for right neighbour

        # Calculate extra memory positions needed:
        left_extra_memory = self._left_total_pairs
        right_extra_memory = self._right_total_pairs

        # Claim extra memory positions to use (if any):
        def claim_memory_positions(extra_memory, mem_positions, qmemory):
            if extra_memory > 0:
                unused_positions = qmemory.unused_positions
                if extra_memory > len(unused_positions):
                    raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
                                       .format(extra_memory, len(unused_positions)))
                for i in unused_positions[:extra_memory]:
                    mem_positions.append(i)
                    qmemory.mem_positions[i].in_use = True

        if self._left_input_mem_pos is not None:
            self.left_entangled_pairs = 0
            # since we are using the input memory position as temporary memory position.
            # we will swap the qubits from input memory position to the available memory positions
            # therefore we need to make sure we do not use the input memory position as available memory position
            self.left_aval_mem_positions = []
            self.left_used_mem_positions = [self._left_input_mem_pos]
            left_extra_memory -= 1
            claim_memory_positions(left_extra_memory, self.left_aval_mem_positions, self._left_qmemory)
        if self._right_input_mem_pos is not None:
            self.right_entangled_pairs = 0
            self.right_aval_mem_positions = []
            self.right_used_mem_positions = [self._right_input_mem_pos]
            right_extra_memory -= 1
            claim_memory_positions(right_extra_memory, self.right_aval_mem_positions, self._right_qmemory)

        # if extra_memory > 0:
        #     unused_positions = self.node.qmemory.unused_positions
        #     if extra_memory > len(unused_positions):
        #         raise RuntimeError("Not enough unused memory positions available: need {}, have {}"
        #                            .format(self._num_pairs - 1, len(unused_positions)))
        #     if self._receiver_mem_pos is not None:
        #         # qubits will be in range receiver_mem_pos -> end of the list
        #         for i in unused_positions[self._receiver_mem_pos::]:
        #             self.left_mem_positions.append(i)
        #             self.node.qmemory.mem_positions[i].in_use = True
        #     if self._input_mem_pos is not None:
        #         # qubits will be in range 0 -> end
        #         # end can differ based on the receiver_mem_pos
        #         # if we have receiver_mem_pos then end will be receiver_mem_pos
        #         # else it will be extra_memory which is num_pairs - 1
        #         if self._receiver_mem_pos is not None:
        #             end = self._receiver_mem_pos
        #         else:
        #             end = self._num_pairs - 1
        #         for i in unused_positions[:end]:
        #             self.right_mem_positions.append(i)
        #             self.node.qmemory.mem_positions[i].in_use = True
        # for i in unused_positions[:extra_memory]:
        #     self._mem_positions.append(i)
        #     self.node.qmemory.mem_positions[i].in_use = True
        # Call parent start method
        return super().start()

    def stop(self):
        # Unclaim used memory positions:
        if self.left_used_mem_positions:
            # starts from 1 because 0 is the default value for input_mem_pos
            for i in self.left_used_mem_positions[1:]:
                self._left_qmemory.mem_positions[i].in_use = False
            self.left_aval_mem_positions = None
        if self.right_used_mem_positions:
            # starts from 1 because 0 is the default value for receiver_mem_pos
            for i in self.right_used_mem_positions[1:]:
                self._right_qmemory.mem_positions[i].in_use = False
            self.right_aval_mem_positions = None
        # Call parent stop method
        super().stop()

    def run(self):

        print(f"GenEntangle {self.name} -> Node {self.node.name} has\n"
              f"\tleft memory available positions: {self.left_aval_mem_positions}\n"
              f"\tright memory available positions: {self.right_aval_mem_positions}\n"
              f"\tleft memory used positions: {self.left_used_mem_positions}\n"
              f"\tright memory used positions: {self.right_used_mem_positions}")

        re_entangle = self.start_expression

        while True:
            # if self.start_expression is not None:
            #     print(f"GenEntangle {self.name} -> Got start expression: {self.start_expression.value}, starting... {self.node.name}")
            #     yield re_entangle
            # check if all pairs are entangled
            # if self.left_entangled_pairs is not None and self.right_entangled_pairs is not None:
            #     if len(self.left_aval_mem_positions) == 0 and len(self.right_aval_mem_positions) == 0:
            #         # If no start expression specified then limit generation to one round
            #         # print(f"GenEntangle {self.name} -> {self.node.name} has entangled all pairs for both neighbours.")
            #         # wait for purification protocol
            #         yield re_entangle
            #         source_protocol = re_entangle.atomic_source
            #         ready_signal = source_protocol.get_signal_by_event(
            #             event=expr.second_term.triggered_events[0], receiver=self)
            #         # pass
            # elif self.left_entangled_pairs is not None:
            #     if len(self.left_aval_mem_positions) == 0:
            #         # print(f"GenEntangle {self.name} -> {self.node.name} has entangled all pairs for left neighbour.")
            #         # pass
            #         yield re_entangle
            #         pass
            #         # pass
            # elif self.right_entangled_pairs is not None:
            #     if len(self.right_aval_mem_positions) == 0:
            #         # print(f"GenEntangle {self.name} -> {self.node.name} has entangled all pairs for right neighbour.")
            #         # pass
            #         yield re_entangle
            #         pass
            #         # pass
            if self._qmem_input_port_left is not None and self._qmem_input_port_right is not None:
                print(f"GenEntangle {self.name} -> Node {self.node.name} had both left and right neighbours.")
                if self._is_source and len(self.right_aval_mem_positions) > 0:
                    qsource = self.node.subcomponents[self._qsource_name]
                    # if qsource._busy_until > ns.sim_time():
                    #     yield self.await_timer(qsource._busy_until - ns.sim_time())
                    qsource.trigger()
                    print(f"GenEntangle {self.name} -> Node {self.node.name} generating qubit with "
                          f"entangled {self.right_entangled_pairs}")
                    # wait for qsource to generate qubit and make sure both we give enough

                    yield self.await_port_output(qsource.ports['qout0'])
                    qubit_1, qubit_2 = qsource.ports['qout0'].rx_output().items
                    # perform fidelity measurement
                    initial_fidelity = qapi.fidelity([qubit_1, qubit_2], ks.b00)
                    print(f"GenEntangle {self.name} -> Node {self.node.name} initial fidelity: {initial_fidelity}")
                    # send qubit right qmemory
                    self._right_qport.tx_input(qubit_1)
                    # send the qubit to the right neighbour
                    self.node.ports["qout0"].tx_output(qubit_2)
                    # yield self.await_timer(duration=10000.0)
                    # time for the qubit to be sent
                    # yield self.await_timer(duration=10000.0)

                expr = yield (self.await_port_input(self._qmem_input_port_left) |
                              self.await_port_input(self._qmem_input_port_right) |
                              re_entangle)

                if expr.triggered_events:  # Ensure there are triggered events to process
                    for event in expr.triggered_events:
                        if event.source == self._qmem_input_port_left:
                            yield from self.handle_entangle_left()
                        elif event.source == self._qmem_input_port_right:
                            yield from self.handle_entangle_right()
                        elif event.source == re_entangle:
                            yield from self.handle_re_entangle(re_entangle)

            elif self._qmem_input_port_left is not None:
                print(f"GenEntangle {self.name} -> Node {self.node.name} had only left neighbour.")
                expr = yield (self.await_port_input(self._qmem_input_port_left) |
                              re_entangle)
                if expr.triggered_events:  # Ensure there are triggered events to process
                    for event in expr.triggered_events:
                        if event.source == self._qmem_input_port_left:
                            yield from self.handle_entangle_left()
                        else:
                            yield from self.handle_re_entangle(re_entangle)

            elif self._qmem_input_port_right is not None:
                print(f"GenEntangle {self.name} -> Node {self.node.name} had only right neighbour {self.right}.")
                if self._is_source and len(self.right_aval_mem_positions) > 0:
                    qsource = self.node.subcomponents[self._qsource_name]
                    # avoid generating qubits if the qsource is busy
                    # if qsource._busy_until > ns.sim_time():
                    #     yield self.await_timer(qsource._busy_until - ns.sim_time())
                    qsource.trigger()
                    print(f"GenEntangle {self.name} -> Node {self.node.name} generating qubit with "
                          f"entangled {self.right_entangled_pairs}")
                    # wait for qsource to generate qubit and make sure both we give enough

                    yield self.await_port_output(qsource.ports['qout0'])
                    qubit_1, qubit_2 = qsource.ports['qout0'].rx_output().items
                    # perform fidelity measurement
                    initial_fidelity = qapi.fidelity([qubit_1, qubit_2], ks.b00)
                    print(f"GenEntangle {self.name} -> Node {self.node.name} initial fidelity: {initial_fidelity}")
                    # send qubit right qmemory
                    self._right_qport.tx_input(qubit_1)
                    # send the qubit to the right neighbour
                    print(f"GenEntangle {self.name} -> Node {self.node.name} sending qubit to {self.right}")
                    self.node.ports["qout0"].tx_output(qubit_2)
                    # yield self.await_timer(duration=10000.0)
                    # time for the qubit to be sent

                expr = (yield self.await_port_input(self._qmem_input_port_right) | re_entangle)
                if expr.triggered_events:  # Ensure there are triggered events to process
                    for event in expr.triggered_events:
                        if event.source == self._qmem_input_port_right:
                            yield from self.handle_entangle_right()
                        else:
                            yield from self.handle_re_entangle(re_entangle)

    def handle_re_entangle(self, expr):
        yield expr
        if expr.value:
            source_protocol = expr.atomic_source
            ready_signal = source_protocol.get_signal_by_event(
                event=expr.triggered_events[0], receiver=self)
            print(f"GenEntangle {self.name} -> Node {self.node.name} received signal: {ready_signal.result}")

    def handle_entangle_left(self):
        # yield self.await_port_input(self._qmem_input_port_left)
        # if the qubit is received from the left neighbour
        if len(self.left_aval_mem_positions) == 0:
            return
        # check pos
        left_mem_pos = self.left_aval_mem_positions.pop()
        print(f"GenEntangle {self.name} -> Node {self.node.name} Received qubit from left neighbour {self.left},"
              f" Swapping qubit from position {self._left_input_mem_pos} to {left_mem_pos}")

        self._left_qmemory.execute_instruction(
            INSTR_SWAP, [self._left_input_mem_pos, left_mem_pos])
        if self._left_qmemory.busy:
            yield self.await_program(self._left_qmemory)

        self.left_entangled_pairs += 1
        self.left_used_mem_positions.append(left_mem_pos)
        print(f"GenEntangle {self.name} -> Node {self.node.name} entangled left pairs: {self.left_entangled_pairs}\n"
              f"\tUsed memory positions: {self.left_used_mem_positions}\n"
              f"\tAvailable memory positions: {self.left_aval_mem_positions}")
        self.send_signal(Signals.SUCCESS, {"left_mem_pos": left_mem_pos})

    def handle_entangle_right(self):
        # if the qubit is received from the right neighbour
        if len(self.right_aval_mem_positions) == 0:
            return
        right_mem_pos = self.right_aval_mem_positions.pop()
        print(f"GenEntangle {self.name} -> Node {self.node.name} Received qubit from QSource,"
              f"Swapping qubit from position {self._right_input_mem_pos} to {right_mem_pos}")
        self._right_qmemory.execute_instruction(
            INSTR_SWAP, [self._right_input_mem_pos, right_mem_pos])
        if self._right_qmemory.busy:
            yield self.await_program(self._right_qmemory)
        self.right_entangled_pairs += 1
        self.right_used_mem_positions.append(right_mem_pos)
        print(f"GenEntangle {self.name} -> Node {self.node.name} entangled right pairs: {self.right_entangled_pairs}")
        self.send_signal(Signals.SUCCESS, {"right_mem_pos": right_mem_pos})

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        if self.node.qmemory is None:
            return False
        # check if left and right neighbours are present but memory positions are not assigned
        if self.left is not None and self.left_aval_mem_positions is None and len(
                self._left_qmemory.unused_positions) < self._left_total_pairs - 1:
            return False
        if self.right is not None and self.right_aval_mem_positions is None and len(
                self._right_qmemory.unused_positions) < self._right_total_pairs - 1:
            return False
        # check if left and right neighbours are present and memory positions are assigned correctly
        # -1 here since we are using the input memory position as temporary memory position
        elif (self.left_aval_mem_positions is not None and
              len(self.left_aval_mem_positions) != self._left_total_pairs - 1):
            return False
        elif (self.right_aval_mem_positions is not None and
              len(self.right_aval_mem_positions) != self._right_total_pairs - 1):
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
