import operator
from functools import reduce

import numpy as np
import netsquid as ns
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals

from protocols.MessageHandler import MessageType, ClassicalMessage
from utils import Logging, SignalMessages


class EntanglementHandler(NodeProtocol):
    """
    Protocol to manage entanglement for a node.
    The protocol keeps track of the number of entangled pairs and the memory position for the qubit
    """

    def __init__(self, node, name, num_pairs, entangle_nodes,
                 node_distance, memory_depolar_rate,
                 qubit_input_signals, cc_message_handler,
                 logger=None):
        """
        Initialize the protocol.

        Parameters
        ----------
        node : :class:`~netsquid.nodes.node.Node`
            The node that the protocol is attached to.
        num_pairs : int
            The max number of qubits that can be entangled i.e qmemory size.
        entangle_nodes : dict
            A dict mapping of entangle node name to entangle node protocol name
        node_distance : float
            The distance between the nodes in km.
        memory_depolar_rate : float
            The depolarization rate of the qubits in memory. (Hz)
        qubit_input_signals : list
            The signals that the protocol listens to for qubit input.
        cc_message_handler : MessageHandler
            The message handler for classical communication.
        logger : Logging
            The logger to log messages.
        """
        if entangle_nodes is None or len(entangle_nodes) == 0:
            raise ValueError("entangle_node must be specified.")

        super().__init__(node=node, name=name)
        # since we will use on memory position for entanglement operation, therefore we need to subtract 1 for each node
        # in case of multiple entangle_node, we need to multiply by the number of entangle nodes
        self.max_pairs = (num_pairs - 1) * len(entangle_nodes)
        # store the entangle nodes
        self.entangle_nodes = entangle_nodes
        # mapping of entangled qubits to memory positions key: node name, value: {memory position, fidelity}
        self.entangled_qubits = {node: {} for node in entangle_nodes.keys()}
        # mapping of temporary qubits to memory positions
        self.temp_qubits = {node: {} for node in entangle_nodes.keys()}
        # keep track of the number of entangled pairs
        self.entangled_pairs_count = 0
        # entangle_message_queue
        self.entangle_message_queue = []
        # store the entangle nodes
        self.entangled_nodes = entangle_nodes
        # store the depolar rate and node distance
        self.depolar_rate = memory_depolar_rate
        self.node_distance = node_distance
        # qubit input signal, which can be from source or remote node
        await_signals = [self.await_signal(protocol, Signals.SUCCESS) for protocol in qubit_input_signals]
        # re-entangle ready signal from GenEntanglement protocol
        re_entangle_ready_signals = [self.await_signal(protocol, MessageType.RE_ENTANGLE_READY) for protocol in
                                     qubit_input_signals]
        self.re_entangle_ready_signals = reduce(operator.or_, re_entangle_ready_signals)
        # have expression to wait for the qubit input signal
        self.qubit_input_signal = reduce(operator.or_, await_signals)
        # classical message handler
        self.cc_message_handler = cc_message_handler

        # set the logger
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        # add signals for sending
        for entangle_protocol in self.entangle_nodes.values():
            self.add_new_signal(f"{entangle_protocol}_re_entangle_ready")
            self.add_new_signal(f"{entangle_protocol}_re_entangle")

    def add_new_signal(self, signal_label):
        """
        Add a new signal to the protocol. This allows the protocol to emit the signal with the signal label.
        :param signal_label:
        :return:
        """
        self.add_signal(signal_label)

    def process_entangle_message(self, message: ClassicalMessage):
        """
        Process the entangle message if the qubit is ready. Otherwise, store the message back in the queue.
        """
        from_node = message.from_node
        to_node = message.to_node
        # get the entangle data from the message
        entangle_data = message.data
        # get the mem_pos from the entangle data
        mem_pos = entangle_data.mem_pos
        self.logger.info(f"ManageEntangle {self.name} -> Entanglement signal from {from_node} to {to_node},"
                         f" mem_pos: {mem_pos}", color="blue")
        if mem_pos in self.temp_qubits[from_node]:
            # add the qubit to the entangled qubits
            self.entangled_qubits[from_node][mem_pos] = self.temp_qubits[from_node][mem_pos]
            # remove the temporary qubit
            del self.temp_qubits[from_node][mem_pos]
            self.entangled_pairs_count += 1
            self.logger.info(f"ManageEntangle {self.name} -> Entanglement complete\n "
                             f"\tEntangled_pairs_count: {self.entangled_pairs_count}\n"
                             f"\tExpected pairs: {self.max_pairs}"
                             f"\tProgress: {self.entangled_pairs_count / self.max_pairs}", color="green")
        else:
            # store the qubit in the temporary qubits
            self.entangle_message_queue.append(message)

            self.logger.info(f"ManageEntangle {self.name} -> Entangle signal from {from_node}, mem_pos: {mem_pos}"
                             f" not ready yet", color="blue")

    def process_re_entangle_message(self, message: ClassicalMessage):
        """
        Process the re-entangle message from upper layer. 
        The source node will wait re-entangle ready signal from the destination node.
        Then source node will clear up the memeory and generate new qubit to send to the destination node.

        """
        # get the entangle data from the message

        entangle_data = message.data
        entangle_node = entangle_data.entangle_node
        mem_pos = entangle_data.mem_pos
        self.logger.info(f"ManageEntangle {self.name} -> Re-entangle signal, entangle_node: {entangle_node},"
                         f" mem_pos: {mem_pos}", color="yellow")
        self.send_signal(f"{entangle_node}_re_entangle",
                         SignalMessages.EntangleSignalMessage(entangle_node, mem_pos))

    def process_message_queue(self):

        temp = self.entangle_message_queue
        self.entangle_message_queue = []
        for message in temp:
            self.process_entangle_message(message)

    def estimate_fidelity_theoretical(self, initial_fidelity):
        """Estimate fidelity based on noise parameters and channel length."""
        # depolar_rate = noise_params['depolar_rate']
        # dephase_rate = noise_params['dephase_rate']

        # Depolarizing effect
        time_spend = self.node_distance / 200e3
        # p_depolar = 1 - np.exp(-depolar_rate * channel_length)
        # f_depolar = (1 - p_depolar) + (p_depolar / 4)
        p_depolar = 1 - np.exp(-self.depolar_rate * time_spend)
        f_depolar = (1 - p_depolar) + (p_depolar / 4)

        # # Dephasing effect
        # p_dephase = 1 - np.exp(-dephase_rate * channel_length)
        # f_dephase = 1 - p_dephase / 2

        # Combine effects (assuming independent noise processes)
        final_fidelity = initial_fidelity * f_depolar

        return final_fidelity

    def run(self):
        entangle_signals = (self.await_signal(self.cc_message_handler, signal_label=MessageType.ENTANGLED) |
                            self.await_signal(self.cc_message_handler, signal_label=MessageType.RE_ENTANGLE) |
                            self.await_signal(self.cc_message_handler,
                                              signal_label=MessageType.RE_ENTANGLE_READY_REMOTE) |
                            self.re_entangle_ready_signals)
        while True:
            # wait for entanglement
            expr = yield self.qubit_input_signal | entangle_signals
            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    gen_data: SignalMessages.GenEntanglementSignalMessage = result
                    mem_pos = gen_data.mem_pos
                    is_source = gen_data.is_source
                    qmemory_name = gen_data.qmemory_name
                    entangle_node = gen_data.entangle_node
                    initial_fidelity = gen_data.init_fidelity
                    if is_source:
                        # store the qubit in the temporary qubits
                        self.temp_qubits[entangle_node][mem_pos] = self.estimate_fidelity_theoretical(initial_fidelity)
                        self.logger.info(
                            f"ManageEntangle {self.name} -> Entangle signal from QSource, mem_pos: {mem_pos}\n"
                            f"\tInitial Fidelity: {initial_fidelity}\n "
                            f"\tEstimated Fidelity: {self.temp_qubits[entangle_node][mem_pos]}", color="blue")
                    else:
                        # add the qubit to the entangled qubits
                        self.logger.info(
                            f"ManageEntangle {self.name} -> Entangle signal from {entangle_node}, mem_pos: {mem_pos}",
                            color="blue")
                        # we don't need to estimate the fidelity for the remote node
                        # as we don't know the initial fidelity. Here it will be None
                        self.entangled_qubits[entangle_node][mem_pos] = initial_fidelity
                        self.entangled_pairs_count += 1
                        # send the entangled signal to the source node
                        self.cc_message_handler.send_message(MessageType.ENTANGLED, entangle_node,
                                                             ClassicalMessage(
                                                                 self.node.name, entangle_node,
                                                                 SignalMessages.EntangleSignalMessage(self.node.name,
                                                                                                      mem_pos)
                                                             ))
            elif expr.second_term.value:
                # case we have entanglement signal
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result: ClassicalMessage = ready_signal.result
                    if ready_signal.label == MessageType.ENTANGLED:
                        self.process_entangle_message(result)
                    elif ready_signal.label == MessageType.RE_ENTANGLE:
                        # process re-entangle signal
                        self.process_re_entangle_message(result)
                    elif ready_signal.label == MessageType.RE_ENTANGLE_READY:
                        # process re-entangle ready signal
                        # only remote node will receive this signal
                        # we need to send an classical message to the source node
                        entangle_data = result.data
                        mem_pos = entangle_data.mem_pos
                        entangle_node = entangle_data.entangle_node
                        self.cc_message_handler.send_message(MessageType.RE_ENTANGLE_READY_REMOTE, entangle_node,
                                                             ClassicalMessage(
                                                                 self.node.name, entangle_node,
                                                                 SignalMessages.EntangleSignalMessage(self.node.name,
                                                                                                      mem_pos)
                                                             ))
                    elif ready_signal.label == MessageType.RE_ENTANGLE_READY_REMOTE:
                        # process re-entangle ready remote signal
                        # only source node will receive this signal, we can clear up the memory and generate new qubit
                        entangle_data = result.data
                        mem_pos = entangle_data.mem_pos
                        entangle_node = entangle_data.entangle_node
                        message = SignalMessages.EntangleSignalMessage(entangle_node, mem_pos)
                        self.send_signal(f"{entangle_node}_re_entangle_ready", message)

            if self.entangled_pairs_count >= self.max_pairs:
                # send finish signal to the source node
                self.logger.info(f"ManageEntangle {self.name} -> Entanglement complete", color="yellow")
                self.send_signal(Signals.SUCCESS, self.entangled_qubits)
                break

    def reset(self):
        self.entangled_qubits = {node: {} for node in self.entangled_nodes}
        self.temp_qubits = {node: {} for node in self.entangled_nodes}
        self.entangled_pairs_count = 0
        self.entangle_message_queue = []
        super().reset()

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        # check all entangled qmemory is connected
        for node in self.entangled_nodes:
            try:
                memory = self.node.subcomponents[f"{node}_qmemory"]
            except KeyError:
                print(f"Memory {node}_qmemory not found")
                return False
        return True
