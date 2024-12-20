import operator
from functools import reduce
from collections import defaultdict

import numpy as np
from netsquid.qubits import operators
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
from netsquid.components.instructions import  INSTR_Z, INSTR_X

from protocols.MessageHandler import MessageType
from utils import Logging
from utils.ClassicalMessages import ClassicalMessage
from utils.SignalMessages import *

class TransportProtocol(NodeProtocol):
    """
    A protocol that send qubits in hop by hop setting.

    For example
    A -> B -> C

    A will teleport qubit in to B
    B will teleport qubit in to C
    """
    """
        Initialize the protocol
        :param node: the node that the protocol is attached to
        :param name: the name of the protocol
        :param qubit_ready_protocols: the lower layers that will send the qubit ready signal
         (i.e purfication or verification)
        :param entangled_node: the out going edge for this node, i.e A -> B -> C, A entangled B, B entangled C
        :param source: the source node name
        :param destination: the destination node name
        :param cc_message_handler: the classical message handler
        :param qubit_size: the number of qubits needs to be transmitted
        :param logger: the logger
        :param is_top_layer: if we are the top layer of the simulation
    """
    def __init__(self, node,
                 name,
                 qubit_ready_protocols,
                 entangled_node,
                 source,
                 destination,
                 cc_message_handler,
                 qubit_size,
                 logger,
                 is_top_layer=False):
        super().__init__(node=node, name=name)
        # identify role of the node
        self.is_source_node = False
        self.is_destination_node = False
        if self.node.name == source:
            self.is_source_node = True
        if self.node.name == destination:
            self.is_destination_node = True
        # store next hop node name
        self.entangled_node = entangled_node
        # qubit input signal from lower layers, can be purification or verification
        await_signals = [self.await_signal(protocol, Signals.SUCCESS) for protocol in qubit_ready_protocols]
        # have expression to wait for ANY qubit input signal
        self.qubit_input_signal = reduce(operator.or_, await_signals)
        # classical message handler
        self.cc_message_handler = cc_message_handler
        # keep track of the node name and actual memory name
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger
        self.qubit_size = int(qubit_size)
        self.is_top_layer = is_top_layer

        # variable for transport use
        self.entangled_qubits = defaultdict(dict) # key: entangled_node(in_node), value = {mem_pos: fid}
        # qubits that needs to be transport to next hop
        self.transport_qubits = defaultdict(dict) # key: entangled_node(in_node), value = {mem_pos: fid}
    def run(self):
        """
        Run the protocol
        :return:
        """
        transport_signal = (self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_REQUEST) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_READY) |
                        self.await_signal(self.cc_message_handler, signal_label=MessageType.TRANSPORT_APPLY_CORRECTION))
        self.logger.info(f"Transport {self.name} -> Start transport protocol", color="cyan")
        while True:
            # handle other operations first then we try to perform the transport operation
            expr = yield self.qubit_input_signal | transport_signal

            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    # result -> EntangleSignalMessage (Maybe Purification or Verification)
                    result = ready_signal.result
                    mem_pos = result.mem_pos
                    entangle_node = result.entangle_node
                    self.logger.info(f"Transport {self.name} -> Qubit Ready signal from {entangle_node}\n "
                                     f"mem_pos: {mem_pos}",
                                     color="blue")
                    self.entangled_qubits[entangle_node][mem_pos] = result.fidelity


    def check_transport_ready(self):
        """
        check if we can transport qubits
        :return:
        """

