"""
Message protocol which handles all classical messages between nodes.
"""
import operator
from enum import Enum, auto
from functools import reduce

from netsquid.protocols.nodeprotocols import NodeProtocol, LocalProtocol
from netsquid.components.component import Message, Port


class MessageType(Enum):

    # entanglement signals
    GEN_ENTANGLE_READY = auto()
    ENTANGLED = auto()
    # re-entanglement signals
    RE_ENTANGLE = auto()
    RE_ENTANGLE_READY = auto()
    RE_ENTANGLE_READY_REMOTE = auto()
    RE_ENTANGLE_READY_SOURCE = auto()
    RE_ENTANGLE_FROM_UPPER_LAYER = auto()
    # purification signals
    PURIFICATION_START = auto()
    PURIFICATION_RESULT = auto()
    PURIFICATION_TARGET_MET = auto()
    # verification signals
    VERIFICATION_REQUEST = auto()
    VERIFICATION_READY = auto()
    VERIFICATION_START = auto()
    VERIFICATION_RESULT = auto()
    # swap signals
    SWAP_NEED = auto()
    SWAP_READY = auto()
    SWAP_RESULT = auto()
    SWAP_FAILED = auto()
    CORRECTION_SUCCESS = auto()

    # general signals
    PROTOCOL_FINISHED = auto()


class MessageHandler(NodeProtocol):
    """
    A protocol that handles all classical messages between nodes.
    """

    def __init__(self, node, name, cc_ports):
        super().__init__(node=node, name=name)
        self.node = node
        self.cc_ports = cc_ports
        self.add_signals()

    def send_message(self, message_type, dest, data):
        """
        Send a message to the destination node.
        :param message_type: message signal type
        :param dest: destination node name
        :param data: message data
        :return:
        """
        cport = self.cc_ports[dest]
        cport.tx_output(Message(data, header=message_type))

    def add_signals(self):
        # add all signals from enum
        for signal in MessageType:
            self.add_signal(signal)

    # def send_signals(self, signal, msg):
    #     """
    #     Emit a signal from MessageHandler.
    #     :param signal: signal to emit
    #     :param msg: message data
    #     """
    #     self.node.send_signal(signal, msg)

    def run(self):
        while True:
            # yield until a message is received
            expr = yield reduce(operator.or_, [self.await_port_input(port) for port in self.cc_ports.values()])
            for event in expr.triggered_events:
                port = event.source
                message = port.rx_input()
                for msg in message.items:
                    self.send_signal(message.meta['header'], msg)
            # if message.header == MessageType.ENTANGLED:
            #     for msg in message.items:
            #         self.send_signal(Signals.SUCCESS, msg)
            # elif message.header == MessageType.SWAP_NEED:
            #     for msg in message.items:
            #         self.send_signal(Signals.SUCCESS, msg)
            # elif message.header == MessageType.SWAP_READY:
            #     for msg in message.items:
            #         self.send_signal(Signals.SUCCESS, msg)
            # elif message.header == MessageType.SWAP_RESULT:
            #     self.node.qmemory.put(message.data)
            # elif message.header == MessageType.SWAP_FAILED:
            #     self.node.qmemory.put(message.data)
            # elif message.header == MessageType.CORRECTION_SUCCESS:
            #     self.node.qmemory.put(message.data)
            # elif message.header == MessageType.RE_ENTANGLE:
            #     self.node.qmemory.put(message.data)
            # else:
            #     raise ValueError(f"Unknown message type: {message.header}")
