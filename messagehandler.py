"""
Message protocol which handles all classical messages between nodes.
"""
import operator
from enum import Enum, auto
from functools import reduce

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


class MessageType(Enum):
    ENTANGLED = auto()
    SWAP_NEED = auto()
    SWAP_READY = auto()
    SWAP_RESULT = auto()
    SWAP_FAILED = auto()
    CORRECTION_SUCCESS = auto()
    RE_ENTANGLE = auto()


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
