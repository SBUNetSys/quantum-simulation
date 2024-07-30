"""
Message protocol which handles all classical messages between nodes.
"""
from enum import Enum, auto

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
    APPLY_CORRECTION = auto()
    RE_ENTANGLE = auto()


class MessageHandler(NodeProtocol):
    """
    A protocol that handles all classical messages between nodes.
    """

    def __init__(self, node, swapping_tree):
        super().__init__()
        self.node = node
        self.swapping_tree = swapping_tree
        self.swap_index = 0
        # qubits that are entangled node_name -> memory position
        self.entangled_qubits = {}
    def send_message(self, receiver, message_type, data=None):
        pass
    def run(self):
        yield self.await_port_input(self.node.ports['qin1'])
        yield self.await_port_input(self.node.ports['qin2'])
        yield self.await_program(self.node.qmemory.execute_program(INSTR_SWAP(self.q1, self.q2)))
        yield self.await_program(self.node.qmemory.execute_program(INSTR_SWAP(self.q2, self.q1)))
        self.node.ports['qout1'].tx_output(self.q1)
