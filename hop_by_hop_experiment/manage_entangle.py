import operator
from functools import reduce

import numpy as np
import netsquid as ns
import pydynaa as pd

from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits.ketutil import outerprod
from netsquid.qubits.ketstates import s0, s1
from netsquid.qubits import operators as ops, ketstates, operators
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

from messagehandler import MessageType


def print_blue(text):
    print(f"\033[94m{text}\033[0m")


def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def print_green(text):
    print(f"\033[92m{text}\033[0m")


class ManageEntanglement(NodeProtocol):
    """
    Protocol to manage entanglement for a node.
    The protocol keeps track of the number of entangled pairs and the memory position for the qubit
    """

    def __init__(self, node, name, num_pairs, entangle_nodes,
                 node_distance, memory_depolar_rate,
                 qubit_input_signals, cc_message_handler):
        """
        Initialize the protocol.

        Parameters
        ----------
        node : :class:`~netsquid.nodes.node.Node`
            The node that the protocol is attached to.
        num_pairs : int
            The max number of qubits that can be entangled i.e qmemory size.
        entangle_nodes : list
            A list of nodes name that we have entanglement with.
        node_distance : float
            The distance between the nodes in km.
        memory_depolar_rate : float
            The depolarization rate of the qubits in memory. (Hz)
        qubit_input_signals : list
            The signals that the protocol listens to for qubit input.
        cc_message_handler : MessageHandler
            The message handler for classical communication.
        """
        if entangle_nodes is None or len(entangle_nodes) == 0:
            raise ValueError("entangle_node must be specified.")

        super().__init__(node=node, name=name)
        # since we will use on memory position for entanglement operation, therefore we need to subtract 1 for each node
        # in case of multiple entangle_node, we need to multiply by the number of entangle nodes
        self.max_pairs = (num_pairs - 1) * len(entangle_nodes)
        # mapping of entangled qubits to memory positions key: node name, value: {memory position, fidelity}
        self.entangled_qubits = {node: {} for node in entangle_nodes}
        # mapping of temporary qubits to memory positions
        self.temp_qubits = {node: {} for node in entangle_nodes}
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
        # have expression to wait for the qubit input signal
        self.qubit_input_signal = reduce(operator.or_, await_signals)
        # classical message handler
        self.cc_message_handler = cc_message_handler

    def add_new_signal(self, signal):
        self.add_signal(signal)

    def process_entangle_message(self, message):
        """
        Process the entangle message if the qubit is ready.
        """
        from_node = message["from"]
        to_node = message["to"]
        mem_pos = message["mem_pos"]
        print_blue(f"ManageEntangle {self.name} -> Entanglement signal from {from_node} to {to_node},"
                   f" mem_pos: {mem_pos}")
        if mem_pos in self.temp_qubits[from_node]:
            # add the qubit to the entangled qubits
            self.entangled_qubits[from_node][mem_pos] = self.temp_qubits[from_node][mem_pos]
            # remove the temporary qubit
            del self.temp_qubits[from_node][mem_pos]
            self.entangled_pairs_count += 1
            print_green(f"ManageEntangle {self.name} -> Entanglement complete\n "
                        f"\tEntangled_pairs_count: {self.entangled_pairs_count}\n"
                        f"\tExpected pairs: {self.max_pairs}"
                        f"\tProgress: {self.entangled_pairs_count / self.max_pairs}")
        else:
            # store the qubit in the temporary qubits
            self.entangle_message_queue.append(message)
            print_blue(f"ManageEntangle {self.name} -> Entangle signal from {from_node}, mem_pos: {mem_pos}"
                       f" not ready yet")

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
        entangle_signal = self.await_signal(self.cc_message_handler, signal_label=MessageType.ENTANGLED)
        while True:
            # wait for entanglement
            expr = yield self.qubit_input_signal | entangle_signal
            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    mem_pos = result["mem_pos"]
                    self.is_source = result["is_source"]
                    qmemory_name = result["qmemory"]
                    entangle_node = result["entangle_node"]
                    initial_fidelity = result["initial_fidelity"]
                    if self.is_source:
                        # store the qubit in the temporary qubits
                        self.temp_qubits[entangle_node][mem_pos] = self.estimate_fidelity_theoretical(initial_fidelity)
                        print_blue(f"ManageEntangle {self.name} -> Entangle signal from QSource, mem_pos: {mem_pos}\n"
                                   f"\tInitial Fidelity: {initial_fidelity}\n "
                                   f"\tEstimated Fidelity: {self.temp_qubits[entangle_node][mem_pos]}")
                    else:
                        # add the qubit to the entangled qubits
                        print_blue(
                            f"ManageEntangle {self.name} -> Entangle signal from {entangle_node}, mem_pos: {mem_pos}")
                        # we don't need to estimate the fidelity for the remote node
                        # as we don't know the initial fidelity. Here it will be None
                        self.entangled_qubits[entangle_node][mem_pos] = initial_fidelity
                        self.entangled_pairs_count += 1
                        # send the entangled signal to the source node
                        self.cc_message_handler.send_message(MessageType.ENTANGLED, entangle_node, {
                            "from": self.node.name,
                            "to": entangle_node,
                            "mem_pos": mem_pos})
            elif expr.second_term.value:
                # case we have entanglement signal
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    if ready_signal.label == MessageType.ENTANGLED:
                        self.process_entangle_message(result)
            # process the message queue
            self.process_message_queue()
            # check end condition
            if self.entangled_pairs_count >= self.max_pairs:
                # send finish signal to the source node
                print_yellow(f"ManageEntangle {self.name} -> Entanglement complete")
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
