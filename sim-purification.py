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

from entangle import *
from purification import *


class FilteringExample(LocalProtocol):
    r"""Protocol for a complete filtering experiment.

    Combines the sub-protocols:
    - :py:class:`~netsquid.examples.entanglenodes.EntangleNodes`
    - :py:class:`~netsquid.examples.purify.Filter`

    Will run for specified number of times then stop, recording results after each run.

    Parameters
    ----------
    node_a : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    node_b : :py:class:`~netsquid.nodes.node.Node`
        Must be specified before protocol can start.
    num_runs : int
        Number of successful runs to do.
    epsilon : float
        Parameter used in filter's measurement operator.

    Attributes
    ----------
    results : :py:obj:`dict`
        Dictionary containing results. Results are :py:class:`numpy.array`\s.
        Results keys are *F2*, *pairs*, and *time*.

    Subprotocols
    ------------
    entangle_A : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node A.
    entangle_B : :class:`~netsquid.examples.entanglenodes.EntangleNodes`
        Entanglement generation protocol running on node B.
    purify_A : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node A.
    purify_B : :class:`~netsquid.examples.purify.Filter`
        Purification protocol running on node B.

    Notes
    -----
        The filter purification does not support the stabilizer formalism.

    """

    def __init__(self, node_a, node_b, node_c, num_runs, epsilon=0.3):
        super().__init__(nodes={"A": node_a, "B": node_b, "C": node_c}, name="Filtering example")
        self._epsilon = epsilon
        self.num_runs = num_runs
        # Initialise sub-protocols
        # entangle A -> B
        self.add_subprotocol(GenEntanglement(input_mem_pos=0,
                                             total_pairs=5,
                                             entangle_node=node_b.name,
                                             node=node_a,
                                             name=f"entangle_{node_a.name}->{node_b.name}",
                                             is_source=True,
                                             ))
        # entangle B -> A
        self.add_subprotocol(GenEntanglement(input_mem_pos=0,
                                             total_pairs=5,
                                             entangle_node=node_a.name,
                                             node=node_b,
                                             name=f"entangle_{node_b.name}->{node_a.name}",
                                             is_source=False,
                                             ))
        # entangle B -> C
        self.add_subprotocol(GenEntanglement(input_mem_pos=0,
                                             total_pairs=5,
                                             entangle_node=node_c.name,
                                             node=node_b,
                                             name=f"entangle_{node_b.name}->{node_c.name}",
                                             is_source=True,
                                             ))
        # entangle C -> B
        self.add_subprotocol(GenEntanglement(input_mem_pos=0,
                                             total_pairs=5,
                                             entangle_node=node_b.name,
                                             node=node_c,
                                             name=f"entangle_{node_c.name}->{node_b.name}",
                                             is_source=False,
                                             ))

        # purification A -> B
        self.add_subprotocol(PurifyEntangle(cc_port=node_a.get_conn_port(node_b.ID),
                                            entangle_node=node_b.name,
                                            node=node_a,
                                            name="purify_AB",
                                            target_fidelity=0.99))
        # purification B -> A
        self.add_subprotocol(PurifyEntangle(cc_port=node_b.get_conn_port(node_a.ID),
                                            entangle_node=node_a.name,
                                            node=node_b,
                                            name="purify_BA",
                                            target_fidelity=0.99))
        # purification B -> C
        self.add_subprotocol(PurifyEntangle(cc_port=node_b.get_conn_port(node_c.ID),
                                            entangle_node=node_c.name,
                                            node=node_b,
                                            name="purify_BC",
                                            target_fidelity=0.99))
        # purification C -> B
        self.add_subprotocol(PurifyEntangle(cc_port=node_c.get_conn_port(node_b.ID),
                                            entangle_node=node_b.name,
                                            node=node_c,
                                            name="purify_CB",
                                            target_fidelity=0.99))
        # Set start expressions
        self.subprotocols["purify_AB"].start_expression = (
            self.subprotocols["purify_AB"].await_signal(self.subprotocols[f"entangle_{node_a.name}->{node_b.name}"],
                                                        Signals.SUCCESS))
        self.subprotocols["purify_BA"].start_expression = (
            self.subprotocols["purify_BA"].await_signal(self.subprotocols[f"entangle_{node_b.name}->{node_a.name}"],
                                                        Signals.SUCCESS))
        self.subprotocols["purify_BC"].start_expression = (
            self.subprotocols["purify_BC"].await_signal(self.subprotocols[f"entangle_{node_b.name}->{node_c.name}"],
                                                        Signals.SUCCESS))
        self.subprotocols["purify_CB"].start_expression = (
            self.subprotocols["purify_CB"].await_signal(self.subprotocols[f"entangle_{node_c.name}->{node_b.name}"],
                                                        Signals.SUCCESS))

        # set the start expression for the entanglement protocols
        # wait for the purification protocol to send re-generation signal
        self.subprotocols[f"entangle_{node_a.name}->{node_b.name}"].re_entangle_sender = self.subprotocols["purify_AB"]
        self.subprotocols[f"entangle_{node_b.name}->{node_a.name}"].re_entangle_sender = self.subprotocols["purify_BA"]
        self.subprotocols[f"entangle_{node_b.name}->{node_c.name}"].re_entangle_sender = self.subprotocols["purify_BC"]
        self.subprotocols[f"entangle_{node_c.name}->{node_b.name}"].re_entangle_sender = self.subprotocols["purify_CB"]

        self.subprotocols["purify_AB"].add_new_signal(f"entangle_{node_a.name}->{node_b.name}")
        self.subprotocols["purify_BA"].add_new_signal(f"entangle_{node_b.name}->{node_a.name}")
        self.subprotocols["purify_BC"].add_new_signal(f"entangle_{node_b.name}->{node_c.name}")
        self.subprotocols["purify_CB"].add_new_signal(f"entangle_{node_c.name}->{node_b.name}")



        # self.subprotocols["entangle_AB"].re = (
        #     self.subprotocols["entangle_AB"].await_signal(self.subprotocols["purify_AB"],
        #                                                   "entangle"))
        # self.subprotocols["entangle_BA"].start_expression = (
        #     self.subprotocols["entangle_BA"].await_signal(self.subprotocols["purify_BA"],
        #                                                   "entangle"))
        # self.subprotocols["entangle_BC"].start_expression = (
        #     self.subprotocols["entangle_BC"].await_signal(self.subprotocols["purify_BC"],
        #                                                   "entangle"))
        # self.subprotocols["entangle_CB"].start_expression = (
        #     self.subprotocols["entangle_CB"].await_signal(self.subprotocols["purify_CB"],
        #                                                   "entangle"))

        # start_expr_ent_A = (self.subprotocols["entangle_A"].await_signal(
        #     self.subprotocols["purify_A"], Signals.FAIL) |
        #                     self.subprotocols["entangle_A"].await_signal(
        #                         self, Signals.WAITING))
        # start_expr_ent_B = (self.subprotocols["entangle_B"].await_signal(
        #     self.subprotocols["purify_B"], Signals.FAIL) |
        #                     self.subprotocols["entangle_B"].await_signal(
        #                         self, Signals.WAITING))
        # start_expr_ent_C = (self.subprotocols["entangle_C"].await_signal(
        #     self.subprotocols["purify_C"], Signals.FAIL) |
        #                     self.subprotocols["entangle_C"].await_signal(
        #                         self, Signals.WAITING))

        # self.subprotocols["entangle_A"].start_expression = start_expr_ent_A
        # self.subprotocols["entangle_B"].start_expression = start_expr_ent_B
        # self.subprotocols["entangle_C"].start_expression = start_expr_ent_C

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            # self.subprotocols["entangle_A"].right_entangled_pairs = 0
            # self.send_signal(Signals.WAITING)
            yield (self.await_signal(self.subprotocols["purify_AB"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_BA"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_BC"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_CB"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_AB"].get_signal_result(Signals.SUCCESS,
                                                                        self)
            signal_B = self.subprotocols["purify_BC"].get_signal_result(Signals.SUCCESS,
                                                                        self)
            # signal_C = self.subprotocols["purify_C"].get_signal_result(Signals.SUCCESS,
            #                                                            self)
            result = {
                "AB": signal_A,
                "BC": signal_B,
                "start_time": start_time,
            }
            print(result)
            self.send_signal(Signals.SUCCESS, result)

            for subprotocol in self.subprotocols.values():
                subprotocol.reset()
            # self.reset()


def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1e-3,
                          node_distance=50, nodes_list=["node_A", "node_B", "node_C"], network_name="purify_network"):
    """Create an example network for use with the purification protocols.

    Connection flow:

    A -> B -> C

    A Component Diagram:
    - Left Quantum memory
    - Right Quantum memory
    - QSource
    A Connection Diagram:
    - Classical channel to B
    - Quantum channel to B (will send qubit to B during Entangle protocol)
    - Quantum channel to itself, mapped to right quantum memory qout -> right_qmemory.qin0

    B Component Diagram:
    - Left Quantum memory
    - Right Quantum memory
    - QSource
    B Connection Diagram:
    - Classical channel to A
    - Classical channel to C
    - Quantum channel to C (will send qubit to C during Entangle protocol)
    - Quantum channel from A, directly forward -> left_qmemory.qin0
    - Quantum channel to itself, mapped to right quantum memory qout -> right_qmemory.qin0

    C Component Diagram:
    - Left Quantum memory

    C Connection Diagram:
    - Classical channel to B
    - Quantum channel from B, directly forward -> left_qmemory.qin0


    Returns
    -------
    :class:`~netsquid.components.component.Component`
        A network component with nodes and channels as subcomponents.

    Notes
    -----
        This network is also used by the matching integration test.

    """
    network = Network(network_name)
    nodes = network.add_nodes(nodes_list)

    # add components to the nodes
    for index, node in enumerate(nodes):
        state_sampler = StateSampler([ns.b00], [1])
        node.add_subcomponent(QSource(name=f"QSource_{node.name}", state_sampler=state_sampler,
                                      models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
                                      num_ports=1, status=SourceStatus.EXTERNAL))
        if index - 1 >= 0:
            node.add_subcomponent(QuantumProcessor(name=nodes[index - 1].name + "_qmemory",
                                                   num_positions=10,
                                                   fallback_to_nonphysical=True,
                                                   memory_noise_models=DepolarNoiseModel(depolar_rate)))
        if index + 1 < len(nodes):
            # case of we are the source node
            node.add_subcomponent(QuantumProcessor(name=nodes[index + 1].name + "_qmemory",
                                                   num_positions=10,
                                                   fallback_to_nonphysical=True,
                                                   memory_noise_models=DepolarNoiseModel(depolar_rate)))

    # add connections between the nodes
    for index, node in enumerate(nodes):

        if index + 1 < len(nodes):
            right_node = nodes[index + 1]
            # case of we are the source node
            internal_qchannel = QuantumChannel(name=f"QChannel_{node.name}->{node.name}", length=0,
                                               models={"quantum_loss_model": None,
                                                       "delay_model": FibreDelayModel(c=200e3)},
                                               depolar_rate=depolar_rate)
            # internal qchannel to link right_qmemory for source node
            node.add_subcomponent(internal_qchannel, name="internal_qchannel")
            (node.subcomponents["internal_qchannel"].ports["recv"]
             .connect(node.subcomponents[right_node.name + "_qmemory"].ports["qin0"]))
            # create a quantum channel between the source node and the next node
            qchannel = QuantumChannel(name=f"QChannel_{node.name}->{right_node.name}", length=node_distance,
                                      models={"quantum_loss_model": None,
                                              "delay_model": FibreDelayModel(c=200e3)},
                                      depolar_rate=depolar_rate)

            port_name_a, port_name_b = network.add_connection(
                node, right_node, channel_to=qchannel, label="quantum",
                port_name_node1=f"qout_{nodes[index + 1].name}",
                port_name_node2=f"qin_{node.name}")
            # map the input from node to right_node's qmemory, which is the memory of the left node
            right_node.ports[port_name_b].forward_input(right_node.subcomponents[f"{node.name}_qmemory"].ports[f"qin0"])

            # Add the classical channel between the nodes
            for j in range(index + 1, len(nodes)):
                conn_cchannel = DirectConnection(
                    f"CChannelConn_{nodes[index].name}_{nodes[j].name}",
                    ClassicalChannel(f"CChannel_{nodes[index].name}->{nodes[j].name}", length=node_distance,
                                     models={"delay_model": FibreDelayModel(c=200e3)}),
                    ClassicalChannel(f"CChannel_{nodes[j].name}->{nodes[index].name}", length=node_distance,
                                     models={"delay_model": FibreDelayModel(c=200e3)}))
                network.add_connection(node, nodes[j], connection=conn_cchannel)

    # node_a, node_b, node_c = network.add_nodes(["node_A", "node_B", "node_C"])
    # node_a.add_subcomponent(QuantumProcessor(
    #     "right_qmemory", num_positions=10, fallback_to_nonphysical=True,
    #     memory_noise_models=DepolarNoiseModel(depolar_rate)))
    # state_sampler_a = StateSampler(
    #     [ns.b00], [1])
    # node_a.add_subcomponent(QSource(
    #     "QSource_A", state_sampler=state_sampler_a,
    #     models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
    #     num_ports=1, status=SourceStatus.EXTERNAL))
    #
    # node_b.add_subcomponent(QuantumProcessor(
    #     "right_qmemory", num_positions=10, fallback_to_nonphysical=True,
    #     memory_noise_models=DepolarNoiseModel(depolar_rate)))
    # node_b.add_subcomponent(QuantumProcessor(
    #     "left_qmemory", num_positions=10, fallback_to_nonphysical=True,
    #     memory_noise_models=DepolarNoiseModel(depolar_rate)))
    # state_sampler_b = StateSampler(
    #     [ns.b00], [1])
    # node_b.add_subcomponent(QSource(
    #     "QSource_B", state_sampler=state_sampler_b,
    #     models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
    #     num_ports=1, status=SourceStatus.EXTERNAL))
    #
    # node_c.add_subcomponent(QuantumProcessor(
    #     "left_qmemory", num_positions=10, fallback_to_nonphysical=True,
    #     memory_noise_models=DepolarNoiseModel(depolar_rate)))
    #
    # a_b_conn_cchannel = DirectConnection(
    #     "CChannelConn_AB",
    #     ClassicalChannel("CChannel_A->B", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}),
    #     ClassicalChannel("CChannel_B->A", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}))
    # network.add_connection(node_a, node_b, connection=a_b_conn_cchannel)
    #
    # b_c_conn_cchannel = DirectConnection(
    #     "CChannelConn_BC",
    #     ClassicalChannel("CChannel_B->C", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}),
    #     ClassicalChannel("CChannel_C->B", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}))
    # network.add_connection(node_b, node_c, connection=b_c_conn_cchannel)
    #
    # # node_A.connect_to(node_B, conn_cchannel)
    # a_b_qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
    #                               models={"quantum_loss_model": None,
    #                                       "delay_model": FibreDelayModel(c=200e3)},
    #                               depolar_rate=depolar_rate)
    # a_internal_qchannel = QuantumChannel("QChannel_A->A", length=0,
    #                                      models={"quantum_loss_model": None,
    #                                              "delay_model": FibreDelayModel(c=200e3)},
    #                                      depolar_rate=depolar_rate)
    # # internal qchannel to link right_qmemory for node A
    # node_a.add_subcomponent(a_internal_qchannel, name="internal_qchannel")
    # (node_a.subcomponents["internal_qchannel"].ports["recv"].
    #  connect(node_a.subcomponents["right_qmemory"].ports["qin0"]))
    #
    # port_name_a, port_name_b = network.add_connection(
    #     node_a, node_b, channel_to=a_b_qchannel, label="quantum", port_name_node1="qout0", port_name_node2="qin0")
    # print(f"Added connection between {node_a.name} and {node_b.name} with ports {port_name_a} and {port_name_b}")
    # # Link Alice ports:
    # # node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
    # #     node_a.ports[port_name_a])
    # # node_a.subcomponents["QSource_A"].ports["qout0"].connect(
    # #     node_a.subcomponents["right_qmemory"].ports["qin0"])
    # # Link Bob ports:
    # node_b.ports[port_name_b].forward_input(node_b.subcomponents["left_qmemory"].ports[f"qin0"])
    #
    # # node_B.connect_to(node_C, conn_cchannel)
    # b_c_qchannel = QuantumChannel("QChannel_B->C", length=node_distance,
    #                               models={"quantum_loss_model": None,
    #                                       "delay_model": FibreDelayModel(c=200e3)},
    #                               depolar_rate=0)
    #
    # # internal qchannel to link left_qmemory for node B
    # b_internal_qchannel = QuantumChannel("QChannel_B->B", length=0,
    #                                      models={"quantum_loss_model": None,
    #                                              "delay_model": FibreDelayModel(c=200e3)},
    #                                      depolar_rate=0)
    # node_b.add_subcomponent(b_internal_qchannel, name="internal_qchannel")
    # (node_b.subcomponents["internal_qchannel"].ports["recv"]
    #  .connect(node_b.subcomponents["right_qmemory"].ports["qin0"]))
    #
    # port_name_b, port_name_c = network.add_connection(
    #     node_b, node_c, channel_to=b_c_qchannel, label="quantum", port_name_node1="qout0", port_name_node2="qin0")
    # print(f"Added connection between {node_b.name} and {node_c.name} with ports {port_name_b} and {port_name_c}")
    # # Link Bob ports:
    # # node_b.subcomponents["QSource_B"].ports["qout1"].forward_output(
    # #     node_b.ports[port_name_b])
    # # node_b.subcomponents["QSource_B"].ports["qout0"].connect(
    # #     node_b.subcomponents["right_qmemory"].ports["qin0"])
    # # Link Charlie ports:
    # node_c.ports[port_name_c].forward_input(node_c.subcomponents["left_qmemory"].ports[f"qin0"])

    return network


def example_sim_setup(node_a, node_b, node_c, num_runs, epsilon=0.3):
    """Example simulation setup for purification protocols.

    Returns
    -------
    :class:`~netsquid.examples.purify.FilteringExample`
        Example protocol to run.
    :class:`pandas.DataFrame`
        Dataframe of collected data.

    """
    filt_example = FilteringExample(node_a, node_b, node_c, num_runs=num_runs, epsilon=0.3)

    def record_run(evexpr):
        # Callback that collects data each run
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # Record fidelity
        a_b_entangled_pairs = result["AB"]["satisfied_pairs"]
        b_c_entangled_pairs = result["BC"]["satisfied_pairs"]
        print(f"AB entangled pairs: {a_b_entangled_pairs}")
        print(f"BC entangled pairs: {b_c_entangled_pairs}")
        a_b_actual_fidelities = []
        for mem_pos, theoretical_fidelity in a_b_entangled_pairs.items():
            print_red(f"AB theoretical fidelity at memory position {mem_pos}: {theoretical_fidelity}")
            q_a = node_a.subcomponents[f"node_B_qmemory"].peek(mem_pos)[0]
            q_b = node_b.subcomponents[f"node_A_qmemory"].peek(mem_pos)[0]
            f = fidelity([q_a, q_b], ks.b00)
            a_b_actual_fidelities.append(f)
            print_green(f"AB fidelity at memory position {mem_pos}: {f}")
        average_ab_fidelity = np.mean(list(a_b_entangled_pairs.values()))
        average_ab_actual_fidelity = np.mean(a_b_actual_fidelities)
        b_c_actual_fidelities = []
        for mem_pos, theoretical_fidelity in b_c_entangled_pairs.items():
            print_red(f"BC theoretical fidelity at memory position {mem_pos}: {theoretical_fidelity}")
            q_b = node_b.subcomponents[f"node_C_qmemory"].peek(mem_pos)[0]
            q_c = node_c.subcomponents[f"node_B_qmemory"].peek(mem_pos)[0]
            f = fidelity([q_b, q_c], ks.b00)
            b_c_actual_fidelities.append(f)
            print_green(f"BC fidelity at memory position {mem_pos}: {f}")
        average_bc_fidelity = np.mean(list(b_c_entangled_pairs.values()))
        average_bc_actual_fidelity = np.mean(b_c_actual_fidelities)

        # count the purified count
        a_b_purify_count = result["AB"]["purification_count"]
        b_c_purify_count = result["BC"]["purification_count"]
        print(f"AB purification count: {a_b_purify_count}")
        print(f"BC purification count: {b_c_purify_count}")

        # count the purification success count
        a_b_purify_success_count = result["AB"]["purification_success_count"]
        b_c_purify_success_count = result["BC"]["purification_success_count"]
        print(f"AB purification success count: {a_b_purify_success_count}")
        print(f"BC purification success count: {b_c_purify_success_count}")

        # calculate the purification process time
        a_b_purify_time = result["AB"]["finish_time"] - result["start_time"]
        b_c_purify_time = result["BC"]["finish_time"] - result["start_time"]
        print(f"AB purification time: {a_b_purify_time / 1e9}")
        print(f"BC purification time: {b_c_purify_time / 1e9}")
        # store in dic
        data = {"AB": a_b_entangled_pairs, "BC": b_c_entangled_pairs,
                "AB_purify_count": a_b_purify_count, "BC_purify_count": b_c_purify_count,
                "AB_purify_success_count": a_b_purify_success_count,
                "BC_purify_success_count": b_c_purify_success_count,
                "AB_purify_time": a_b_purify_time,
                "BC_purify_time": b_c_purify_time,
                "AB_average_fidelity": average_ab_fidelity,
                "BC_average_fidelity": average_bc_fidelity,
                "AB_actual_average_fidelity": average_ab_actual_fidelity,
                "BC_actual_average_fidelity": average_bc_actual_fidelity}
        # pretty print the data
        return data

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=filt_example,
                                     event_type=Signals.SUCCESS.value))
    return filt_example, dc


if __name__ == "__main__":
    network = example_network_setup()
    filt_example, dc = example_sim_setup(network.get_node("node_A"),
                                         network.get_node("node_B"),
                                         network.get_node("node_C"),
                                         num_runs=10)
    filt_example.start()
    ns.sim_run()
    collected_data = dc.dataframe
    print(f"Average AB Fidelity: {collected_data['AB_average_fidelity'].mean()}")
    print(f"Average AB Actual Fidelity: {collected_data['AB_actual_average_fidelity'].mean()}")
    print(f"Average BC Fidelity: {collected_data['BC_average_fidelity'].mean()}")
    print(f"Average BC Actual Fidelity: {collected_data['BC_actual_average_fidelity'].mean()}")
    print(f"Average AB Purification Time: {collected_data['AB_purify_time'].mean() / 1e9}")
    print(f"Average BC Purification Time: {collected_data['BC_purify_time'].mean() / 1e9}")
    print(f"Average AB Purification Count: {collected_data['AB_purify_count'].mean()}")
    print(f"Average BC Purification Count: {collected_data['BC_purify_count'].mean()}")
    print(f"Average AB Purification Success Count: {collected_data['AB_purify_success_count'].mean()}")
    print(f"Average BC Purification Success Count: {collected_data['BC_purify_success_count'].mean()}")
