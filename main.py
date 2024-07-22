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

        self.add_subprotocol(DirectionalEntanglementProtolMultiMem(
            node=node_a, left_node=None, right_node=node_b.name,
            left_total_pairs=5, right_total_pairs=5, name="entangle_A",
            left_input_mem_pos=0, right_input_mem_pos=0,
        ))
        self.add_subprotocol(DirectionalEntanglementProtolMultiMem(
            node=node_b, left_node=node_a.name, right_node=node_c.name,
            left_total_pairs=5, right_total_pairs=5, name="entangle_B",
            left_input_mem_pos=0, right_input_mem_pos=0,
        ))
        self.add_subprotocol(DirectionalEntanglementProtolMultiMem(
            node=node_c, left_node=node_b.name, right_node=None,
            left_total_pairs=5, right_total_pairs=5, name="entangle_C",
            left_input_mem_pos=0, right_input_mem_pos=0,
        ))

        self.add_subprotocol(Purification(node_a,
                                          left_port=None,
                                          right_port=node_a.get_conn_port(node_b.ID),
                                          name="purify_A", target_fidelity=0.9))
        self.add_subprotocol(Purification(node_b,
                                          left_port=node_b.get_conn_port(node_a.ID),
                                          right_port=node_b.get_conn_port(node_c.ID),
                                          name="purify_B", target_fidelity=0.9))
        self.add_subprotocol(Purification(node_c,
                                          left_port=node_c.get_conn_port(node_b.ID),
                                          right_port=None,
                                          name="purify_C", target_fidelity=0.9))

        # self.add_subprotocol(Filter(node_a, node_a.get_conn_port(node_b.ID),
        #                             epsilon=epsilon, name="purify_A"))
        # self.add_subprotocol(Filter(node_b, node_b.get_conn_port(node_a.ID),
        #                             epsilon=epsilon, name="purify_B"))
        # self.add_subprotocol(Filter(node_c, node_c.get_conn_port(node_b.ID),
        #                             epsilon=epsilon, name="purify_C"))

        # Set start expressions
        self.subprotocols["purify_A"].start_expression = (
            self.subprotocols["purify_A"].await_signal(self.subprotocols["entangle_A"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_B"].start_expression = (
            self.subprotocols["purify_B"].await_signal(self.subprotocols["entangle_B"],
                                                       Signals.SUCCESS))
        self.subprotocols["purify_C"].start_expression = (
            self.subprotocols["purify_C"].await_signal(self.subprotocols["entangle_C"],
                                                       Signals.SUCCESS))
        # set the start expression for the entanglement protocols
        # wait for the purification protocol to send re-generation signal
        self.subprotocols["entangle_A"].start_expression = (
            self.subprotocols["entangle_A"].await_signal(self.subprotocols["purify_A"],
                                                         "entangle"))
        self.subprotocols["entangle_B"].start_expression = (
            self.subprotocols["entangle_B"].await_signal(self.subprotocols["purify_B"],
                                                         "entangle"))
        self.subprotocols["entangle_C"].start_expression = (
            self.subprotocols["entangle_C"].await_signal(self.subprotocols["purify_C"],
                                                         "entangle"))

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
            yield (self.await_signal(self.subprotocols["purify_A"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_B"], Signals.SUCCESS) &
                   self.await_signal(self.subprotocols["purify_C"], Signals.SUCCESS))
            signal_A = self.subprotocols["purify_A"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_B = self.subprotocols["purify_B"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            signal_C = self.subprotocols["purify_C"].get_signal_result(Signals.SUCCESS,
                                                                       self)
            result = {
                "pos_A": signal_A,
                "pos_B": signal_B,
                "pos_C": signal_C,
                "time": sim_time() - start_time,

            }
            print(result)
            self.send_signal(Signals.SUCCESS, result)


def example_network_setup(source_delay=1e5, source_fidelity_sq=0.8, depolar_rate=1,
                          node_distance=20):
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
    network = Network("purify_network")

    node_a, node_b, node_c = network.add_nodes(["node_A", "node_B", "node_C"])
    node_a.add_subcomponent(QuantumProcessor(
        "right_qmemory", num_positions=10, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    state_sampler_a = StateSampler(
        [ns.b00], [1])
    node_a.add_subcomponent(QSource(
        "QSource_A", state_sampler=state_sampler_a,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=1, status=SourceStatus.EXTERNAL))

    node_b.add_subcomponent(QuantumProcessor(
        "right_qmemory", num_positions=10, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    node_b.add_subcomponent(QuantumProcessor(
        "left_qmemory", num_positions=10, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))
    state_sampler_b = StateSampler(
        [ns.b00], [1])
    node_b.add_subcomponent(QSource(
        "QSource_B", state_sampler=state_sampler_b,
        models={"emission_delay_model": FixedDelayModel(delay=source_delay)},
        num_ports=1, status=SourceStatus.EXTERNAL))

    node_c.add_subcomponent(QuantumProcessor(
        "left_qmemory", num_positions=10, fallback_to_nonphysical=True,
        memory_noise_models=DepolarNoiseModel(depolar_rate)))

    a_b_conn_cchannel = DirectConnection(
        "CChannelConn_AB",
        ClassicalChannel("CChannel_A->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_B->A", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_a, node_b, connection=a_b_conn_cchannel)
    # a_b_conn_cchannel_entangle = DirectConnection(
    #     "CChannelConn_AB_entangle",
    #     ClassicalChannel("CChannel_A->B", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}),
    #     ClassicalChannel("CChannel_B->A", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}))
    # network.add_connection(node_a, node_b, connection=a_b_conn_cchannel_entangle)

    b_c_conn_cchannel = DirectConnection(
        "CChannelConn_BC",
        ClassicalChannel("CChannel_B->C", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}),
        ClassicalChannel("CChannel_C->B", length=node_distance,
                         models={"delay_model": FibreDelayModel(c=200e3)}))
    network.add_connection(node_b, node_c, connection=b_c_conn_cchannel)

    # b_c_conn_cchannel_entangle = DirectConnection(
    #     "CChannelConn_BC_entangle",
    #     ClassicalChannel("CChannel_B->C", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}),
    #     ClassicalChannel("CChannel_C->B", length=node_distance,
    #                      models={"delay_model": FibreDelayModel(c=200e3)}))
    # network.add_connection(node_b, node_c, connection=b_c_conn_cchannel_entangle)

    # node_A.connect_to(node_B, conn_cchannel)
    a_b_qchannel = QuantumChannel("QChannel_A->B", length=node_distance,
                                  models={"quantum_loss_model": None,
                                          "delay_model": FibreDelayModel(c=200e3)},
                                  depolar_rate=0)
    a_internal_qchannel = QuantumChannel("QChannel_A->A", length=0,
                                         models={"quantum_loss_model": None,
                                                 "delay_model": FibreDelayModel(c=200e3)},
                                         depolar_rate=0)
    # internal qchannel to link right_qmemory for node A
    node_a.add_subcomponent(a_internal_qchannel, name="internal_qchannel")
    (node_a.subcomponents["internal_qchannel"].ports["recv"].
     connect(node_a.subcomponents["right_qmemory"].ports["qin0"]))

    port_name_a, port_name_b = network.add_connection(
        node_a, node_b, channel_to=a_b_qchannel, label="quantum", port_name_node1="qout0", port_name_node2="qin0")
    print(f"Added connection between {node_a.name} and {node_b.name} with ports {port_name_a} and {port_name_b}")
    # Link Alice ports:
    # node_a.subcomponents["QSource_A"].ports["qout1"].forward_output(
    #     node_a.ports[port_name_a])
    # node_a.subcomponents["QSource_A"].ports["qout0"].connect(
    #     node_a.subcomponents["right_qmemory"].ports["qin0"])
    # Link Bob ports:
    node_b.ports[port_name_b].forward_input(node_b.subcomponents["left_qmemory"].ports[f"qin0"])

    # node_B.connect_to(node_C, conn_cchannel)
    b_c_qchannel = QuantumChannel("QChannel_B->C", length=node_distance,
                                  models={"quantum_loss_model": None,
                                          "delay_model": FibreDelayModel(c=200e3)},
                                  depolar_rate=0)

    # internal qchannel to link left_qmemory for node B
    b_internal_qchannel = QuantumChannel("QChannel_B->B", length=0,
                                         models={"quantum_loss_model": None,
                                                 "delay_model": FibreDelayModel(c=200e3)},
                                         depolar_rate=0)
    node_b.add_subcomponent(b_internal_qchannel, name="internal_qchannel")
    (node_b.subcomponents["internal_qchannel"].ports["recv"]
     .connect(node_b.subcomponents["right_qmemory"].ports["qin0"]))

    port_name_b, port_name_c = network.add_connection(
        node_b, node_c, channel_to=b_c_qchannel, label="quantum", port_name_node1="qout0", port_name_node2="qin0")
    print(f"Added connection between {node_b.name} and {node_c.name} with ports {port_name_b} and {port_name_c}")
    # Link Bob ports:
    # node_b.subcomponents["QSource_B"].ports["qout1"].forward_output(
    #     node_b.ports[port_name_b])
    # node_b.subcomponents["QSource_B"].ports["qout0"].connect(
    #     node_b.subcomponents["right_qmemory"].ports["qin0"])
    # Link Charlie ports:
    node_c.ports[port_name_c].forward_input(node_c.subcomponents["left_qmemory"].ports[f"qin0"])

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
        q_A, = node_a.qmemory.pop(positions=[result["pos_A"]])
        q_B, = node_b.qmemory.pop(positions=[result["pos_B"]])
        q_C, = node_c.qmemory.pop(positions=[result["pos_C"]])
        f2 = qapi.fidelity([q_A, q_B], ks.b01, squared=True)
        f3 = qapi.fidelity([q_B, q_C], ks.b01, squared=True)
        print(f"{sim_time():.1f}: Fidelity = {f2:.3f}")
        return {"F2": f2, "pairs": result["pairs"], "time": result["time"]}

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
                                         num_runs=1)
    filt_example.start()
    ns.sim_run()