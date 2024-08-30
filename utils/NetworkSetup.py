import numpy as np

import netsquid as ns
from netsquid.nodes.network import Network
from netsquid.nodes.connections import DirectConnection
from netsquid.components import ClassicalChannel, QuantumChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits.state_sampler import StateSampler


def calculate_channel_depolar_rate(length_km, loss_db_per_km=0.2, c=200e3):
    # c is speed of light in fiber (m/s)
    loss_rate = 1 - 10 ** (-loss_db_per_km * length_km / 10)
    transit_time = (length_km * 1000) / c  # in seconds
    depolar_rate = -np.log(1 - loss_rate) / transit_time
    return depolar_rate


def setup_network(nodes_list, network_name,
                  memory_capacity=10,
                  source_delay=1e5,
                  memory_depolar_rate=10,
                  node_distance=20, ):
    """
    Create a network with nodes and channels.

    Connection flow:

    A -> B -> C


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
                                      num_ports=1, status=SourceStatus.EXTERNAL))
        if index - 1 >= 0:
            node.add_subcomponent(QuantumProcessor(name=nodes[index - 1].name + "_qmemory",
                                                   num_positions=memory_capacity,
                                                   fallback_to_nonphysical=True,
                                                   memory_noise_models=
                                                   [DepolarNoiseModel(memory_depolar_rate)] * memory_capacity))
        if index + 1 < len(nodes):
            # case of we are the source node
            node.add_subcomponent(QuantumProcessor(name=nodes[index + 1].name + "_qmemory",
                                                   num_positions=memory_capacity,
                                                   fallback_to_nonphysical=True,
                                                   memory_noise_models=
                                                   [DepolarNoiseModel(memory_depolar_rate)] * memory_capacity))
    # get qchannels deploar rate
    qchannel_depolar_rate = calculate_channel_depolar_rate(node_distance)
    # add connections between the nodes
    for index, node in enumerate(nodes):

        if index + 1 < len(nodes):
            right_node = nodes[index + 1]
            # case of we are the source node
            internal_qchannel = QuantumChannel(name=f"QChannel_{node.name}->{node.name}", length=0,
                                               models={"quantum_loss_model": None,
                                                       # "delay_model": FibreDelayModel(c=200e3),
                                                       "quantum_noise_model": DepolarNoiseModel(qchannel_depolar_rate)})
            # internal qchannel to link right_qmemory for source node
            node.add_subcomponent(internal_qchannel, name="internal_qchannel")
            (node.subcomponents["internal_qchannel"].ports["recv"]
             .connect(node.subcomponents[right_node.name + "_qmemory"].ports["qin0"]))
            # create a quantum channel between the source node and the next node
            qchannel = QuantumChannel(name=f"QChannel_{node.name}->{right_node.name}", length=node_distance,
                                      models={"quantum_loss_model": None,
                                              # "delay_model": FibreDelayModel(c=200e3),
                                              "quantum_noise_model": DepolarNoiseModel(qchannel_depolar_rate)})

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
    return network
