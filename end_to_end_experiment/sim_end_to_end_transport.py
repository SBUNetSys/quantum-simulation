import json
import os

import numpy as np
import pydynaa as pd
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
import netsquid.qubits.operators as ops
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals

from utils.NetworkSetup import setup_network
from utils import Logging, GenSwappingTree
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
from protocols.EndToEnd import EndToEndProtocol
from protocols.Transport import Transportation


class EndToEndTransportWithPurificationExample(LocalProtocol):
    """
    A simple example of a swapping protocol.
    """

    def __init__(self, network_nodes: list,
                 num_runs=1,
                 node_path=None,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 qubits_to_transport=1):
        if node_path is None:
            raise ValueError("node_path must be provided")
        # generate the swapping tree and levels
        swapping_nodes, _, _ = GenSwappingTree.generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.final_entanglement = (node_path[0], node_path[-1])
        self.all_nodes = network_nodes
        self.max_entangle_pairs = max_entangle_pairs
        self.logger = Logging.Logger("EndToEnd", logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)

        super().__init__(nodes={node.name: node for node in network_nodes}, name="EndToEndExample")
        self.num_runs = num_runs
        # Initialize the entangle protocol
        for index, node in enumerate(network_nodes):
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            lower_protocols = []
            # Initialize the GenEntanglement protocol and EntanglementHandler protocol
            if index - 1 >= 0:
                # case of we have a previous node
                gen_protocol = GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger
                )
                self.add_subprotocol(gen_protocol)
                eh_handler = EntanglementHandler(node=node,
                                                 name=f"entanglement_handler_{node.name}->{network_nodes[index - 1].name}",
                                                 num_pairs=self.max_entangle_pairs,
                                                 qubit_input_protocol=gen_protocol,
                                                 cc_message_handler=self.subprotocols[
                                                     f"message_handler_{node.name}"],
                                                 entangle_node=network_nodes[index - 1].name,
                                                 memory_depolar_rate=memory_depolar_rate,
                                                 node_distance=node_distance,
                                                 is_top_layer=False,
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                # add purification
                pure_protocol = Purification(node=node,
                                             name=f"purify_{node.name}->{network_nodes[index - 1].name}",
                                             entangled_node=network_nodes[index - 1].name,
                                             entanglement_handler=eh_handler,
                                             cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                             max_entangled_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                lower_protocols.append(pure_protocol)

            if index + 1 < len(network_nodes):
                # case of we have a next node
                gen_protocol = GenEntanglement(
                    input_mem_pos=0,
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger
                )
                self.add_subprotocol(gen_protocol)
                eh_handler = EntanglementHandler(node=node,
                                                 name=f"entanglement_handler_{node.name}->{network_nodes[index + 1].name}",
                                                 num_pairs=self.max_entangle_pairs,
                                                 qubit_input_protocol=gen_protocol,
                                                 cc_message_handler=self.subprotocols[
                                                     f"message_handler_{node.name}"],
                                                 entangle_node=network_nodes[index + 1].name,
                                                 memory_depolar_rate=memory_depolar_rate,
                                                 node_distance=node_distance,
                                                 is_top_layer=False,
                                                 logger=null_logger
                                                 )
                self.add_subprotocol(eh_handler)
                gen_protocol.entanglement_handler = eh_handler
                pure_protocol = Purification(node=node,
                                             name=f"purify_{node.name}->{network_nodes[index + 1].name}",
                                             entangled_node=network_nodes[index + 1].name,
                                             entanglement_handler=eh_handler,
                                             cc_message_handler=self.subprotocols[
                                                 f"message_handler_{node.name}"],
                                             max_entangled_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
                                             )
                # Initialize the purification protocol
                self.add_subprotocol(pure_protocol)
                lower_protocols.append(pure_protocol)
            # Add end to end protocol to each node
            end_to_end = EndToEndProtocol(node=node,
                                          name=f"e2e_{node.name}",
                                          swapping_nodes=self.swap_nodes,
                                          final_entanglement=self.final_entanglement,
                                          cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                          qubit_ready_protocols=lower_protocols,
                                          max_pairs=self.max_entangle_pairs - 1,
                                          logger=self.logger,
                                          is_top_layer=False)
            self.add_subprotocol(end_to_end)
            if index == 0:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[-1].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)
            if index == len(network_nodes) - 1:
                transport = Transportation(node=node,
                                           name=f"e2e_transport_{node.name}",
                                           qubit_ready_protocols=[end_to_end],
                                           entangled_node=network_nodes[0].name,
                                           source=network_nodes[0].name,
                                           destination=network_nodes[-1].name,
                                           cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                           transmitting_qubit_size=qubits_to_transport,
                                           logger=self.logger,
                                           is_top_layer=True,
                                           )
                self.add_subprotocol(transport)

    def run(self):

        end_time = None
        self.start_subprotocols()
        start_time = sim_time()
        for index in range(self.num_runs):
            if end_time is not None:
                start_time = sim_time()
            yield self.await_signal(self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"e2e_transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_FINISHED,self)
            """
            result = {entangle_node: name, mem_poses:[]}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "duration": end_time - start_time, }
            for mem_pos, fid in results["results"].items():
                result_dic["total_count"] += 1
                # get the qubit
                # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                #     mem_pos, skip_noise=True)[0]
                # # measure the state
                # fidelity = qapi.fidelity(qubit, ns.y0)
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
            # final success rate
            result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports


def example_sim_run_with_purification(nodes,
                                      num_runs,
                                      memory_depolar_rate,
                                      node_distance,
                                      max_entangle_pairs,
                                      target_fidelity,
                                      qubits_to_transport):
    e2e_example = EndToEndTransportWithPurificationExample(network_nodes=nodes,
                                                           num_runs=num_runs,
                                                           node_path=[node.name for node in nodes],
                                                           max_entangle_pairs=max_entangle_pairs,
                                                           memory_depolar_rate=memory_depolar_rate,
                                                           node_distance=node_distance,
                                                           target_fidelity=target_fidelity,
                                                           qubits_to_transport=qubits_to_transport, )

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        # print(f"Run completed: {result}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=e2e_example, event_type=Signals.SUCCESS.value))
    return e2e_example, dc


def run_test_example_with_purification(qubit_number=1):
    nodes_list = [f"Node_{i}" for i in range(3)]
    network = setup_network(nodes_list, "hop-by-hop-transportation",
                            memory_capacity=128, memory_depolar_rate=100,
                            node_distance=3, source_delay=1)
    # create a protocol to entangle two nodes
    sample_nodes = [node for node in network.nodes.values()]
    transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                              num_runs=1000,
                                                              memory_depolar_rate=100,
                                                              node_distance=3,
                                                              max_entangle_pairs=2,
                                                              target_fidelity=0.995,
                                                              qubits_to_transport=qubit_number)
    # Run the simulation
    transport_example.start()
    ns.sim_run()
    # Collect the data
    results = dc.dataframe
    print(results.columns)
    print(results)


def run_multi_node(max_node, qubit_number=1):
    final_data = {}
    os.makedirs("./transportation_results", exist_ok=True)
    for node_count in range(3, max_node + 1):
        node_data = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network(nodes_list, "end-to-end-transportation",
                                memory_capacity=128, memory_depolar_rate=100,
                                node_distance=3, source_delay=1)
        # create a protocol to entangle two nodes
        sample_nodes = [node for node in network.nodes.values()]
        transport_example, dc = example_sim_run_with_purification(sample_nodes,
                                                                  num_runs=1000,
                                                                  memory_depolar_rate=100,
                                                                  node_distance=3,
                                                                  max_entangle_pairs=2,
                                                                  target_fidelity=0.995,
                                                                  qubits_to_transport=qubit_number)
        # Run the simulation
        transport_example.start()
        ns.sim_run()
        # Collect the data
        collected_data = dc.dataframe
        for c in collected_data.columns:
            node_data[c] = collected_data[c].mean()
            if len(collected_data[c]) < 1000:
                print(f"Failed Finished 1000 run {len(collected_data[c])}/1000")
            print(f"{node_count}->{c}: {collected_data[c].mean()}")
        final_data[node_count] = node_data
        transport_example.stop()
        ns.sim_reset()
    with open(f"./transportation_results/e2e_transport_{max_node}.json", "w") as f:
        json.dump(final_data, f)


if __name__ == '__main__':
    # seed = np.random.randint(0, 10000)
    # seed = 764
    # np.random.seed(seed)
    # print(f'{seed}')
    # run_test_example_with_purification()
    run_multi_node(11)
