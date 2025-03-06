import gc
import json
import os
import traceback
from collections import defaultdict
from weakref import finalize

import numpy as np
import pydynaa as pd
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.qubits import qubitapi as qapi
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals
from netsquid.qubits import ketstates as ks
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network
from utils import Logging, GenSwappingTree
from utils.Gates import controlled_unitary, measure_operator
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandler import EntanglementHandler
from protocols.GenEntanglement import GenEntanglement
from protocols.Purification import Purification
from protocols.Verification import Verification
from protocols.SecurityVerification import SecurityVerification
from protocols.Transport import Transportation
from protocols.EndToEnd import EndToEndProtocol
from protocols.Security import Security

import netsquid.qubits.operators as ops
from utils.SignalMessages import ProtocolFinishedSignalMessage


class SecurityWithVerificationExample(LocalProtocol):
    """
    Protocol for a complete verification example.
    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 m_size=3,
                 batch_size=10,
                 qubits_to_transport=1,
                 skip_noise=False,
                 CU_gate=None,
                 CCU_gate=None,
                 node_path=None, ):
        if len(network_nodes) < 1:
            raise ValueError("This protocol requires at least nodes.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.m_size = m_size
        self.batch_size = batch_size
        self.qubits_to_transport = qubits_to_transport
        swapping_nodes, _, _ = GenSwappingTree.generate_swapping_tree(node_path)
        self.swap_nodes = swapping_nodes
        self.final_entanglement = (node_path[0], node_path[-1])
        super().__init__(nodes={node.name: node for node in network_nodes}, name="ExampleTransportation")
        # create logger
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)
        self.skip_noise = skip_noise

        # Initialize the controlled unitary matrix and measurement operators
        # CU_matrix = controlled_unitary(batch_size)
        measurement_m0, measurement_m1 = measure_operator()
        # CU_gate = ops.Operator("CU_Gate", CU_matrix)
        # CCU_gate = CU_gate.conj

        # initialize the protocol for each node
        for index, node in enumerate(network_nodes):
            # Initialize the MessageHandler protocol
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self.get_cc_ports(node)
                                                ))
            qubit_input_protocols = []
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
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                verify_protocol = Verification(node=node,
                                               name=f"verify_{node.name}->{network_nodes[index - 1].name}",
                                               entangled_node=network_nodes[index - 1].name,
                                               purification_protocol=pure_protocol,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               m_size=m_size,
                                               batch_size=batch_size,
                                               CU_Gate=CU_gate,
                                               CCU_Gate=CCU_gate,
                                               measurement_m0=measurement_m0,
                                               measurement_m1=measurement_m1,
                                               logger=null_logger,
                                               is_top_layer=False,
                                               max_verify_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)
                qubit_input_protocols.append(verify_protocol)

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
                # Initialize the purification protocol
                pure_protocol = Purification(node=node,
                                             name=f"purify_{node.name}->{network_nodes[index + 1].name}",
                                             entangled_node=network_nodes[index + 1].name,
                                             entanglement_handler=eh_handler,
                                             cc_message_handler=self.subprotocols[
                                                 f"message_handler_{node.name}"],
                                             max_purify_pair=self.max_entangle_pairs,
                                             target_fidelity=target_fidelity,
                                             is_top_layer=False,
                                             logger=null_logger
                                             )
                self.add_subprotocol(pure_protocol)
                verify_protocol = Verification(node=node,
                                               name=f"verify_{node.name}->{network_nodes[index + 1].name}",
                                               entangled_node=network_nodes[index + 1].name,
                                               purification_protocol=pure_protocol,
                                               cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                               m_size=m_size,
                                               batch_size=batch_size,
                                               CU_Gate=CU_gate,
                                               CCU_Gate=CCU_gate,
                                               measurement_m0=measurement_m0,
                                               measurement_m1=measurement_m1,
                                               logger=null_logger,
                                               is_top_layer=False,
                                               max_verify_pairs=self.max_entangle_pairs,
                                               )
                self.add_subprotocol(verify_protocol)
                qubit_input_protocols.append(verify_protocol)

            entangle_name = ""
            if index + 1 < len(network_nodes):
                entangle_name = network_nodes[index + 1].name
            else:
                entangle_name = network_nodes[index - 1].name
            # add security
            """
            def __init__(self, node,
                 name,
                 qubit_ready_protocols,
                 entangled_node,
                 source,
                 destination,
                 cc_message_handler,
                 logger,
                 transport_signal_protocol,
                 verification_signal_protocol,
                 is_top_layer=False):
            """
            security = Security(node=node,
                                name=f"security_{node.name}",
                                qubit_ready_protocols=qubit_input_protocols,
                                entangled_node=entangle_name,
                                source=network_nodes[0].name,
                                destination=network_nodes[-1].name,
                                cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                logger=self.logger,
                                transport_signal_protocol=None,
                                verification_signal_protocol=None)
            self.add_subprotocol(security)

            security_end_to_end = EndToEndProtocol(node=node,
                                                   name=f"security_e2e_{node.name}",
                                                   swapping_nodes=self.swap_nodes,
                                                   final_entanglement=self.final_entanglement,
                                                   cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                                   qubit_ready_protocols=[security],
                                                   max_pairs=self.max_entangle_pairs - 1,
                                                   logger=null_logger,
                                                   is_top_layer=False)
            self.add_subprotocol(security_end_to_end)

            if index == 0:
                security_verify_protocol = SecurityVerification(node=node,
                                                                name=f"security_verify_{node.name}->{network_nodes[-1].name}",
                                                                entangled_node=entangle_name,
                                                                target_node=network_nodes[-1].name,
                                                                purification_protocol=security_end_to_end,
                                                                cc_message_handler=self.subprotocols[
                                                                    f"message_handler_{node.name}"],
                                                                m_size=m_size,
                                                                batch_size=batch_size,
                                                                CU_Gate=CU_gate,
                                                                CCU_Gate=CCU_gate,
                                                                measurement_m0=measurement_m0,
                                                                measurement_m1=measurement_m1,
                                                                logger=self.logger,
                                                                is_top_layer=False,
                                                                max_entangled_pairs=self.max_entangle_pairs,
                                                                )
                security.verification_signal_protocol = security_verify_protocol
                security.transport_signal_protocol = self
                self.add_subprotocol(security_verify_protocol)
            elif index == len(network_nodes) - 1:
                security_verify_protocol = SecurityVerification(node=node,
                                                                name=f"security_verify_{node.name}->{network_nodes[0].name}",
                                                                entangled_node=entangle_name,
                                                                target_node=network_nodes[0].name,
                                                                purification_protocol=security_end_to_end,
                                                                cc_message_handler=self.subprotocols[
                                                                    f"message_handler_{node.name}"],
                                                                m_size=m_size,
                                                                batch_size=batch_size,
                                                                CU_Gate=CU_gate,
                                                                CCU_Gate=CCU_gate,
                                                                measurement_m0=measurement_m0,
                                                                measurement_m1=measurement_m1,
                                                                logger=self.logger,
                                                                is_top_layer=False,
                                                                max_entangled_pairs=self.max_entangle_pairs,
                                                                )
                self.add_subprotocol(security_verify_protocol)
                security.transport_signal_protocol = self.subprotocols[f"security_{network_nodes[0].name}"]
            else:
                security.transport_signal_protocol = self.subprotocols[f"security_{network_nodes[0].name}"]

            # add transport protocol
            transport = Transportation(node=node,
                                       name=f"transport_{node.name}",
                                       qubit_ready_protocols=[security],
                                       entangled_node=entangle_name,
                                       source=network_nodes[0].name,
                                       destination=network_nodes[-1].name,
                                       cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                                       transmitting_qubit_size=qubits_to_transport,
                                       logger=self.logger,
                                       is_top_layer=True,
                                       is_after_security=True,
                                       )
            self.add_subprotocol(transport)

    def run(self):
        self.start_subprotocols()
        # for subprotoco, val in self.subprotocols.items():
        #     print(f"Subprotocol: {subprotoco}")

        for index in range(self.num_runs):
            start_time = sim_time()

            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()
            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_FINISHED, self)
            """
            result = {entangle_node: name, mem_poses:[]}
            """
            result_dic = {"teleport_success_count": 0,
                          "total_count": 0,
                          "teleport_success_rate": 0,
                          "teleport_fids": [],
                          "duration": end_time - start_time, }
            for mem_pos, fid in results["results"].items():
                result_dic["total_count"] += 1
                # # get the qubit
                # qubit = self.all_nodes[-1].subcomponents[f"{results['entangle_node']}_qmemory"].pop(
                #     mem_pos, skip_noise=True)[0]
                # # measure the state
                # fidelity = qapi.fidelity(qubit, ns.y0)
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
                result_dic['teleport_fids'].append(fid)
            # final success rate
            # result_dic["teleport_success_rate"] = result_dic["teleport_success_count"] / result_dic["total_count"]
            # for subprotocol_name, subprotocol in self.subprotocols.items():
            #     if "purify" in subprotocol_name:
            #         subprotocol.cc_message_handler.send_signal(MessageType.VERIFICATION_FINISHED,
            #                                                    ProtocolFinishedSignalMessage(
            #                                                        from_protocol=subprotocol,
            #                                                        from_node=subprotocol.node.name,
            #                                                        entangle_node=subprotocol.entangled_node
            #                                                    ))

            self.send_signal(Signals.SUCCESS, {"results": result_dic,
                                               "run_index": index})
            break
            # p_done = False
            # print(f"Start Stop Purification of run index {index}")
            # start_end_time = sim_time()
            # while not p_done:
            #     yield self.await_timer(1000)
            #     all_done = True
            #     for subprotocol_name, subprotocol in self.subprotocols.items():
            #         if "purify" in subprotocol_name:
            #             if subprotocol.is_running:
            #                 all_done = False
            #     if all_done:
            #         p_done = True
            #     if sim_time() - start_end_time > 100000:
            #         break
            # # print(f"Finished Stop Purification of run index {index}")
            # for subprotocol in self.subprotocols.values():
            #     subprotocol.reset()
        # remove any gates after finish running
        # for subprotocol in self.subprotocols.values():
        #     if "verify" in subprotocol.name:
        #         subprotocol.clean_gates()

    def get_cc_ports(self, node):
        cc_ports = {}
        for n in self.all_nodes:
            if n != node:
                cc_ports[n.name] = node.get_conn_port(n.ID)
        return cc_ports

    def stop(self):
        for subprotocol in self.subprotocols.values():
            subprotocol.stop()


def example_sim_run_with_security(nodes, num_runs, memory_depolar_rate,
                                  node_distance, max_entangle_pairs, target_fidelity, m_size, batch_size,
                                  qubit_to_transport, CU_gate, CCU_gate,
                                  skip_noise=True):
    """
    Run the example security protocol
    :param nodes: list of nodes
    :param num_runs: number of runs
    :param memory_depolar_rate: memory depolar rate
    :param node_distance: node distance
    :param max_entangle_pairs: maximum entangle pairs
    :param target_fidelity: target fidelity
    :param m_size: m size
    :param batch_size: batch size
    :param qubit_to_transport: number of qubits to transmit
    :param skip_noise: skip noise when popping qubits
    :return:
    """
    # Create the protocol
    transport_example = SecurityWithVerificationExample(network_nodes=nodes,
                                                        num_runs=num_runs,
                                                        max_entangle_pairs=max_entangle_pairs,
                                                        memory_depolar_rate=memory_depolar_rate,
                                                        node_distance=node_distance,
                                                        target_fidelity=target_fidelity,
                                                        m_size=m_size,
                                                        batch_size=batch_size,
                                                        skip_noise=skip_noise,
                                                        qubits_to_transport=qubit_to_transport,
                                                        CU_gate=CU_gate,
                                                        CCU_gate=CCU_gate,
                                                        node_path=[node.name for node in nodes])

    # Run the protocol
    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Verification Run {result['run_index']} completed, fid{result['results']['teleport_fids']}, "
              f"sim_time {sim_time()}")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False,
                       include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=transport_example,
                                     event_type=Signals.SUCCESS.value))
    return transport_example, dc


def run_example(qubit_number=1, node_count=3, distance=1.0):
    CU_matrix = controlled_unitary(4)
    CU_gate = ops.Operator("CU_Gate", CU_matrix)
    CCU_gate = CU_gate.conj
    os.makedirs("security_results", exist_ok=True)
    node_data = {}
    run_count = 0
    if os.path.exists(f"./security_results/{node_count}nodes_{distance}km_security_raw.json"):
        with open(f"./security_results/{node_count}nodes_{distance}km_security_raw.json") as f:
            node_data = json.load(f)
            run_count = len(node_data["teleport_success_count"])
    while run_count < 1000:
        try:
            print(f"Run #{run_count}/1000")
            nodes_list = [f"Node_{i}" for i in range(3)]
            network = setup_network(nodes_list, "hop-by-hop-transportation",
                                    memory_capacity=2000, memory_depolar_rate=63109,
                                    node_distance=distance, source_delay=1)
            # create a protocol to entangle two nodes
            sample_nodes = [node for node in network.nodes.values()]
            security_example, dc = example_sim_run_with_security(sample_nodes, num_runs=1,
                                                                 memory_depolar_rate=63109,
                                                                 node_distance=distance,
                                                                 max_entangle_pairs=2000,
                                                                 target_fidelity=0.98,
                                                                 m_size=3,
                                                                 batch_size=4,
                                                                 skip_noise=True,
                                                                 qubit_to_transport=qubit_number,
                                                                 CU_gate=CU_gate,
                                                                 CCU_gate=CCU_gate, )

            # Run the simulation
            security_example.start()
            ns.sim_run()
            # Collect the data
            collected_data = dc.dataframe
            for c in collected_data.columns:
                if c not in node_data:
                    node_data[c] = []
                if c == "teleport_fids":
                    s = []
                    for t in collected_data[c]:
                        s += t
                    node_data[c].append(np.mean(s))
                else:
                    node_data[c].append(collected_data[c].mean())
                    # if c not in node_data:
                #     node_data[c] = []
                # node_data[c].append(collected_data[c].mean())
            with open(f"./security_results/{node_count}nodes_{distance}km_security_raw.json",
                      'w') as f:
                json.dump(node_data, f)
            final_result = {}
            for k, v in node_data.items():
                final_result[k] = np.mean(v)
                print(f"{node_count} Node ->{k}: {final_result[k]}")
            with open(f"./security_results/{node_count}nodes_{distance}km_security.json",
                      "w") as f:
                json.dump(final_result, f)
            security_example.stop()
            ns.set_random_state(rng=np.random.RandomState())
            print("Resetting network")
            ns.sim_reset()
            security_example = None
            gc.collect()
            run_count += 1
        except Exception as e:
            print(e)
            traceback.print_exc()
            security_example.stop()
            security_example = None
            ns.set_random_state(rng=np.random.RandomState())
            print("Resetting network")
            ns.sim_reset()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        opt = int(sys.argv[1])
        if opt == 1:
            run_example(qubit_number=1, node_count=3, distance=0.5)
        elif opt == 2:
            run_example(qubit_number=1, node_count=3, distance=1.0)
    else:
        print("Usage: python sim_security.py opt")
