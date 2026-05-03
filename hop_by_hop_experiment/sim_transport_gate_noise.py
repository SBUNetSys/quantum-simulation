"""
Hop-by-hop transport experiment with gate noise and delay.

Differences from sim_vbqt_transportation_concurrent.py:
  - Uses protocols/TransportGateNoise.py (per-hop correction) instead of MultiHopTransport.py
  - Gate operations run through a physical QuantumProcessor so duration and depolar noise are applied
  - No verification
  - Awaits TRANSPORT_FINISHED instead of MULTI_HOP_FINISHED
"""
import gc
import json
import os
import pathlib
import time
import traceback
from multiprocessing import Process, Queue

import numpy as np
import pydynaa as pd
import netsquid as ns
from netsquid.util.simtools import sim_time
from netsquid.util.datacollector import DataCollector
from netsquid.protocols.nodeprotocols import LocalProtocol
from netsquid.protocols.protocol import Signals
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NetworkSetup import setup_network_parallel_with_gate_noise
from utils import Logging
from protocols.MessageHandler import MessageHandler, MessageType
from protocols.EntanglementHandlerConcurrent import EntanglementHandlerConcurrent
from protocols.GenEntanglementConcurrent import GenEntanglementConcurrent
from protocols.Purification import Purification
from protocols.TransportGateNoise import Transportation
from utils.SignalMessages import ProtocolFinishedSignalMessage


class TransportGateNoiseExample(LocalProtocol):
    """
    Outer protocol for a single transport run with gate noise.
    Uses Transport.py-style per-hop correction; awaits TRANSPORT_FINISHED.
    """

    def __init__(self, network_nodes,
                 num_runs=1,
                 max_entangle_pairs=2,
                 memory_depolar_rate=1,
                 node_distance=20,
                 target_fidelity=0.99,
                 qubits_to_transport=1,
                 with_purify=True):
        if len(network_nodes) < 1:
            raise ValueError("Requires at least one node.")
        self.all_nodes = network_nodes
        self.num_runs = num_runs
        self.max_entangle_pairs = max_entangle_pairs
        self.qubits_to_transport = qubits_to_transport
        super().__init__(nodes={node.name: node for node in network_nodes},
                         name="TransportGateNoiseExample")
        self.logger = Logging.Logger(self.name, logging_enabled=False)
        null_logger = Logging.Logger("null", logging_enabled=False)

        for index, node in enumerate(network_nodes):
            self.add_subprotocol(MessageHandler(node=node,
                                                name=f"message_handler_{node.name}",
                                                cc_ports=self._cc_ports(node)))
            qubit_input_protocols = []

            if index - 1 >= 0:
                gen = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index - 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index - 1].name}",
                    is_source=False,
                    logger=null_logger)
                self.add_subprotocol(gen)
                eh = EntanglementHandlerConcurrent(
                    node=node,
                    name=f"entanglement_handler_{node.name}->{network_nodes[index - 1].name}",
                    num_pairs=self.max_entangle_pairs,
                    qubit_input_protocol=gen,
                    cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                    entangle_node=network_nodes[index - 1].name,
                    memory_depolar_rate=memory_depolar_rate,
                    node_distance=node_distance,
                    is_top_layer=False,
                    logger=null_logger)
                self.add_subprotocol(eh)
                gen.entanglement_handler = eh
                if with_purify:
                    pure = Purification(
                        node=node,
                        name=f"purify_{node.name}->{network_nodes[index - 1].name}",
                        entangled_node=network_nodes[index - 1].name,
                        entanglement_handler=eh,
                        cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                        max_purify_pair=self.max_entangle_pairs,
                        target_fidelity=target_fidelity,
                        is_top_layer=False,
                        logger=null_logger)
                    self.add_subprotocol(pure)
                    qubit_input_protocols.append(pure)
                else:
                    qubit_input_protocols.append(eh)

            if index + 1 < len(network_nodes):
                gen = GenEntanglementConcurrent(
                    total_pairs=self.max_entangle_pairs,
                    entangle_node=network_nodes[index + 1].name,
                    node=node,
                    name=f"entangle_{node.name}->{network_nodes[index + 1].name}",
                    is_source=True,
                    logger=null_logger)
                self.add_subprotocol(gen)
                eh = EntanglementHandlerConcurrent(
                    node=node,
                    name=f"entanglement_handler_{node.name}->{network_nodes[index + 1].name}",
                    num_pairs=self.max_entangle_pairs,
                    qubit_input_protocol=gen,
                    cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                    entangle_node=network_nodes[index + 1].name,
                    memory_depolar_rate=memory_depolar_rate,
                    node_distance=node_distance,
                    is_top_layer=False,
                    logger=null_logger)
                self.add_subprotocol(eh)
                gen.entanglement_handler = eh
                if with_purify:
                    pure = Purification(
                        node=node,
                        name=f"purify_{node.name}->{network_nodes[index + 1].name}",
                        entangled_node=network_nodes[index + 1].name,
                        entanglement_handler=eh,
                        cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                        max_purify_pair=self.max_entangle_pairs,
                        target_fidelity=target_fidelity,
                        is_top_layer=False,
                        logger=null_logger)
                    self.add_subprotocol(pure)
                    qubit_input_protocols.append(pure)
                else:
                    qubit_input_protocols.append(eh)

            entangle_name = (network_nodes[index + 1].name if index + 1 < len(network_nodes)
                             else network_nodes[index - 1].name)
            transport = Transportation(
                node=node,
                name=f"transport_{node.name}",
                qubit_ready_protocols=qubit_input_protocols,
                entangled_node=entangle_name,
                source=network_nodes[0].name,
                destination=network_nodes[-1].name,
                cc_message_handler=self.subprotocols[f"message_handler_{node.name}"],
                transmitting_qubit_size=qubits_to_transport,
                logger=self.logger,
                is_top_layer=True)
            self.add_subprotocol(transport)

    def run(self):
        self.start_subprotocols()
        for i in range(self.num_runs):
            start_time = sim_time()
            yield self.await_signal(self.subprotocols[f"transport_{self.all_nodes[-1].name}"],
                                    MessageType.TRANSPORT_FINISHED)
            end_time = sim_time()

            results = self.subprotocols[f"transport_{self.all_nodes[-1].name}"].get_signal_result(
                MessageType.TRANSPORT_FINISHED, self)
            result_dic = {
                "teleport_success_count": 0,
                "total_count": 0,
                "teleport_success_rate": 0,
                "teleport_fids": [],
                "duration": end_time - start_time,
            }
            for mem_pos, fid in results["results"].items():
                result_dic["total_count"] += 1
                if fid > 0.99:
                    result_dic["teleport_success_count"] += 1
                result_dic["teleport_fids"].append(fid)
            result_dic["teleport_success_rate"] = (
                result_dic["teleport_success_count"] / result_dic["total_count"])

            for subprotocol_name, subprotocol in self.subprotocols.items():
                if "purify" in subprotocol_name:
                    subprotocol.cc_message_handler.send_signal(
                        MessageType.VERIFICATION_FINISHED,
                        ProtocolFinishedSignalMessage(
                            from_protocol=subprotocol,
                            from_node=subprotocol.node.name,
                            entangle_node=subprotocol.entangled_node))

            self.send_signal(Signals.SUCCESS, {"results": result_dic, "run_index": i})

            p_done = False
            p_start = sim_time()
            while not p_done:
                yield self.await_timer(1000)
                p_done = all(
                    not sp.is_running
                    for name, sp in self.subprotocols.items()
                    if "purify" in name)
                if sim_time() - p_start > 10000:
                    break
            for subprotocol in self.subprotocols.values():
                subprotocol.reset()

    def _cc_ports(self, node):
        return {n.name: node.get_conn_port(n.ID) for n in self.all_nodes if n != node}

    def stop(self):
        for sp in self.subprotocols.values():
            sp.stop()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def example_sim_run(nodes, num_runs, memory_depolar_rate, node_distance,
                    max_entangle_pairs, target_fidelity, qubit_to_transport, with_purify):
    example = TransportGateNoiseExample(
        network_nodes=nodes,
        num_runs=num_runs,
        max_entangle_pairs=max_entangle_pairs,
        memory_depolar_rate=memory_depolar_rate,
        node_distance=node_distance,
        target_fidelity=target_fidelity,
        qubits_to_transport=qubit_to_transport,
        with_purify=with_purify)

    def record_run(evexpr):
        protocol = evexpr.triggered_events[-1].source
        result = protocol.get_signal_result(Signals.SUCCESS)
        print(f"Run {result['run_index']} done, "
              f"fids={result['results']['teleport_fids']}, "
              f"sim_time={sim_time():.0f} ns")
        return result["results"]

    dc = DataCollector(record_run, include_time_stamp=False, include_entity_name=False)
    dc.collect_on(pd.EventExpression(source=example, event_type=Signals.SUCCESS.value))
    return example, dc


def run_gate_noise_sim_worker(queue, distance, target_fid, depolar_rate, node_count,
                              with_purify, run_index, data_save_path, raw_data_save_path,
                              gate_depolar_rate, gate_duration_ns, cnot_depolar_rate, cnot_duration_ns):
    try:
        node_data = {}
        node_data_raw = {}
        nodes_list = [f"Node_{i}" for i in range(node_count)]
        network = setup_network_parallel_with_gate_noise(
            nodes_list, "hop-by-hop-gate-noise",
            memory_capacity=101,
            memory_depolar_rate=depolar_rate,
            node_distance=distance,
            gate_depolar_rate=gate_depolar_rate,
            gate_duration_ns=gate_duration_ns,
            cnot_depolar_rate=cnot_depolar_rate,
            cnot_duration_ns=cnot_duration_ns)

        sample_nodes = list(network.nodes.values())
        transport_example, dc = example_sim_run(
            sample_nodes,
            num_runs=1,
            memory_depolar_rate=depolar_rate,
            node_distance=distance,
            max_entangle_pairs=100,
            target_fidelity=target_fid,
            qubit_to_transport=1,
            with_purify=with_purify)

        transport_example.start()
        ns.sim_run()

        collected_data = dc.dataframe
        for c in collected_data.columns:
            if c not in node_data:
                node_data[c] = []
                node_data_raw[c] = []
            if c == "teleport_fids":
                s = []
                for t in collected_data[c]:
                    s += t
                node_data_raw[c] = s
                node_data[c].append(np.mean(s))
                node_data["teleport_fids_all"] = s
            else:
                node_data[c].append(collected_data[c].mean())

        transport_example.stop()
        ns.sim_stop()
        ns.sim_reset()
        transport_example = None
        gc.collect()
        queue.put(('success', run_index, node_data, node_data_raw))

    except Exception as e:
        queue.put(('error', run_index, str(e), traceback.format_exc()))


def run_gate_noise_sim_multiprocess(distance, target_fid, depolar_rate, node_count,
                                    with_purify, preload=False, total_runs=1000,
                                    save_path="./gate_noise_results/",
                                    gate_depolar_rate=20000, gate_duration_ns=50,
                                    cnot_depolar_rate=33000, cnot_duration_ns=300):
    os.makedirs(save_path, exist_ok=True)
    data_save_path = os.path.join(
        save_path,
        f"gate_noise_transport_{node_count}_nodes_{distance}km@{depolar_rate}hz_"
        f"purify_{with_purify}_gd{gate_depolar_rate}_cd{cnot_depolar_rate}_qubit.json")
    raw_data_save_path = os.path.join(
        save_path,
        f"gate_noise_transport_{node_count}_nodes_{distance}km@{depolar_rate}hz_"
        f"purify_{with_purify}_gd{gate_depolar_rate}_cd{cnot_depolar_rate}_qubit_raw.json")

    all_result = {}
    all_result_raw = {}
    if preload and os.path.exists(data_save_path) and os.path.exists(raw_data_save_path):
        try:
            with open(data_save_path) as f:
                all_result = json.load(f)
            with open(raw_data_save_path) as f:
                all_result_raw = json.load(f)
        except Exception as e:
            print(f"Error loading existing data: {e}")

    success_run = len(all_result) if len(all_result) == len(all_result_raw) else 0

    while success_run < total_runs:
        print(f"Run {success_run + 1} / {total_runs}")
        queue = Queue()
        process = Process(
            target=run_gate_noise_sim_worker,
            args=(queue, distance, target_fid, depolar_rate, node_count,
                  with_purify, success_run + 1, data_save_path, raw_data_save_path,
                  gate_depolar_rate, gate_duration_ns, cnot_depolar_rate, cnot_duration_ns))
        process.start()
        try:
            process.join(timeout=300)
            ns.sim_stop()
            ns.sim_reset()
            ns.set_random_state(rng=np.random.RandomState())
            if process.is_alive():
                print("Process did not terminate, killing it")
                process.terminate()
                process.join()
            if process.exitcode != 0:
                print(f"Process killed by signal {-process.exitcode}")
                continue
            result = queue.get(timeout=300)
            if result[0] == 'success':
                _, run_index, node_data, node_data_raw = result
                success_run += 1
                all_result[str(success_run)] = node_data
                all_result_raw[str(success_run)] = node_data_raw
                with open(data_save_path, 'w') as f:
                    json.dump(all_result, f, indent=4, cls=NumpyEncoder)
                with open(raw_data_save_path, 'w') as f:
                    json.dump(all_result_raw, f, indent=4, cls=NumpyEncoder)
                print(f"Completed run {success_run}")
            elif result[0] == 'error':
                _, run_index, error_msg, stack_trace = result
                print(f"Error in run {run_index}: {error_msg}")
                print(stack_trace)
        except Exception as e:
            print(f"Process failed or timed out: {e}")
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
        if not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                pass
        time.sleep(1)

    print(f"Completed all {total_runs} runs")
    return all_result, all_result_raw


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hop-by-hop transport with gate noise')
    parser.add_argument('--distance', type=float, default=1.0, help='Node distance in km')
    parser.add_argument('--target-fid', type=float, default=0.99, help='Target fidelity for purification')
    parser.add_argument('--depolar-rate', type=int, default=6000, help='Memory depolar rate (Hz)')
    parser.add_argument('--node-count', type=int, default=3, help='Number of nodes')
    parser.add_argument('--with-purify', action='store_true', help='Enable purification')
    parser.add_argument('--preload', action='store_true', help='Resume from existing results file')
    parser.add_argument('--total-runs', type=int, default=1000, help='Total simulation runs')
    parser.add_argument('--save-path', type=str, default='./gate_noise_results/', help='Output directory')
    parser.add_argument('--gate-depolar-rate', type=int, default=20000,
                        help='Single-qubit gate depolar rate (Hz); default ~0.1%% error at 50 ns')
    parser.add_argument('--gate-duration-ns', type=int, default=50,
                        help='Single-qubit gate duration (ns)')
    parser.add_argument('--cnot-depolar-rate', type=int, default=33000,
                        help='CNOT gate depolar rate (Hz); default ~1%% error at 300 ns')
    parser.add_argument('--cnot-duration-ns', type=int, default=300,
                        help='CNOT gate duration (ns)')
    args = parser.parse_args()

    print(
        f"Running gate noise transport experiment:"
        f"\n\tdistance={args.distance} km"
        f"\n\ttarget_fid={args.target_fid}"
        f"\n\tdepolar_rate={args.depolar_rate} Hz"
        f"\n\tnode_count={args.node_count}"
        f"\n\twith_purify={args.with_purify}"
        f"\n\tgate_depolar_rate={args.gate_depolar_rate} Hz"
        f"\n\tgate_duration_ns={args.gate_duration_ns} ns"
        f"\n\tcnot_depolar_rate={args.cnot_depolar_rate} Hz"
        f"\n\tcnot_duration_ns={args.cnot_duration_ns} ns"
        f"\n\ttotal_runs={args.total_runs}"
        f"\n\tpreload={args.preload}"
        f"\n\tsave_path={args.save_path}"
    )

    run_gate_noise_sim_multiprocess(
        distance=args.distance,
        target_fid=args.target_fid,
        depolar_rate=args.depolar_rate,
        node_count=args.node_count,
        with_purify=args.with_purify,
        preload=args.preload,
        total_runs=args.total_runs,
        save_path=args.save_path,
        gate_depolar_rate=args.gate_depolar_rate,
        gate_duration_ns=args.gate_duration_ns,
        cnot_depolar_rate=args.cnot_depolar_rate,
        cnot_duration_ns=args.cnot_duration_ns,
    )
