import copy
import operator
from functools import reduce

import numpy as np
from netsquid.components.qprocessor import sim_time
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.protocols.protocol import Signals
import netsquid as ns

from protocols.MessageHandler import MessageType
from utils import Logging, SignalMessages
from utils.ClassicalMessages import ClassicalMessage


class EntanglementHandlerConcurrent(NodeProtocol):
    """
    Protocol to manage entanglement for with a remote node.
    The protocol keeps track of the number of entangled pairs and the memory position for the qubit
    """

    def __init__(self, node, name, num_pairs, entangle_node,
                 node_distance, memory_depolar_rate,
                 qubit_input_protocol, cc_message_handler,
                 logger=None, is_top_layer=False):
        """
        Initialize the protocol.

        Parameters
        ----------
        node : :class:`~netsquid.nodes.node.Node`
            The node that the protocol is attached to.
        num_pairs : int
            The max number of qubits that can be entangled i.e qmemory size.
        entangle_node : str
            The name of the node that the protocol will entangle with.
        node_distance : float
            The distance between the nodes in km.
        memory_depolar_rate : float
            The depolarization rate of the qubits in memory. (Hz)
        qubit_input_protocol : GenEntanglement
            The lower layer protocol (GenEntanglement) for qubit generation success signal.
        cc_message_handler : MessageHandler
            The message handler for classical communication.
        logger : Logging.Logger
            The logger to log messages.
        is_top_layer : bool
            This flag is used to indicate if the protocol is the top layer protocol. If True, the protocol will
            stop the simulation when the entanglement is complete. Otherwise, it will run forever and waiting for
            re-entangle process.
        """
        if entangle_node is None:
            raise ValueError("entangle_node must be specified.")

        super().__init__(node=node, name=name)
        # since we will use on memory position for entanglement operation, therefore we need to subtract 1 for each node
        # in case of multiple entangle_node, we need to multiply by the number of entangle nodes
        self.max_pairs = num_pairs
        # store the entangle nodes
        self.entangle_node = entangle_node
        # mapping of entangled qubits to memory positions key: memory position, value: fidelity
        self.entangled_qubits = {}
        # mapping of temporary qubits to memory positions
        self.temp_qubits = {}  # {mempos: (fid, time)}
        # entangle_message_queue
        self.entangle_message_queue = []
        # re-entangle message queue
        self.re_entangle_message_queue = []
        # deal with remote ready messages
        self.re_entangle_flush_time = None
        # store the depolar rate and node distance
        self.depolar_rate = memory_depolar_rate
        self.node_distance = node_distance
        # time out for qubit lost
        self.timeout_time = 2.2 * ((self.node_distance / 200e3) * 1e9)
        self.qubit_lost_count = 0
        # store the qubit input protocol
        self.qubit_input_protocol = qubit_input_protocol
        # have expression to wait for the qubit input signal
        self.qubit_input_signal = (self.await_signal(qubit_input_protocol,
                                                     signal_label=MessageType.GEN_ENTANGLE_SUCCESS) |
                                   self.await_signal(qubit_input_protocol,
                                                     signal_label=MessageType.ENTANGLED_QUBIT_LOST))
        # classical message handler
        self.cc_message_handler = cc_message_handler

        # set the logger
        if logger is None:
            self.logger = Logging.Logger(f"{self.name}_logger", logging_enabled=True)
        else:
            self.logger = logger

        # add signals for sending
        self.add_new_signal(f"{qubit_input_protocol.name}_re_entangle_ready")
        self.add_new_signal(f"{qubit_input_protocol.name}_re_entangle")

        self.add_new_signal(MessageType.RE_ENTANGLE_QUBIT_LOST)
        self.add_new_signal(MessageType.ENTANGLED)
        self.add_new_signal(MessageType.RE_ENTANGLE_CONCURRENT)
        self.add_new_signal(MessageType.ENTANGLEMENT_HANDLER_FINISHED)
        self.add_new_signal(MessageType.ENTANGLED_SUCCESS)

        self.shutdown = False
        self.processed_re_entangled = False
        self.is_top_layer = is_top_layer
        # if self.is_top_layer:
        #     self.add_new_signal(MessageType.PROTOCOL_FINISHED)

    def add_new_signal(self, signal_label):
        """
        Add a new signal to the protocol. This allows the protocol to emit the signal with the signal label.
        :param signal_label:
        :return:
        """
        self.add_signal(signal_label)

    def process_entangle_message(self, message: ClassicalMessage):
        """
        Process the entangle message if the qubit is ready. Otherwise, store the message back in the queue.]

        """
        if self.shutdown:
            # we don't need to process the message if the protocol is going to shutdown
            return

        from_node = message.from_node
        to_node = message.to_node
        # get the entangle data from the message
        entangle_data = message.data
        # get the mem_pos from the entangle data
        mem_pos = entangle_data.mem_pos
        self.logger.info(f"ManageEntangle {self.name} -> Entanglement signal from {from_node} to {to_node}\n"
                         f"mem_pos: {mem_pos}\n"
                         f"time: {sim_time()}\n"
                         f"message time: {entangle_data.timestamp}\n"
                         f"temp qubits: {self.temp_qubits}", color="blue")
        entangled_poses = []
        entangled_fids = []
        if type(mem_pos) is list:
            non_entangle_poses = []
            for pos in mem_pos:
                if pos in self.temp_qubits:
                    # add the qubit to the entangled qubits
                    fid, _ = self.temp_qubits[pos]
                    # self.entangled_qubits[mem_pos] = self.temp_qubits[mem_pos]
                    self.entangled_qubits[pos] = fid
                    # remove the temporary qubit
                    del self.temp_qubits[pos]
                    entangled_poses.append(pos)
                    entangled_fids.append(fid)
                    self.logger.info(f"ManageEntangle {self.name} -> Entanglement Successful\n"
                                     f"\tFrom: {from_node}\n"
                                     f"\tMem_pos: {pos}\n"
                                     f"\tTemp Qubits: {self.temp_qubits}\n"
                                     f"\tEntangled_pairs_count: {len(self.entangled_qubits)}\n"
                                     f"\tExpected pairs: {self.max_pairs}"
                                     f"\tProgress: {len(self.entangled_qubits) / self.max_pairs}", color="green")

                else:
                    non_entangle_poses.append(pos)
            if len(non_entangle_poses) > 0:
                message.data.mem_pos = non_entangle_poses
                self.entangle_message_queue.append(message)
                self.logger.info(
                    f"ManageEntangle {self.name} -> Remote Entangle signal from {from_node} arrived early\n"
                    f"\tmem_pos: {mem_pos}\n",
                    color="Red")
        else:
            if mem_pos in self.temp_qubits:
                # add the qubit to the entangled qubits
                fid, _ = self.temp_qubits[mem_pos]
                # self.entangled_qubits[mem_pos] = self.temp_qubits[mem_pos]
                entangled_poses.append(mem_pos)
                entangled_fids.append(fid)
                self.entangled_qubits[mem_pos] = fid
                # remove the temporary qubit
                del self.temp_qubits[mem_pos]
                self.logger.info(f"ManageEntangle {self.name} -> Entanglement Successful\n"
                                 f"\tFrom: {from_node}\n"
                                 f"\tMem_pos: {mem_pos}\n"
                                 f"\tTemp Qubits: {self.temp_qubits}\n"
                                 f"\tEntangled_pairs_count: {len(self.entangled_qubits)}\n"
                                 f"\tExpected pairs: {self.max_pairs}"
                                 f"\tProgress: {len(self.entangled_qubits) / self.max_pairs}", color="green")
            else:
                self.entangle_message_queue.append(message)
                self.logger.info(
                    f"ManageEntangle {self.name} -> Remote Entangle signal from {from_node} arrived early\n"
                    f"\tmem_pos: {mem_pos}\n",
                    color="Red")

        # send the entangle pair to the upper layer
        if len(entangled_poses) > 0:
            self.send_signal(MessageType.ENTANGLED_SUCCESS,
                             SignalMessages.EntangleSuccessSignalMessage(
                                 self.node.name,
                                 from_node,
                                 entangled_poses,
                                 entangled_fids,
                                 is_source=True))

    def process_re_entangle_message(self, message: SignalMessages.ReEntangleSignalMessage):
        """
        Process the re-entangle message from upper layer. 
        The source node will wait re-entangle ready signal from the destination node.
        Then source node will clear up the memeory and generate new qubit to send to the destination node.

        """
        # get the entangle data from the message
        entangle_node = message.entangle_node
        mem_poses = message.re_entangle_mem_poses
        # remove the qubits from the entangled qubits
        for mem_pos in mem_poses:
            if mem_pos in self.entangled_qubits:
                del self.entangled_qubits[mem_pos]
        self.qubit_lost_count = 0
        self.logger.info(f"ManageEntangle {self.name} -> Re-entangle signal, entangle_node: {entangle_node},"
                         f" mem_pos: {mem_poses}", color="yellow")
        self.send_signal(MessageType.RE_ENTANGLE_CONCURRENT, message)

    def process_message_queue(self):

        temp = self.entangle_message_queue
        self.entangle_message_queue = []
        for message in temp:
            self.process_entangle_message(message)

    def process_re_entangle_message_queue(self):
        # if len(self.entangled_qubits.keys()) + self.qubit_lost_count == self.max_pairs:
        if self.re_entangle_flush_time is None or ns.sim_time() - self.re_entangle_flush_time >= 2000:
            if len(self.re_entangle_message_queue) == 0:
                return
            self.re_entangle_flush_time = ns.sim_time()
            temp = self.re_entangle_message_queue
            self.re_entangle_message_queue = []
            ready_mem_poses = []
            for message in temp:
                ready_mem_poses += message.re_entangle_mem_poses
            self.logger.info(f"ManageEntangle {self.name} -> Processed Batch Memory Ready\n"
                             f"\tmem_pos: {ready_mem_poses}", color="purple")
            batch_message = SignalMessages.ReEntangleSignalMessage(self.entangle_node, ready_mem_poses)
            self.process_re_entangle_message(batch_message)

    def check_temp_entanglement(self):
        """
        this function check all temp qubit established time
        if is over 5000 ns, we consider it was lost and need re-entangle
        :return
        """
        re_entangle_poses = []
        current_time = sim_time()
        pos = list(self.temp_qubits.keys())
        for p in pos:
            _, t = self.temp_qubits[p]
            if current_time - t > self.timeout_time:
                self.logger.info(f"ManageEntangle {self.name} -> Time out for entanglement establishment\n"
                                 f"\tRe-entangle Qubits Mem: {p}\n"
                                 f"\tCurrent Time: {current_time}\n"
                                 f"\tEntangle Time: {t}\n", color="red")
                re_entangle_poses.append(p)
                self.temp_qubits.pop(p)
        # send to lower layer as the remote is ready for re-entangle.
        # we can do this because remote never received this qubit, therefore it is ready by default
        if len(re_entangle_poses) > 0:
            self.logger.info(f"ManageEntangle {self.name} -> Time out for entanglement establishment\n"
                             f"\tRe-entangle Qubits Mem: {re_entangle_poses}\n"
                             f"\tTemp Qubits: {self.temp_qubits.keys()}\n"
                             f"\tEntangled Qubits: {self.entangled_qubits.keys()}\n", color="red")
            re_entangle_data = SignalMessages.ReEntangleSignalMessage(
                self.entangle_node, re_entangle_poses, re_entangle_type="timeout")
            self.send_signal(MessageType.RE_ENTANGLE_QUBIT_LOST,
                             re_entangle_data)
            # # send the entangled signal to lower layer try to re-activate the generation process
            # self.send_signal(MessageType.ENTANGLED, None)

    def estimate_fidelity_theoretical(self, initial_fidelity):
        """Estimate fidelity based on noise parameters and channel length."""
        # depolar_rate = noise_params['depolar_rate']
        # dephase_rate = noise_params['dephase_rate']

        # Depolarizing effect
        time_spend = self.node_distance / 200e3
        # p_depolar = 1 - np.exp(-depolar_rate * channel_length)
        # f_depolar = (1 - p_depolar) + (p_depolar / 4)
        # p_depolar = 1 - np.exp(-(self.depolar_rate) * time_spend)
        p_depolar = 1 - np.exp(-8641 * time_spend)
        f_depolar = (1 - p_depolar) + (p_depolar / 4)

        # final_fidelity = initial_fidelity * (0.25 + 0.75 * np.exp(-self.depolar_rate * time_spend))

        # # Dephasing effect
        # p_dephase = 1 - np.exp(-dephase_rate * channel_length)
        # f_dephase = 1 - p_dephase / 2

        # Combine effects (assuming independent noise processes)
        final_fidelity = initial_fidelity * f_depolar

        return final_fidelity

    def run(self):
        self.logger.info(f"ManageEntangle {self.name} -> Started\n")
        entangle_signals = (self.await_signal(self.cc_message_handler, signal_label=MessageType.ENTANGLED) |
                            self.await_signal(self.cc_message_handler, signal_label=MessageType.ENTANGLED_QUBIT_LOST) |
                            self.await_signal(self.cc_message_handler,
                                              signal_label=MessageType.RE_ENTANGLE_FROM_UPPER_LAYER))
        yield self.await_timer(1)
        start_time = sim_time()
        while True:
            # wait for entanglement
            self.processed_re_entangled = False
            expr = yield self.qubit_input_signal | entangle_signals
            if expr.first_term.value:
                # case we have qubit input signal
                for event in expr.first_term.triggered_events:
                    source_protocol = event.source
                    ready_signal = source_protocol.get_signal_by_event(
                        event=event, receiver=self)
                    result = ready_signal.result
                    gen_data: SignalMessages.NewEntanglementSignalMessage = result
                    if gen_data.source_node != self.node.name:
                        continue
                    if gen_data.timestamp < start_time:
                        continue
                    mem_pos = gen_data.mem_pos
                    is_source = gen_data.is_source
                    qmemory_name = gen_data.qmemory_name
                    entangle_node = gen_data.entangle_node
                    initial_fidelity = gen_data.init_fidelity
                    if is_source:
                        # store the qubit in the temporary qubits
                        if type(mem_pos) is list:
                            for pos in mem_pos:
                                self.temp_qubits[pos] = (
                                    self.estimate_fidelity_theoretical(initial_fidelity), sim_time())
                                self.logger.info(
                                    f"ManageEntangle {self.name} -> Entangle signal from QSource"
                                    f"\tmem_pos: {mem_pos}\n"
                                    f"\tInitial Fidelity: {initial_fidelity}\n "
                                    f"\tEstimated Fidelity: {self.temp_qubits[pos]}\n"
                                    f"\tTime {sim_time()}", color="blue")
                        else:
                            self.temp_qubits[mem_pos] = (
                                self.estimate_fidelity_theoretical(initial_fidelity), sim_time())
                            self.logger.info(
                                f"ManageEntangle {self.name} -> Entangle signal from QSource"
                                f"\tmem_pos: {mem_pos}\n"
                                f"\tInitial Fidelity: {initial_fidelity}\n "
                                f"\tEstimated Fidelity: {self.temp_qubits[mem_pos]}\n"
                                f"\tTime {sim_time()}", color="blue")

                    else:
                        if ready_signal.label == MessageType.ENTANGLED_QUBIT_LOST:
                            self.logger.info(
                                f"ManageEntangle {self.name} -> Entangle signal from Remote node\n"
                                f"Qubit lost"
                                f"from {entangle_node}\n"
                                f"mem_pos: {mem_pos}\n"
                                f"time {sim_time()}",
                                color="yellow")
                            # send the entangled signal to the source node
                            self.cc_message_handler.send_message(MessageType.ENTANGLED_QUBIT_LOST, entangle_node,
                                                                 ClassicalMessage(
                                                                     from_node=self.node.name, to_node=entangle_node,
                                                                     data=SignalMessages.ReEntangleSignalMessage(
                                                                         self.node.name,
                                                                         mem_pos,
                                                                         is_source=is_source, )
                                                                 ))
                        else:
                            # we don't need to estimate the fidelity for the remote node
                            # as we don't know the initial fidelity. Here it will be None
                            # we assume is 1 from source node
                            cal_fid = self.estimate_fidelity_theoretical(1.0)
                            entangled_poses = []
                            entangled_fids = []
                            if type(mem_pos) is list:
                                for pos in mem_pos:
                                    # add the qubit to the entangled qubits
                                    self.logger.info(
                                        f"ManageEntangle {self.name} -> Entangle signal from Remote node\n"
                                        f"from {entangle_node}\n"
                                        f"mem_pos: {pos}\n"
                                        f"time {sim_time()}",
                                        color="yellow")
                                    self.entangled_qubits[pos] = cal_fid
                                    entangled_poses.append(pos)
                                    entangled_fids.append(cal_fid)
                            else:
                                self.logger.info(
                                    f"ManageEntangle {self.name} -> Entangle signal from Remote node\n"
                                    f"from {entangle_node}\n"
                                    f"mem_pos: {mem_pos}\n"
                                    f"time {sim_time()}",
                                    color="yellow")
                                self.entangled_qubits[mem_pos] = cal_fid
                                entangled_fids.append(cal_fid)
                                entangled_poses.append(mem_pos)
                            # send the entangled signal to the source node
                            self.cc_message_handler.send_message(MessageType.ENTANGLED, entangle_node,
                                                                 ClassicalMessage(
                                                                     from_node=self.node.name, to_node=entangle_node,
                                                                     data=SignalMessages.EntangleSignalMessage(
                                                                         self.node.name,
                                                                         self.node.name,
                                                                         entangled_poses)
                                                                 ))
                            # send the entangle pair to the upper layer
                            self.send_signal(MessageType.ENTANGLED_SUCCESS,
                                             SignalMessages.EntangleSuccessSignalMessage(
                                                 self.node.name,
                                                 entangle_node,
                                                 entangled_poses,
                                                 entangled_fids,
                                                 is_source=False
                                             ))

            elif expr.second_term.value:
                # case we have entanglement signal
                for event in expr.second_term.triggered_events:
                    source_protocol = event.source
                    try:
                        ready_signal = source_protocol.get_signal_by_event(
                            event=event, receiver=self)
                    except Exception as e:
                        self.logger.error(f"Error: {e}")
                        continue
                    result = ready_signal.result
                    if isinstance(result, ClassicalMessage):
                        if result.from_node != self.entangle_node:
                            # we don't process the message that is not from the entangle node
                            continue
                        if result.data.timestamp < start_time:
                            continue
                    if isinstance(result, SignalMessages.ReEntangleSignalMessage):
                        if result.entangle_node != self.entangle_node:
                            # we don't process the message that is not for the current node
                            continue
                        if result.timestamp < start_time:
                            continue
                    if ready_signal.label == MessageType.ENTANGLED:
                        result: ClassicalMessage
                        self.process_entangle_message(result)
                        # self.process_re_entangle_message(result)
                    elif ready_signal.label == MessageType.ENTANGLED_QUBIT_LOST:
                        result: ClassicalMessage
                        self.qubit_lost_count += len(result.data.re_entangle_mem_poses)
                        self.re_entangle_message_queue.append(result.data)
                    elif ready_signal.label == MessageType.RE_ENTANGLE_FROM_UPPER_LAYER:
                        # re-entangle from upper layer
                        if result.entangle_node != self.entangle_node:
                            continue
                        if not result.is_source:
                            continue
                        self.re_entangle_message_queue.append(result)
                        self.logger.info(f"ManageEntangle {self.name} -> Re-entangle signal from upper layer\n"
                                         f"\tentangle_node: {result.entangle_node}\n"
                                         f"\tmem_pos: {result.re_entangle_mem_poses}\n"
                                         f"\tcurrent_entangled pairs: {self.entangled_qubits}\n"
                                         f"\tcurrent_entangled count: {len(self.entangled_qubits)}\n"
                                         f"\tqubit_lost_count: {self.qubit_lost_count}\n",
                                         color="purple")

            self.process_message_queue()
            self.process_re_entangle_message_queue()

            if self.shutdown:
                # by default the graceful shutdown will not continue generation of qubits
                self.send_signal(MessageType.ENTANGLEMENT_HANDLER_FINISHED, self.entangled_qubits)
                break

            if self.is_top_layer:
                if len(self.entangled_qubits) >= self.max_pairs:
                    # send finish signal to the source node
                    self.logger.info(f"ManageEntangle {self.name} -> Entanglement complete", color="green")
                    self.send_signal(MessageType.ENTANGLEMENT_HANDLER_FINISHED, self.entangled_qubits)
                    break

    def reset(self):
        # mapping of entangled qubits to memory positions key: node name, value: {memory position, fidelity}
        self.entangled_qubits = {}
        # mapping of temporary qubits to memory positions
        self.temp_qubits = {}
        # keep track of the number of entangled pairs
        # entangle_message_queue
        self.entangle_message_queue = []
        # reset the shutdown flag
        self.re_entangle_message_queue = []
        self.re_entangle_flush_time = None
        self.qubit_lost_count = 0
        # reset the shutdown flag
        self.shutdown = False
        super().reset()

    def stop(self):
        super().stop()

    @property
    def is_connected(self):
        if not super().is_connected:
            return False
        # check all entangled qmemory is connected
        try:
            memory = self.node.subcomponents[f"{self.entangle_node}_qmemory"]
        except KeyError:
            print(f"Memory {self.entangle_node}_qmemory not found")
            return False
        return True
