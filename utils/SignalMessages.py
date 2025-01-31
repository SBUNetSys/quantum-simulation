import netsquid as ns


class EntangleSignalMessage:
    """
    Base class for entanglement signal messages
    Most used for re-entanglement
    """

    def __init__(self, source_node, entangle_node, mem_pos):
        self.source_node = source_node
        self.entangle_node = entangle_node
        self.mem_pos = mem_pos
        self.timestamp = ns.sim_time()


class NewEntanglementSignalMessage(EntangleSignalMessage):
    """
    Signal message for new entanglement
    Most used for entanglement creation
    """

    def __init__(self, source_node, entangle_node, mem_pos, qmemory_name, is_source, init_fidelity):
        super().__init__(source_node, entangle_node, mem_pos)
        self.qmemory_name = qmemory_name
        self.is_source = is_source
        self.init_fidelity = init_fidelity


class EntangleSuccessSignalMessage(EntangleSignalMessage):
    """
    Signal message for successful entanglement
    Most used for EntanglementHandler to notify success
    """

    def __init__(self, source_node, entangle_node, mem_pos, fidelity):
        super().__init__(source_node, entangle_node, mem_pos)
        self.fidelity = fidelity


class PurifySignalMessage:
    """
    Base class for purification signal messages
    """

    def __init__(self, entangle_node, qubit1_pos, qubit2_pos):
        self.entangle_node = entangle_node
        self.qubit1_pos = qubit1_pos
        self.qubit2_pos = qubit2_pos
        self.timestamp = ns.sim_time()


class PurifyStartSignalMessage(PurifySignalMessage):
    """
    Signal message for starting purification
    """

    def __init__(self, entangle_node, qubit1_pos, qubit2_pos, m1):
        super().__init__(entangle_node, qubit1_pos, qubit2_pos)
        self.m1 = m1


class PurifyResultSignalMessage(PurifySignalMessage):
    """
    Signal message for requesting purification
    """

    def __init__(self, entangle_node, qubit1_pos, qubit2_pos, m2, result):
        super().__init__(entangle_node, qubit1_pos, qubit2_pos)
        self.m2 = m2
        self.result = result


class PurifyTargetMetSignalMessage(EntangleSignalMessage):
    """
    Signal message for successful purification and the new fidelity met the target fidelity
    """

    def __init__(self, source_node, entangle_node, mem_pos, new_fidelity):
        super().__init__(source_node, entangle_node, mem_pos)
        self.fidelity = new_fidelity


class PurifySuccessSignalMessage(EntangleSignalMessage):
    """
    Signal message for successful purification and sent to upper layer
    """

    def __init__(self, source_node, entangle_node, mem_pos, new_fidelity, is_source):
        super().__init__(source_node, entangle_node, mem_pos)
        self.fidelity = new_fidelity
        self.is_source = is_source


class PurifyFinishedSignalMessage:
    """
    Signal message for successful purification and stop the protocol
    """

    def __init__(self, entangle_node):
        self.entangle_node = entangle_node
        self.timestamp = ns.sim_time()


class ReEntangleSignalMessage:
    """
    Signal message for re-entanglement, now we support list of re-entangle memory positions to avoid
    race condition
    : param entangle_node: entangle node name
    : param re_entangle_mem_poses: list of memory positions to re-entangle
    """

    def __init__(self, entangle_node, re_entangle_mem_poses: list, re_entangle_type="upper"):
        self.entangle_node = entangle_node
        self.re_entangle_mem_poses = re_entangle_mem_poses
        self.re_entangle_type = re_entangle_type
        self.timestamp = ns.sim_time()


class ProtocolFinishedSignalMessage:
    """
    Signal message for protocol finished
    """

    def __init__(self, from_protocol, from_node, entangle_node):
        self.from_protocol = from_protocol
        self.from_node = from_node
        self.entangle_node = entangle_node
        self.timestamp = ns.sim_time()


class VerificationSignalMessage:
    """
    Base class for verification signal messages
    """

    def __init__(self, entangle_node, verification_batch_id, verification_batch_poses: list):
        self.entangle_node = entangle_node
        self.verif_batch_id = verification_batch_id
        self.verif_batch_poses = verification_batch_poses
        self.timestamp = ns.sim_time()


class VerificationStartSignalMessage(VerificationSignalMessage):
    """
    Signal message for starting verification
    """

    def __init__(self, entangle_node, verification_batch_id, verification_batch_poses: list,
                 verification_teleport_measurement: dict):
        super().__init__(entangle_node, verification_batch_id, verification_batch_poses)
        self.verif_teleport_measurement = verification_teleport_measurement


class VerificationResultSignalMessage(VerificationSignalMessage):
    """
    Signal message for verification result
    """

    def __init__(self, entangle_node, verification_batch_id, verification_batch_poses: list,
                 verification_result: int, result_probability: float):
        super().__init__(entangle_node, verification_batch_id, verification_batch_poses)
        self.verif_result = verification_result
        self.result_probability = result_probability


class VerificationSuccessSignalMessage(EntangleSignalMessage):
    """
    Verification success signal message for send to upper layer. It is a subclass of EntangleSignalMessage

    """

    def __init__(self, source_node, entangle_node, is_source, verification_batch_poses: list):
        super().__init__(source_node, entangle_node, verification_batch_poses)
        self.is_source = is_source


class SwapRequestResponseMessage:
    """
    Signal message for end to end swap request

    :param source_node: source node name
    :param target_node: entangle node name
    :param intermediate_node: intermediate node name who will perform the swap
    :param memo_pos: memory position
    :param operation_key: operation key is used to identify the swap operation
    """

    def __init__(self, source_node, target_node, intermediate_node, memo_pos, operation_key):
        self.source_node = source_node
        self.target_node = target_node
        self.intermediate_node = intermediate_node
        self.memo_pos = memo_pos
        self.operation_key = operation_key
        self.timestamp = ns.sim_time()


class SwapApplyCorrectionMessage:
    """
    Signal message for apply swap correction

    :param source_node: source node name
    :param target_node: entangle node name
    :param intermediate_node: intermediate node name who will perform the swap
    :param memo_pos: memory position
    :param target_mem_pos: memory position that is on the other side of node
    :param operation_key: operation key is used to identify the swap operation
    :param m1 : measurement result source -> intermediate
    :param m2 : measurement result intermediate -> target
    """

    def __init__(self, source_node, target_node, intermediate_node, operation_key, memo_pos, target_mem_pos, m1, m2):
        self.source_node = source_node
        self.target_node = target_node
        self.intermediate_node = intermediate_node
        self.operation_key = operation_key
        self.memo_pos = memo_pos
        self.m1 = m1
        self.m2 = m2
        self.target_mem_pos = target_mem_pos
        self.timestamp = ns.sim_time()


class SwapApplyCorrectionSuccessMessage:
    """
    Signal message for swap correction success
    :param operation_key: operation key is used to identify the swap operation
    """

    def __init__(self, operation_key):
        self.operation_key = operation_key
        self.timestamp = ns.sim_time()


class SwapSuccessMessage:
    """
    Signal message for swap success from swap node to the left node
    :param source_node: source node name
    :param target_node: entangle node name
    :param intermediate_node: intermediate node name who will perform the swap
    :param memo_pos: memory position
    :param target_memo_pos: the entangled memory position respect to the remote node
    """

    def __init__(self, source_node, target_node, intermediate_node, memo_pos, target_memo_pos):
        self.source_node = source_node
        self.target_node = target_node
        self.intermediate_node = intermediate_node
        self.memo_pos = memo_pos
        self.target_memo_pos = target_memo_pos
        self.timestamp = ns.sim_time()


class SwapFailedMessage:
    """
    Signal message for swap failure from swap node to the left and right node
    :param source_node: source node name
    :param target_node: entangle node name
    :param memo_pos: memory position
    """

    def __init__(self, source_node, target_node, memo_pos):
        self.source_node = source_node
        self.target_node = target_node
        self.memo_pos = memo_pos
        self.timestamp = ns.sim_time()


class SwapEntangledSuccess(EntangleSignalMessage):
    """
    Signal message for swap success from source node to target node. This will be used to send to upper layer.
    :param source_node: source node name
    :param entangle_node: entangle node name
    :param actual_entangle_node: the actual entangle node name for this entanglement
    :param memo_pos: memory position
    :param target_memo_pos: target memory position with current entanglement
    """

    def __init__(self, source_node, entangle_node, actual_entangle_node, memo_pos, target_memo_pos, is_source):
        super().__init__(source_node, entangle_node, memo_pos)
        self.actual_entangle_node = actual_entangle_node
        self.target_memo_pos = target_memo_pos
        self.is_source = is_source


class TransportRequestMessage:
    """
    Signal message for requesting a teleportation
    :param source_node: source node name
    :param target_node: entangle node name
    :param target_memo_pos: teleporting node qubit memory position
    :param operation_key: operation key is used to identify the transmission operation
    """

    def __init__(self, source_node, target_node, target_memo_pos, operation_key):
        self.source_node = source_node
        self.target_node = target_node
        self.target_memo_pos = target_memo_pos
        self.operation_key = operation_key
        self.timestamp = ns.sim_time()


class TransportResponseMessage:
    """
    Signal message for ready for a teleportation operation
    :param operation_key : the operation key is used to identify the transmission operation
    """

    def __init__(self, operation_key):
        self.operation_key = operation_key
        self.timestamp = ns.sim_time()


class TransportApplyCorrectionMessage:
    """
    Signal message for apply correction on teleportation operation

    :param source_node: source node name
    :param target_node: entangle node name
    :param target_memo_pos: memory position
    :param operation_key: operation key is used to identify the swap operation
    :param m1 : measurement result source -> intermediate
    :param m2 : measurement result intermediate -> target
    """

    def __init__(self, source_node, target_node, target_memo_pos, operation_key, m1, m2):
        self.source_node = source_node
        self.target_node = target_node
        self.operation_key = operation_key
        self.target_memo_pos = target_memo_pos
        self.m1 = m1
        self.m2 = m2
        self.timestamp = ns.sim_time()


class TransportApplySuccessMessage:
    """
    Signal message for successfully applied correction on teleportation operation
    :param operation_key : the operation key is used to identify the transmission operation
    """

    def __init__(self, operation_key):
        self.operation_key = operation_key
        self.timestamp = ns.sim_time()


class SecuritySuccessSignalMessage(EntangleSignalMessage):
    """
    Security success signal message for send to upper layer. It is a subclass of EntangleSignalMessage

    """

    def __init__(self, source_node, entangle_node, is_source, security_mem_poses: list):
        super().__init__(source_node, entangle_node, security_mem_poses)
        self.is_source = is_source


class SecurityVerificationSignalMessage:
    """
    Base class for security verification signal messages
    """

    def __init__(self, entangle_node, source_verification_batch_id,
                 target_verification_batch_id,
                 source_verification_batch_poses: list,
                 target_verification_batch_poses: list):
        self.entangle_node = entangle_node
        self.source_verification_batch_id = source_verification_batch_id
        self.target_verification_batch_id = target_verification_batch_id
        self.source_verification_batch_poses = source_verification_batch_poses
        self.target_verification_batch_poses = target_verification_batch_poses
        self.timestamp = ns.sim_time()


class SecurityVerificationStartSignalMessage(SecurityVerificationSignalMessage):
    """
    Signal message for starting verification
    """

    def __init__(self, entangle_node, source_verification_batch_id,
                 source_verification_batch_poses: list,
                 target_verification_batch_id,
                 target_verification_batch_poses: list,
                 verification_teleport_measurement: dict):
        super().__init__(entangle_node, source_verification_batch_id,
                         target_verification_batch_id, source_verification_batch_poses,
                         target_verification_batch_poses)
        self.verif_teleport_measurement = verification_teleport_measurement


class SecurityVerificationResultSignalMessage(SecurityVerificationSignalMessage):
    """
    Signal message for verification result
    """

    def __init__(self, entangle_node, source_verification_batch_id, target_verification_batch_id,
                 source_verification_batch_poses: list,
                 target_verification_batch_poses: list,
                 verification_result: int, result_probability: float):
        super().__init__(entangle_node, source_verification_batch_id,target_verification_batch_id,
                         source_verification_batch_poses,
                         target_verification_batch_poses)
        self.verif_result = verification_result
        self.result_probability = result_probability
