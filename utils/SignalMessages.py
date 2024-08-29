import netsquid as ns


class EntangleSignalMessage:
    """
    Base class for entanglement signal messages
    Most used for re-entanglement
    """

    def __init__(self, entangle_node, mem_pos):
        self.entangle_node = entangle_node
        self.mem_pos = mem_pos
        self.timestamp = ns.sim_time()


class NewEntanglementSignalMessage(EntangleSignalMessage):
    """
    Signal message for new entanglement
    Most used for entanglement creation
    """

    def __init__(self, entangle_node, mem_pos, qmemory_name, is_source, init_fidelity):
        super().__init__(entangle_node, mem_pos)
        self.qmemory_name = qmemory_name
        self.is_source = is_source
        self.init_fidelity = init_fidelity


class EntangleSuccessSignalMessage(EntangleSignalMessage):
    """
    Signal message for successful entanglement
    Most used for EntanglementHandler to notify success
    """

    def __init__(self, entangle_node, mem_pos, fidelity):
        super().__init__(entangle_node, mem_pos)
        self.fidelity = fidelity


class PurifySignalMessage:
    """
    Base class for purification signal messages
    """

    def __init__(self, entangle_node, qubit1_pos, qubit2_pos):
        self.entangle_node = entangle_node
        self.qubit1_pos = qubit1_pos
        self.qubit2_pos = qubit2_pos


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

    def __init__(self, entangle_node, mem_pos, new_fidelity):
        super().__init__(entangle_node, mem_pos)
        self.fidelity = new_fidelity


class PurifySuccessSignalMessage(EntangleSignalMessage):
    """
    Signal message for successful purification and sent to upper layer
    """

    def __init__(self, entangle_node, mem_pos, new_fidelity, is_source):
        super().__init__(entangle_node, mem_pos)
        self.fidelity = new_fidelity
        self.is_source = is_source


class PurifyFinishedSignalMessage:
    """
    Signal message for successful purification and stop the protocol
    """

    def __init__(self, entangle_node):
        self.entangle_node = entangle_node


class ReEntangleSignalMessage:
    """
    Signal message for re-entanglement, now we support list of re-entangle memory positions to avoid
    race condition
    """

    def __init__(self, entangle_node, re_entangle_mem_poses: list):
        self.entangle_node = entangle_node
        self.re_entangle_mem_poses = re_entangle_mem_poses


class ProtocolFinishedSignalMessage:
    """
    Signal message for protocol finished
    """

    def __init__(self, from_protocol, from_node):
        self.from_protocol = from_protocol
        self.from_node = from_node
        self.timestamp = ns.sim_time()


class VerificationSignalMessage:
    """
    Base class for verification signal messages
    """

    def __init__(self, entangle_node, verification_batch_id, verification_batch_poses: list):
        self.entangle_node = entangle_node
        self.verif_batch_id = verification_batch_id
        self.verif_batch_poses = verification_batch_poses


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