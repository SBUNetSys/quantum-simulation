class EntangleSignalMessage:
    """
    Base class for entanglement signal messages
    Most used for re-entanglement
    """

    def __init__(self, entangle_node, mem_pos):
        self.entangle_node = entangle_node
        self.mem_pos = mem_pos


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
