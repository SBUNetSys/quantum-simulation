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

    def __init__(self, mem_pos, qmemory_name, is_source, init_fidelity, entangle_node):
        super().__init__(entangle_node, mem_pos)
        self.qmemory_name = qmemory_name
        self.is_source = is_source
        self.init_fidelity = init_fidelity
