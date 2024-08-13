class EntangleSignalMessage:
    def __init__(self, entangle_node, mem_pos):
        self.entangle_node = entangle_node
        self.mem_pos = mem_pos


class GenEntanglementSignalMessage:
    """
    Message to generate entanglement
    {"mem_pos": mem_pos, "qmemory": self._qmemory_name,
    "is_source": self._is_source,
    "initial_fidelity": init_fidelity,
    "entangle_node": self.entangle_node}
    """

    def __init__(self, mem_pos, qmemory_name, is_source, init_fidelity, entangle_node):
        self.mem_pos = mem_pos
        self.qmemory_name = qmemory_name
        self.is_source = is_source
        self.init_fidelity = init_fidelity
        self.entangle_node = entangle_node
