"""
All the messages that are exchanged between the nodes are defined here.
"""


class ClassicalMessage:
    def __init__(self, from_node, to_node, data):
        self.from_node = from_node
        self.to_node = to_node
        self.data = data
