class SwapNode:
    def __init__(self, left, right, parent):
        self.left = left
        self.right = right
        self.parent = parent


def generate_swapping_tree(node_path):
    """
    using depth-first search to generate a swapping tree
    :param node_path:
    :return:
    """
    if len(node_path) < 3:
        return None, []

    swap_nodes = []

    levels = []

    while len(node_path) >= 3:
        level = []
        swap_node = []
        for i in range(1, len(node_path) - 1, 2):
            swap = SwapNode(node_path[i - 1], node_path[i + 1], node_path[i])
            level.append(node_path[i])
            swap_node.append(swap)
        levels.append(level)
        swap_nodes.append(swap_node)
        # remove the nodes that are already swapped
        for i in level:
            node_path.remove(i)

    return swap_nodes, levels


if __name__ == '__main__':
    # Example usage
    nodes_path = ["A", "B", "C", "D", "E", "F", "G"]
    swap_nodes, levels = generate_swapping_tree(nodes_path)
    print(swap_nodes)
    print(levels)
