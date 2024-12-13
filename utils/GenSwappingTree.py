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

    swap_nodes_level = []
    swap_nodes = []
    levels = []

    while len(node_path) >= 3:
        level = []
        swap_node = []
        for i in range(1, len(node_path) - 1, 2):
            swap = SwapNode(node_path[i - 1], node_path[i + 1], node_path[i])
            level.append(node_path[i])
            swap_node.append(swap)
            swap_nodes.append(swap)
        levels.append(level)
        swap_nodes_level.append(swap_node)
        # remove the nodes that are already swapped
        for i in level:
            node_path.remove(i)

    return swap_nodes, swap_nodes_level, levels


if __name__ == '__main__':
    # Example usage
    nodes_path = ["A", "B", "C", "D", "E", "F", "G"]
    nodes, swap_levels, level_text = generate_swapping_tree(nodes_path)
    print(nodes)
    print(swap_levels)
    print(level_text)
