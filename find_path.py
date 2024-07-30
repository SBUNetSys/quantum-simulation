class SwapNode:
    def __init__(self, left, right, parent):
        self.left = left
        self.right = right
        self.parent = parent


def generate_swapping_tree(nodes):
    """
    using depth-first search to generate a swapping tree
    :param nodes:
    :return:
    """
    if len(nodes) < 3:
        return None, []

    swap_nodes = []

    levels = []

    while len(nodes) >= 3:
        level = []
        swap_node = []
        for i in range(1, len(nodes) - 1, 2):
            swap = SwapNode(nodes[i - 1], nodes[i + 1], nodes[i])
            level.append(nodes[i])
            swap_node.append(swap)
        levels.append(level)
        swap_nodes.append(swap_node)
        # remove the nodes that are already swapped
        for i in level:
            nodes.remove(i)

    return swap_nodes, levels



def print_swapping_tree(root, swap_nodes):
    print("Swapping operations:")
    for i, swap in enumerate(swap_nodes):
        print(f"Swap {i + 1}: {swap.left}-{swap.right} through {swap.parent}")

    print("\nFinal entanglement:")
    print(f"{root.left}-{root.right} through {root.parent}")


# Example usage
nodes = ["A", "B", "C", "D", "E", "F", "G"]
root, swap_nodes = generate_swapping_tree(nodes)
print_swapping_tree(root, swap_nodes)