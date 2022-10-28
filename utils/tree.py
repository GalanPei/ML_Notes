
def level_order(tree, type="val"):
    ret = []

    def dfs(root, level):
        if not root:
            return
        if len(ret) + 1 > level:
            ret.append([])
        eval(f"ret[level].append(root.{type})")
        dfs(root.left)
        dfs(root.right)
    dfs(tree)
    return ret


class Node(object):
    def __init__(self, val) -> None:
        self.val = val
        self.left = None
        self.right = None

    def level_order(self, type="val"):
        return level_order(self, type)

    def __str__(self) -> str:
        print(self.level_order(type="val"))


class WeightNode(Node):
    def __init__(self, val) -> None:
        super().__init__(val)
        assert(val > 0)
        self.weight = -1

    def update_weight(self):
        self.weight = 1

        def dfs(root):
            if not root:
                return
            if root.left and root.right:
                _left = root.left.val/(root.left.val + root.right.val)
                _right = 1 - _left
                root.left.weight = root.weight * _left
                root.right.weight = root.weight * _right
                dfs(root.left)
                dfs(root.right)
            elif root.left:
                root.left.weight = root.weight
                dfs(root.left)
            elif root.right:
                root.right.weight = root.weight
                dfs(root.right)
        
        dfs(self)
        
class HierarchicalSoftmax(object):
    def __init__(self) -> None:
        pass