class BeamTree:
    def __init__(self, code, prob):
        self.code = code            # code du noeud
        self.prob = prob            # log proba du noeud
        self.cum_child_prob = 0     # meilleure log proba cumulée des noeuds fils
        self.children = []          # noeuds fils
        self.best_child = None      # meilleur noeud fils


def find_best(tree):
    if tree.children == []:
        return tree
    best_child = None
    best_cum_child_prob = 0
    for child in tree.children:
        best_grand_child = find_best(child).best_child
        best_grand_child_cum_child_prob = 0
        if best_grand_child != None:
            best_grand_child_cum_child_prob = best_grand_child.cum_child_prob
        cum_child_prob = child.prob + best_grand_child_cum_child_prob
        if cum_child_prob > best_cum_child_prob:
            best_cum_child_prob = cum_child_prob
            best_child = child
    tree.best_child = best_child
    tree.cum_child_prob = best_cum_child_prob
    return tree

if __name__ == "__main__":
    A = BeamTree('A', 10)
    B = BeamTree('B', 4)
    C = BeamTree('C', 7)
    D = BeamTree('D', 3)
    E = BeamTree('E', 5)
    F = BeamTree('F', 3)
    G = BeamTree('G', 1)

    D.children = [G]
    B.children = [D, E]
    C.children = [F]
    A.children = [B, C]

    A = find_best(A)

    root = A
    value = 0
    while root.best_child is not None:
        print(root.code, end=" ")
        root = root.best_child
