import spacy


nlp = spacy.load("en_core_web_sm")


class QuestionSyntaxTree:

    def __init__(self, question: str):

        self.depth, self.edges, self.leaves = self._get_dependency_tree_stats(question)

    def _walk_dependency_tree(self, node, stats, cur_depth) -> None:
        children = [child for child in node.children]

        if len(children) == 0:
            stats["leaves"] += 1
            if stats["depth"] < cur_depth:
                stats["depth"] = cur_depth
            return

        stats["edges"] += len(children)

        for child in children:
            self._walk_dependency_tree(child, stats, cur_depth+1)

    @staticmethod
    def _union_dependency_trees_stats(dependency_trees_stats: list[dict[str, float]]) -> (int, int, int):
        """Unions the statistics of multiple dependency trees."""
        depths, edges, leaves = [], [], []
        for dependency_tree_stats in dependency_trees_stats:
            depths.append(dependency_tree_stats["depth"])
            edges.append(dependency_tree_stats["edges"])
            leaves.append(dependency_tree_stats["leaves"])

        return max(depths), sum(edges), sum(leaves)

    def _get_dependency_tree_stats(self, question: str) -> (int, int, int):
        """
        Returns the statistics of the question's dependency tree.

        Args:
            question(str): The given question.

        Returns (tuple[int, int, int):
            The (depth, edges, leaves) of the question's dependency tree.
        """
        doc = nlp(question)

        roots = [token for token in doc if token.head == token]

        dependency_trees_stats = []
        for root in roots:
            dependency_tree_stats = {"depth": 0, "edges": 0, "leaves": 0}
            self._walk_dependency_tree(root, dependency_tree_stats, 0)
            dependency_trees_stats.append(dependency_tree_stats)

        return self._union_dependency_trees_stats(dependency_trees_stats)
