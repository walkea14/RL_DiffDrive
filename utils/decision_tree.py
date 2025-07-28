# utils/decision_tree.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class DecisionTreeLogger:
    """
    Records a tree of visited states (nodes) and actions (edges).
    Useful for one evaluation episode, or for sampling a few eps with noise.
    """
    def __init__(self, state_tol=1e-2):
        self.G = nx.DiGraph()
        self.state_tol = state_tol
        self.node_id = 0
        self.last_node = None

    def _round_state(self, s):
        return tuple(np.round(s, 2))

    def add_root(self, state):
        state_r = self._round_state(state)
        self.node_id += 1
        self.G.add_node(self.node_id, state=state_r)
        self.last_node = self.node_id
        return self.node_id

    def add_transition(self, from_id, state_next, action):
        state_r = self._round_state(state_next)
        # see if state already exists
        for n, data in self.G.nodes(data=True):
            if np.allclose(data["state"], state_r, atol=self.state_tol):
                to_id = n
                break
        else:
            self.node_id += 1
            to_id = self.node_id
            self.G.add_node(to_id, state=state_r)
        self.G.add_edge(from_id, to_id, action=tuple(np.round(action, 2)))
        return to_id

    def plot(self, figsize=(8, 6)):
        pos = nx.spring_layout(self.G, seed=0)
        edge_labels = nx.get_edge_attributes(self.G, 'action')
        node_labels = {n: f"{n}\n{self.G.nodes[n]['state']}" for n in self.G.nodes}

        plt.figure(figsize=figsize)
        nx.draw(self.G, pos, with_labels=True, labels=node_labels, node_size=700, font_size=8)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=7)
        plt.title("Decision Tree of Visited States")
        plt.tight_layout()
        plt.show()
