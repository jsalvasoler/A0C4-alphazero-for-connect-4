from math import sqrt

import numpy as np

from src.utils import Config

configuration = Config()


class TreeNode:
    """
    Represents a board state and stores statistics for actions at that state.

    Attributes:
        Nsa: An integer for visit count.
        Wsa: A float for the total action value.
        Qsa: A float for the mean action value.
        Psa: A float for the prior probability of reaching this node.
        action: Column index of the prior move of reaching this node.
        children: A list which stores child nodes.
        child_psas: A vector containing child probabilities.
        parent: A TreeNode representing the parent node.
    """

    def __init__(self, parent=None, action=None, psa=0.0, child_psas=None):
        """Initializes TreeNode with the initial statistics and data."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.children = []
        self.child_psas = [] if child_psas is None else child_psas
        self.parent = parent

    def is_not_leaf(self):
        """
        Checks if a TreeNode is a leaf (has no children).

        Returns:
            A boolean value indicating if a TreeNode is a leaf.
        """
        if len(self.children) > 0:
            return True
        return False

    def select_child(self):
        """
        Selects a child node based on the AlphaZero PUCT formula.

        Returns:
            A child TreeNode which is the most promising according to PUCT.
        """
        return max(
            self.children,
            key=lambda c: c.Qsa + c.Psa * configuration.c_puct * (sqrt(self.Nsa) / (1 + c.Nsa)),
        )

    def expand_node(self, game, psa_vector):
        """
        Expands the current node by adding valid moves as children.

        Args:
            game: An object containing the game state.
            psa_vector: A list containing move probabilities for each column (length = action_size).
        """
        self.child_psas = psa_vector
        valid_moves = game.get_valid_actions()
        for move in valid_moves:
            child_node = TreeNode(parent=self, action=move, psa=psa_vector[move])
            self.children.append(child_node)

    def back_prop(self, v):
        """
        Update the current node's statistics based on the evaluation.

        Args:
            v: A float representing the value of this state for the current player.
        """
        self.Nsa += 1
        self.Wsa += v
        self.Qsa = self.Wsa / self.Nsa


class MonteCarloTreeSearch:
    """Represents a Monte Carlo Tree Search Algorithm.

    Attributes:
        root: A TreeNode representing the board state and its statistics.
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, net):
        """
        Initializes TreeNode with the TreeNode, board and neural network.
        """
        self.root = None
        self.game = None
        self.net = net

    def search(self, game, node, temperature) -> TreeNode:
        """
        MCTS loop to get the best move which can be played at a given state.

        Args:
            game: An object containing the game state.
            node: A TreeNode representing the board state and its statistics.
            temperature: A float to control the level of exploration.

        Returns:
            A child node representing the best move to play at this state.
        """
        self.root = node
        self.game = game

        for _ in range(configuration.num_mcts_sims):
            node = self.root
            game = self.game.clone()  # Create a fresh clone for each loop.

            # Loop when node is not a leaf
            while node.is_not_leaf():
                node = node.select_child()
                game.step(node.action)

            # Check if this leaf is a terminal state
            winner = game.check_winner()
            if winner is not None:
                # Terminal node: use actual game outcome, don't expand
                v = winner
                # Back propagate from terminal node up to root
                while node is not None:
                    v = -v
                    node.back_prop(v)
                    node = node.parent
                continue

            # Get move probabilities and values from the network for this state.
            psa_vector, v = self.net.predict(game.get_state_representation())

            # Mask invalid moves and renormalize
            valid_moves = game.get_valid_actions()
            mask = np.zeros(len(psa_vector))
            for move in valid_moves:
                mask[move] = 1.0
            psa_vector = psa_vector * mask

            # Add Dirichlet noise to the psa_vector of the root node.
            if node.parent is None:
                psa_vector = self.add_dirichlet_noise(game, psa_vector)

            psa_vector_sum = sum(psa_vector)

            # Normalize psa vector
            if psa_vector_sum > 0:
                psa_vector /= psa_vector_sum
            else:
                # Fallback: uniform over valid moves
                for move in valid_moves:
                    psa_vector[move] = 1.0 / len(valid_moves)

            # Expand the current node.
            node.expand_node(game=game, psa_vector=psa_vector)

            # Back propagate node statistics up to the root node.
            while node is not None:
                v = -v
                node.back_prop(v)
                node = node.parent

        # Select child based on temperature
        if temperature < 1:
            # Deterministic: pick most-visited child
            best_child = max(self.root.children, key=lambda c: c.Nsa)
        else:
            # Proportional to visit counts
            visits = np.array([c.Nsa for c in self.root.children], dtype=np.float64)
            visits /= visits.sum()
            best_child = self.root.children[np.random.choice(len(self.root.children), p=visits)]

        return best_child

    def get_policy(self, temperature):
        """
        Get the MCTS-improved policy (visit count distribution) from the root node.

        Args:
            temperature: A float to control exploration vs exploitation.

        Returns:
            A numpy array of length action_size with the visit count policy.
        """
        action_size = self.game.w
        policy = np.zeros(action_size)

        for child in self.root.children:
            policy[child.action] = child.Nsa

        if temperature < 1:
            # Near-deterministic: all weight on most-visited action
            best = np.argmax(policy)
            policy = np.zeros(action_size)
            policy[best] = 1.0
        else:
            # temperature == 1: proportional to visit counts (no exponent needed)
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy /= policy_sum

        return policy

    @staticmethod
    def add_dirichlet_noise(game, psa_vector):
        """Add Dirichlet noise to the psa_vector of the root node.

        This is for additional exploration.

        Args:
            game: An object containing the game state.
            psa_vector: A probability vector.

        Returns:
            A probability vector which has Dirichlet noise added to it.
        """
        dirichlet_input = [configuration.dirichlet_alpha for x in range(game.w)]

        dirichlet_list = np.random.dirichlet(dirichlet_input)
        noisy_psa_vector = []

        for idx, psa in enumerate(psa_vector):
            noisy_psa_vector.append(
                (1 - configuration.epsilon) * psa + configuration.epsilon * dirichlet_list[idx]
            )

        return noisy_psa_vector

    def print_stats(self):
        print("MCTS Stats:")
        # Count total states explored (nodes in the tree)
        count = 0
        for node in self.root.children:
            count += self.count_children(node)
        print("Total states explored:", count)

        # Count levels of the tree
        levels = 0
        for node in self.root.children:
            levels = max(levels, self.count_levels(node))
        print("Levels of the tree:", levels)

    def count_children(self, node):
        count = 1
        for child in node.children:
            count += self.count_children(child)
        return count

    def count_levels(self, node):
        if node.is_not_leaf():
            count = 0
            for child in node.children:
                count = max(count, self.count_levels(child))
            return count + 1
        else:
            return 0
