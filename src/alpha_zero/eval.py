from src.alpha_zero.mcts import TreeNode
from src.boards.bitboard import ConnectGameBitboard as Game
from src.utils import Config

configuration = Config()


class Evaluate:
    """Evaluate the current network against the evaluation network

    Attributes:
        current_mcts: An object for the current network's MCTS.
        eval_mcts: An object for the evaluation network's MCTS.
    """

    def __init__(self, current_mcts, eval_mcts):
        """Initializes Evaluate with the both network's MCTS and game state."""
        self.current_mcts = current_mcts
        self.eval_mcts = eval_mcts

    def evaluate(self):
        """
        Play self-play games between the two networks and record game stats.
        Alternates which network plays first to avoid first-mover bias.

        Returns:
            Wins and losses count from the perspective of the current network.
        """
        wins = 0
        losses = 0

        # Self-play loop
        for i in range(configuration.num_eval_games):
            print("Start Evaluation Self-Play Game:", i, "\n")

            game = Game()  # Create a fresh clone for each game.
            game_over = False
            value = 0
            node = TreeNode()

            # Alternate which network goes first each game
            current_plays_first = i % 2 == 0

            # Keep playing until the game is in a terminal state.
            while not game_over:
                # Determine which MCTS to use for the current player
                is_current_turn = (game.get_current_player() == 0) == current_plays_first

                if is_current_turn:
                    best_child = self.current_mcts.search(game, node, configuration.temp_final)
                else:
                    best_child = self.eval_mcts.search(game, node, configuration.temp_final)

                action = best_child.action
                game_over = game.step(action)  # Play the child node's action.

                print(game)

                if game_over:
                    value = game.check_winner()

                best_child.parent = None
                node = best_child  # Make the child node the root node.

            # Map game result to current network's perspective
            # value: +1 = player 0 wins, -1 = player 1 wins, 0 = draw
            if current_plays_first:
                current_value = value  # current is player 0
            else:
                current_value = -value  # current is player 1

            if current_value == 1:
                print("win")
                wins += 1
            elif current_value == -1:
                print("loss")
                losses += 1
            else:
                print("draw")
            print("\n")

        return wins, losses
