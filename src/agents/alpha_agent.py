from src.alpha_zero.neural_net import NNWrapper
from src.boards.bitboard import ConnectGameBitboard
from src.utils import Agent


class AlphaAgent(Agent):
    """
    This agent uses the best neural network found using the AlphaZero training step to get the most desired action.
    """

    def __init__(self):
        self.game = ConnectGameBitboard()

        self.net = NNWrapper(self.game)
        self.net.load_model(filename='best_model')

    def get_action(self, game: ConnectGameBitboard):
        """
        Get the optimal action according to the policy head of the neural network.

        Args:
            game: An object containing the game state.
        """
        pca_vector, v = self.net.predict(game.get_state_representation())
        action = max(game.get_valid_actions(), key=lambda x: pca_vector[x])
        return action

    def get_priors(self, game: ConnectGameBitboard):
        return self.net.predict(game.get_state_representation())[0]
