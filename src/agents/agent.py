import random

from src.boards.bitboard import ConnectGameBitboard
from src.utils import Agent, Game, SolverAgent


class RandomAgent(Agent):
    """
    This agent chooses a random action from the list of valid actions.
    """

    def get_action(self, game: Game):
        return random.choice(game.get_valid_actions())

    def get_priors(self, game: Game):
        return [1 / len(game.get_valid_actions()) for _ in game.get_valid_actions()]


class OptimalAgent(SolverAgent):
    """Uses the online Connect 4 solver to get exact evaluations.

    Plays optimally: always wins if starting first, always draws if
    starting second against another optimal player.
    See https://connect4.gamesolver.org/
    """

    def get_action(self, game: ConnectGameBitboard):
        evaluations = self.get_optimal_evaluations(game)
        valid_actions = game.get_valid_actions()
        random.shuffle(valid_actions)
        action = max(valid_actions, key=lambda x: evaluations[x])
        return action

    def get_priors(self, game: ConnectGameBitboard):
        return self.get_optimal_evaluations(game)
