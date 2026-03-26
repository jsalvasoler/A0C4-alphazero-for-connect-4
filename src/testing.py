import numpy as np
from tqdm import tqdm

from src.boards.bitboard import ConnectGameBitboard
from src.boards.classic_board import ConnectGameClassicBoard
from src.utils import Agent, SolverAgent


class TestEnvironment:
    """
    This class tests the performance of two agents against each other.

    Accuracy is computed automatically when at least one agent is an SolverAgent,
    since it already fetches optimal evaluations from the online solver.

    Args:
        agent_1: An object containing the first agent.
        agent_2: An object containing the second agent.
        n_games: An integer representing the number of games to play.
        bitboard: A boolean representing whether to use bitboard or classic board.
    """

    def __init__(self, agent_1: Agent, agent_2: Agent, n_games=1000, bitboard=True):
        self.agent_1 = agent_1
        self.agent_2 = agent_2

        self.n_games = n_games
        self.n_wins = 0
        self.n_draws = 0

        self.bitboard = bitboard

        # Use whichever agent is a SolverAgent to evaluate move accuracy for both
        self._solver = next((a for a in (agent_1, agent_2) if isinstance(a, SolverAgent)), None)

    def initialize_board(self):
        """
        Initialize the board.

        Returns:
            Game object.
        """
        if self.bitboard:
            return ConnectGameBitboard()
        else:
            return ConnectGameClassicBoard()

    def run(self):
        """
        Run the tests. Run half the games with agent 1 starting, half with agent 2 starting
        """
        print(f"Running first batch: {self.n_games // 2} games with agent 1 starting")
        wins_1, draws_1, move_acc_1_1, move_acc_2_1 = self.run_batch(
            self.n_games // 2, agent_1_starts=True
        )
        print(f"Running second batch: {self.n_games // 2} games with agent 2 starting")
        wins_2, draws_2, move_acc_1_2, move_acc_2_2 = self.run_batch(
            self.n_games // 2, agent_1_starts=False
        )

        self.n_wins = wins_1 + (self.n_games // 2 - wins_2 - draws_2)
        self.n_draws = draws_1 + draws_2

        print("\n\n -- Results --")
        win_pct = round(self.n_wins / self.n_games * 100, 2)
        loss_pct = round((self.n_games - self.n_wins - self.n_draws) / self.n_games * 100, 2)
        a1_line = f"Agent 1 wins {self.n_wins}/{self.n_games} ({win_pct}%)"
        a2_line = (
            f"Agent 2 wins {self.n_games - self.n_wins - self.n_draws}/{self.n_games} ({loss_pct}%)"
        )
        if self._solver:
            avg_acc_1 = 0.5 * (move_acc_1_1 or 0) + 0.5 * (move_acc_1_2 or 0)
            avg_acc_2 = 0.5 * (move_acc_2_1 or 0) + 0.5 * (move_acc_2_2 or 0)
            a1_line += f" with an accuracy of {avg_acc_1:.4f}"
            a2_line += f" with an accuracy of {avg_acc_2:.4f}"
        print(a1_line)
        print(a2_line)
        print(f"Draws: {self.n_draws}\n")

    def run_batch(self, n_games, agent_1_starts=True):
        """
        Run a batch of games.

        Args:
            n_games: Number of games to play.
            agent_1_starts: A boolean representing whether agent 1 starts.

        Returns:
            Number of wins, number of draws, move accuracy of agent 1, move accuracy of agent 2.
        """
        wins = 0
        draws = 0
        x = 0 if agent_1_starts else 1

        move_acc_1 = []
        move_acc_2 = []

        for _ in tqdm(range(n_games)):
            game = self.initialize_board()
            is_over = False
            turn = 0
            while not is_over:
                if turn % 2 == x:
                    action = self.agent_1.get_action(game)
                    if self._solver:
                        move_acc_1.append(self._solver.get_action_accuracy(game, action))
                else:
                    action = self.agent_2.get_action(game)
                    if self._solver:
                        move_acc_2.append(self._solver.get_action_accuracy(game, action))
                is_over = game.step(action)
                turn += 1

            winner = game.check_winner()
            if winner == 1:
                wins += 1
            elif winner == 0:
                draws += 1
        acc_1 = np.mean(move_acc_1) if move_acc_1 else None
        acc_2 = np.mean(move_acc_2) if move_acc_2 else None
        return wins, draws, acc_1, acc_2


if __name__ == "__main__":
    from pyinstrument import Profiler

    from src.agents.agent import RandomAgent
    from src.agents.alpha_agent import AlphaAgent

    profiler = Profiler()
    profiler.start()

    agent_1 = AlphaAgent()
    agent_2 = RandomAgent()
    testing = TestEnvironment(agent_1, agent_2, n_games=50, bitboard=True)
    testing.run()

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
