from copy import deepcopy as copy

import numpy as np

from src.utils import Game


class ConnectGameBitboard(Game):
    """
    Connect 4 game representation using bitboard for fast move generation and evaluation.
    It is an implementation of the Game interface.
    """

    def __init__(self, width=7, height=6):
        super().__init__()
        self.w = width
        self.h = height

        self._players_map = {0: 1, 1: -1}

        self.board_state = None
        self.col_heights = None
        self.moves = None
        self.history = None
        self.node_count = None
        self.bit_shifts = None
        self.base_search_order = None

        # State representation for that serves as the NN input
        self.state_representation = np.zeros((self.w, self.h), dtype=np.int8)

        self.reset()

    def reset(self):
        """
        Resets the game state to the initial state.
        """
        self.board_state = [0, 0]
        self.col_heights = [(self.h + 1) * i for i in range(self.w)]
        self.moves = 0
        self.history = []
        self.node_count = 0
        self.bit_shifts = self.__get_bit_shifts()

    def __repr__(self):
        """
        Returns a string representation of the board state.
        """
        state = []
        for i in range(self.h):  # row
            row_str = str(self.h - i - 1) + " "
            for j in range(self.w):  # col
                pos = 1 << (self.h + 1) * j + i
                if self.board_state[0] & pos == pos:
                    row_str += "x "
                elif self.board_state[1] & pos == pos:
                    row_str += "o "
                else:
                    row_str += ". "
            state.append(row_str)
        state.append("  " + " ".join([str(i) for i in range(self.w)]))
        state.reverse()  # inverted orientation more readable
        return "\n".join(state)

    def get_current_player(self):
        """
        Returns current player: 0 or 1 (0 always plays first)
        """
        return self.moves & 1

    def get_opponent(self):
        """
        Returns opponent to current player: 0 or 1
        """
        return (self.moves + 1) & 1

    def get_mask(self):
        """
        Returns bitstring of all occupied positions
        """
        return self.board_state[0] | self.board_state[1]

    def get_key(self):
        """
        Returns unique game state identifier
        """
        return self.get_mask() + self.board_state[self.get_current_player()]

    def can_play(self, col):
        """
        Returns true if col (zero indexed) is playable
        """
        return not self.get_mask() & 1 << (self.h + 1) * col + (self.h - 1)

    def play(self, col):
        """
        Play a move in the given column (zero indexed)

        Args:
            col: Column to play in (zero indexed)
        """
        player = self.get_current_player()
        move = 1 << self.col_heights[col]
        assert self.can_play(col), f"Column {col} is full"
        self.col_heights[col] += 1
        self.state_representation[col][self.col_heights[col] - (self.h + 1) * col - 1] = (
            self._players_map[player]
        )
        self.board_state[player] |= move
        self.history.append(col)
        self.moves += 1

    def get_state_representation(self):
        """
        Return a np.array of shape (h, w) with the canonical board state
        The canonical board state is the board state from the perspective of the current player

        Returns:
            A numpy array of shape (h, w) with the canonical board state.
        """
        return self.state_representation * self._players_map[self.get_current_player()]

    def winning_board_state(self):
        """
        Returns true if last played column creates winning alignment
        """
        opp = self.get_opponent()
        for shift in self.bit_shifts:
            test = self.board_state[opp] & (self.board_state[opp] >> shift)
            if test & (test >> 2 * shift):
                return True
        return False if self.moves < self.w * self.h else True

    def get_score(self):
        """
        Returns score of complete game (evaluated for winning opponent)
        """
        return -(self.w * self.h + 1 - self.moves) // 2

    def __get_bit_shifts(self):
        """
        Returns list of bit shifts to check for winning alignments
        """
        return [
            1,  # | vertical
            self.h,  # \ diagonal
            self.h + 1,  # - horizontal
            self.h + 2,  # / diagonal
        ]

    def step(self, action: int) -> bool:
        """
        Play a move in the given column (zero indexed) and check if game is over.

        Args:
            action: Column to play in (zero indexed)

        Returns:
            True if game is over, False otherwise.
        """

        self.play(action)
        if self.check_winner() is not None:
            return True
        return False

    def get_valid_actions(self) -> list:
        """
        Returns:
            List of valid actions (zero indexed)
        """
        return [c for c in range(self.w) if self.can_play(c)]

    def check_winner(self) -> int | None:
        """
        Check if there is a winner.
        Returns:
            1 if starting player wins,
            -1 if opponent wins,
            0 if there is a draw,
            None if game is not over.
        """
        if self.winning_board_state():
            if self.moves == self.w * self.h:
                return 0
            return self._players_map[self.get_opponent()]
        return None

    def clone(self):
        """
        Returns:
            A deep copy of the current game state.
        """
        return copy(self)


if __name__ == "__main__":
    game = ConnectGameBitboard()

    is_over = False
    while not is_over:
        print(game)
        import random

        action = random.choice(game.get_valid_actions())
        is_over = game.step(action)

    print(game)
    print(game.check_winner())
