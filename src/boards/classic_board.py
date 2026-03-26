import numpy as np

from src.utils import Game


class ConnectGameClassicBoard(Game):
    def __init__(self):
        super().__init__()
        self.size = (6, 7)
        self.in_a_row = 4
        self.board = None
        self.turn = (
            None  # can be 1 and -1. The next piece to be placed is going to be of sign self.turn.
        )
        self.winner = None  # can be 1, -1, and 0 (draw). None if the game is not over.

        self.reset()

    def reset(self):
        self.board = np.zeros(self.size, dtype=int)
        self.turn = 1
        self.winner = None

    def step(self, action) -> bool:
        assert self.winner is None, "Game has already ended."
        assert self.board[0, action] == 0, "Invalid action. The column is full."

        # Look for the first non-empty row in the column action
        for i in range(self.size[0] - 1, -1, -1):
            if self.board[i, action] == 0:
                self.board[i, action] = self.turn
                break

        self.winner = self.check_winner()
        self.turn *= -1  # switch turn

        # Return True if the game is over
        return self.winner is not None

    def check_winner(self):
        if np.sum(self.board == 1) < self.in_a_row and np.sum(self.board == -1) < self.in_a_row:
            return None
        # Check horizontal
        for i in range(self.size[0]):
            for j in range(self.size[1] - self.in_a_row + 1):
                if np.all(self.board[i, j : j + self.in_a_row] == self.turn):
                    return self.turn

        # Check vertical
        for i in range(self.size[0] - self.in_a_row + 1):
            for j in range(self.size[1]):
                if np.all(self.board[i : i + self.in_a_row, j] == self.turn):
                    return self.turn

        # Check diagonal
        for i in range(self.size[0] - self.in_a_row + 1):
            for j in range(self.size[1] - self.in_a_row + 1):
                if np.all(
                    np.diag(self.board[i : i + self.in_a_row, j : j + self.in_a_row]) == self.turn
                ):
                    return self.turn
                if np.all(
                    np.diag(np.fliplr(self.board[i : i + self.in_a_row, j : j + self.in_a_row]))
                    == self.turn
                ):
                    return self.turn

        # Check draw
        if np.all(self.board != 0):
            return 0

        return None

    def get_valid_actions(self):
        # An action is simply the column index
        valid_actions = [j for j in range(self.size[1]) if self.board[0, j] == 0]

        return valid_actions

    def __repr__(self):
        # return string in elegant manner using unicode characters
        ans = "  " + " ".join([str(i) for i in range(self.size[1])]) + "\n"
        for i in range(self.size[0]):
            ans += str(i) + " "
            for j in range(self.size[1]):
                if self.board[i, j] == 1:
                    ans += "x "
                elif self.board[i, j] == -1:
                    ans += "o "
                else:
                    ans += ". "
            ans += "\n"
        return ans


if __name__ == "__main__":
    from src.boards.bitboard import ConnectGameBitboard

    board = ConnectGameBitboard()
    print(board)

    while board.check_winner() is None:
        action = int(input("Enter action: "))
        board.step(action)
        print(board)
    print(board.check_winner())
