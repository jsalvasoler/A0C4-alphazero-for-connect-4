"""
This file implements a simple pygame UI for the game.
"""

import sys
import time

import pygame

from src.boards.bitboard import ConnectGameBitboard


class UI:
    def __init__(self, board: ConnectGameBitboard, agent=None):
        self.board = board
        self.agent = agent

        # Constants
        self.WIDTH, self.HEIGHT = 420, 480
        self.GRID_SIZE = 7
        self.CELL_SIZE = 60
        self.RADIUS = self.CELL_SIZE // 2 - 5
        self.PLAYER_COLORS = [(255, 0, 0), (0, 0, 255)]  # Red and Blue
        self.BG_COLOR = (255, 255, 255)
        self.LINE_COLOR = (0, 0, 0)
        self.FONT_SIZE = 30
        self.FONT_COLOR = (0, 0, 0)

        # Initialize Pygame
        pygame.init()

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Connect 4")
        self.font = pygame.font.Font(None, self.FONT_SIZE)

    def run(self):

        # Main game loop
        running = True
        is_over = False
        priors = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif (
                    self.agent is not None and not is_over and self.board.get_current_player() == 1
                ):
                    time.sleep(0.5)
                    # Agent's turn
                    col = self.agent.get_action(self.board)
                    priors = self.agent.get_priors(self.board)
                    is_over = self.board.step(col)
                    print(f"Column: {col} is played")
                    print(self.board)
                    print("is_over:", is_over)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX = event.pos[0]
                    col = mouseX // self.CELL_SIZE
                    if not board.can_play(col):
                        continue
                    priors = None
                    # Find the first available row in the selected column
                    is_over = board.step(col)
                    print(f"Column: {col} is played")
                    print(board)
                    print("is_over:", is_over)

            # Draw the board
            self.screen.fill(self.BG_COLOR)
            self.draw_board(board, not is_over, priors)

            if is_over:
                text = self.font.render(
                    f"{ {1: 'Red', -1: 'Blue'}.get(board.check_winner()) } wins!",
                    True,
                    self.FONT_COLOR,
                )
                text_rect = text.get_rect(
                    center=(self.WIDTH // 2, self.HEIGHT - self.CELL_SIZE // 2)
                )
                self.screen.blit(text, text_rect)
                running = False

            # Update the display
            pygame.display.flip()

            if is_over:
                time.sleep(3)

    # Function to draw the Connect 4 board
    def draw_board(self, board, turn=True, priors=None):
        for row in range(self.board.h):
            for col in range(self.board.w):
                pygame.draw.rect(
                    self.screen,
                    self.BG_COLOR,
                    (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE),
                )
                pygame.draw.circle(
                    self.screen,
                    self.LINE_COLOR,
                    (
                        col * self.CELL_SIZE + self.CELL_SIZE // 2,
                        row * self.CELL_SIZE + self.CELL_SIZE // 2,
                    ),
                    self.RADIUS,
                    0,
                )

        for row in range(self.board.h):
            for col in range(self.board.w):
                if self.board.state_representation[col][row] == 1:
                    pygame.draw.circle(
                        self.screen,
                        self.PLAYER_COLORS[0],
                        (
                            col * self.CELL_SIZE + self.CELL_SIZE // 2,
                            self.HEIGHT - (row + 3) * self.CELL_SIZE + self.CELL_SIZE // 2,
                        ),
                        self.RADIUS,
                        0,
                    )
                elif self.board.state_representation[col][row] == -1:
                    pygame.draw.circle(
                        self.screen,
                        self.PLAYER_COLORS[1],
                        (
                            col * self.CELL_SIZE + self.CELL_SIZE // 2,
                            self.HEIGHT - (row + 3) * self.CELL_SIZE + self.CELL_SIZE // 2,
                        ),
                        self.RADIUS,
                        0,
                    )

        # Print the player's turn
        if turn:
            text = self.font.render(
                f"Turn of { {0: 'red', 1: 'blue'}.get(board.get_current_player()) }",
                True,
                self.FONT_COLOR,
            )
            text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT - self.CELL_SIZE // 2))
            self.screen.blit(text, text_rect)

        if priors is not None:
            if max(priors) > 1 or min(priors) < 0:
                priors = [(a - min(priors)) / (max(priors) - min(priors)) for a in priors]
            # Print the priors under each column. Color of the played column has a different shade.
            for col in range(self.board.w):
                text = self.font.render(f"{priors[col]:.2f}", True, self.FONT_COLOR)
                text_rect = text.get_rect(
                    center=(
                        col * self.CELL_SIZE + self.CELL_SIZE // 2,
                        self.HEIGHT - 3 * self.CELL_SIZE // 2,
                    )
                )
                self.screen.blit(text, text_rect)


if __name__ == "__main__":
    board = ConnectGameBitboard()

    from src.agents.agent import OptimalAgent

    agent = OptimalAgent()
    ui = UI(board, agent=agent)
    ui.run()
