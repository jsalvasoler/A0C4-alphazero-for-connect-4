"""
CLI entry point for A0C4 — AlphaZero for Connect 4.

Usage:
    uv run python -m src.cli play --agent random
    uv run python -m src.cli play --agent optimal
    uv run python -m src.cli play --agent alpha
    uv run python -m src.cli train
    uv run python -m src.cli test --agent1 alpha --agent2 random --games 100
"""

import argparse
import sys


def make_agent(name: str):
    from src.agents.agent import RandomAgent, OptimalAgent
    from src.agents.alpha_agent import AlphaAgent

    agents = {
        "random": RandomAgent,
        "optimal": OptimalAgent,
        "alpha": AlphaAgent,
    }
    if name not in agents:
        print(f"Unknown agent: {name}. Choose from: {', '.join(agents)}")
        sys.exit(1)
    return agents[name]()


def cmd_play(args):
    from src.boards.bitboard import ConnectGameBitboard
    from src.boards.ui import UI

    board = ConnectGameBitboard()
    agent = make_agent(args.agent)
    ui = UI(board, agent=agent)
    ui.run()


def cmd_train(args):
    from src.alpha_zero.neural_net import NNWrapper
    from src.alpha_zero.train import Train
    from src.boards.bitboard import ConnectGameBitboard as Game

    game = Game()
    net = NNWrapper(game)
    train = Train(game, net)
    train.run()


def cmd_test(args):
    from src.testing import TestEnvironment

    agent1 = make_agent(args.agent1)
    agent2 = make_agent(args.agent2)
    env = TestEnvironment(agent1, agent2, n_games=args.games)
    env.run()


def main():
    parser = argparse.ArgumentParser(description="A0C4 — AlphaZero for Connect 4")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # play
    play_parser = subparsers.add_parser("play", help="Play against an agent in the UI")
    play_parser.add_argument("--agent", default="optimal", choices=["random", "optimal", "alpha"],
                             help="Agent to play against (default: optimal)")

    # train
    subparsers.add_parser("train", help="Train the AlphaZero neural network")

    # test
    test_parser = subparsers.add_parser("test", help="Pit two agents against each other")
    test_parser.add_argument("--agent1", default="alpha", choices=["random", "optimal", "alpha"],
                             help="First agent (default: alpha)")
    test_parser.add_argument("--agent2", default="random", choices=["random", "optimal", "alpha"],
                             help="Second agent (default: random)")
    test_parser.add_argument("--games", type=int, default=100, help="Number of games (default: 100)")

    args = parser.parse_args()

    if args.command == "play":
        cmd_play(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "test":
        cmd_test(args)


if __name__ == "__main__":
    main()
