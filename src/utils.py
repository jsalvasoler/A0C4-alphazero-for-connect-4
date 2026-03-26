from abc import ABC, abstractmethod
from typing import Union
import requests
import os
import shelve

import yaml


class Game(ABC):
    """
    Abstract class for a game
    """

    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action) -> bool:
        pass

    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def check_winner(self) -> Union[int, None]:
        """
        Check if there is a winner.
        Return:
             1 if starting player wins,
             -1 if opponent wins,
             0 if there is a draw,
             None if game is not over.
        """
        pass


class Agent(ABC):
    @abstractmethod
    def get_action(self, game: Game):
        pass

    @abstractmethod
    def get_priors(self, game: Game):
        pass


class SolverAgent(Agent):
    """Agent that has access to the online Connect 4 solver for optimal evaluations."""

    _BASE_URL = 'https://connect4.gamesolver.org/solve?pos='
    _HEADERS = {'User-Agent': 'Mozilla/5.0'}
    _MAX_CACHE_BYTES = 250 * 1024 * 1024

    def __init__(self):
        self._session = requests.Session()

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(root, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = shelve.open(os.path.join(cache_dir, 'cache.db'))

    def __del__(self):
        if hasattr(self, '_cache'):
            self._cache.close()

    def get_optimal_evaluations(self, game) -> list:
        key = "".join([str(s + 1) for s in game.history])
        if key in self._cache:
            value = self._cache[key]
            if len(value) == 7:
                return value
            del self._cache[key]

        url = f'{self._BASE_URL}{key}'
        response = self._session.get(url, headers=self._HEADERS)
        response.raise_for_status()
        scores = response.json()['score']

        self._cache[key] = scores
        self._cache.sync()

        return scores

    def get_action_accuracy(self, game, action) -> float:
        evaluations = self.get_optimal_evaluations(game)
        if evaluations[action] == 100:
            return 0
        x = evaluations[action]
        valid_evals = [e for e in evaluations if e != 100]
        return 1 if x == max(valid_evals) else 0


class Config:
    """
    Configuration object for the project.
    """
    def __init__(self):
        # Read config from config.yaml.
        self.__root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._config_path = os.path.join(self.__root, 'config.yaml')

        with open(self._config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        self.model_dir_path = os.path.join(self.__root, 'models')

    def __getattr__(self, item):
        # if the item is already an attribute of self, return it
        if item in self.__dict__:
            return self.__dict__[item]

        # if the item exists as a key in self._config, return it
        try:
            return self._config[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")
