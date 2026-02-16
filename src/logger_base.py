import abc
import time
from typing import Any

import numpy as np


class LoggerBase(abc.ABC):
    """Logger interface definition."""

    @property
    def n_episodes(self) -> int:
        """Number of episodes."""
        return 0

    @abc.abstractmethod
    def start_new_episode(self):
        """Register start of new episode."""

    @abc.abstractmethod
    def stop_episode(self, total_steps: int):
        """Register end of episode.

        Parameters
        ----------
        total_steps : int
            Total number of steps in the episode that just terminated.
        """

    @abc.abstractmethod
    def define_experiment(
        self,
        env_name: str | None = None,
        algorithm_name: str | None = None,
        hparams: dict | None = None,
    ):
        """Define the experiment.

        Parameters
        ----------
        env_name : str, optional
            The name of the gym environment.

        algorithm_name : str, optional
            The name of the reinforcement learning algorithm.

        hparams : dict, optional
            Hyperparameters of the experiment.
        """

    @abc.abstractmethod
    def record_stat(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
        verbose: int | None = None,
        format_str: str = "{0:.3f}",
    ):
        """Record statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        value : Any
            Value that should be recorded.

        episode : int, optional
            Episode which we record the statistic.

        step : int, optional
            Step at which we record the statistic.

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """

    def define_checkpoint_frequency(  # noqa: B027
        self, key: str, checkpoint_interval: int
    ):
        """Define the checkpoint frequency for a function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        checkpoint_interval : int
            Number of steps after which the function approximator should be
            saved.
        """

    def record_epoch(  # noqa: B027
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
    ):
        """Record training epoch of function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        value : Any
            Function approximator.

        episode : int, optional
            Episode which we record the statistic.

        step : int, optional
            Step at which we record the statistic.

        t : float, optional
            Wallclock time, measured with time.time().
        """

class MemoryLogger(LoggerBase):
    """Logger class to record experiment statistics in memory.

    This logger stores experiment statistics in memory.

    Parameters
    ----------
    verbose : int, optional
        Verbosity level.
    """

    env_name: str | None
    algorithm_name: str | None
    start_time: float
    hparams: dict | None = None
    _n_episodes: int
    n_steps: int
    stats_loc: dict[str, list[tuple[int | None, int | None, float | None]]]
    stats: dict[str, list[Any]]

    def __init__(self):
        self.env_name = None
        self.algorithm_name = None
        self.start_time = 0.0
        self.hparams = None
        self._n_episodes = 0
        self.n_steps = 0
        self.stats_loc = {}
        self.stats = {}

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    def start_new_episode(self):
        """Register start of new episode."""
        self._n_episodes += 1

    def stop_episode(self, total_steps: int):
        """Register end of episode.

        Increase step counter and records 'episode_length'.

        Parameters
        ----------
        total_steps : int
            Total number of steps in the episode that just terminated.
        """
        self.n_steps += total_steps
        self.record_stat("episode_length", total_steps, verbose=0)

    def define_experiment(
        self,
        env_name: str | None = None,
        algorithm_name: str | None = None,
        hparams: dict | None = None,
    ):
        """Define the experiment.

        Parameters
        ----------
        env_name : str, optional
            The name of the gym environment.

        algorithm_name : str, optional
            The name of the reinforcement learning algorithm.

        hparams : dict, optional
            Hyperparameters of the experiment.
        """
        self.env_name = env_name
        self.algorithm_name = algorithm_name
        self.start_time = time.time()
        self.hparams = hparams

    def record_stat(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
        verbose: int | None = None,
        format_str: str = "{0:.3f}",
    ):
        """Record statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        value : Any
            Value that should be recorded.

        episode : int, optional
            Episode which we record the statistic.

        step : int, optional
            Step at which we record the statistic.

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        if key not in self.stats:
            self.stats_loc[key] = []
            self.stats[key] = []
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        self.stats_loc[key].append((episode, step, t))
        self.stats[key].append(value)

    def get_stat(self, key: str, x_key="episode"):
        """Get statistics.

        Parameters
        ----------
        key : str
            The name of the statistic.

        x_key : str in ['episode', 'step', 'time'], optional
            x-values.

        Returns
        -------
        x : array, shape (n_measurements,)
            Either episodes or steps at recorded value.

        y : array, shape (n_measurements,)
            Requested statistics.
        """
        assert key in self.stats
        X_KEYS = ["episode", "step", "time"]
        assert x_key in X_KEYS
        x_idx = X_KEYS.index(x_key)
        x = np.asarray(list(map(lambda x: x[x_idx], self.stats_loc[key])))
        y = np.asarray(self.stats[key])
        return x, y

    def define_checkpoint_frequency(self, key: str, checkpoint_interval: int):
        """Define the checkpoint frequency for a function approximator.

        Parameters
        ----------
        key : str
            The name of the function approximator.

        checkpoint_interval : int
            Number of steps after which the function approximator should be
            saved.
        """
        """Does nothing."""

    def record_epoch(
        self,
        key: str,
        value: Any,
        episode: int | None = None,
        step: int | None = None,
        t: float | None = None,
    ):
        """Does nothing."""
