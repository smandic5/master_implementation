import abc
import atexit
import contextlib
import time
from typing import Any

try:
    import aim
except ImportError:
    aim = None


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
        
        
class AIMLogger(LoggerBase):
    """Use AIM to log experiment statistics.

    Parameters
    ----------
    step_counter : str, one of ['episode', 'step', 'time'], optional
        Define which value should be used as a step counter.

    log_system_params : bool, optional
        Log system parameters, e.g. memory and CPU consumption.
    """

    counter_idx: int
    log_system_params: bool
    start_time: float
    hparams: dict | None
    _n_episodes: int
    n_steps: int

    def __init__(
        self, step_counter: str = "step", log_system_params: bool = False
    ):
        if aim is None:
            raise ImportError(
                "Aim is required to use this logger, but is not installed."
            )
        assert step_counter in ["episode", "step", "time"]
        self.counter_idx = ["episode", "step", "time"].index(step_counter)
        self.log_system_params = log_system_params
        self.run = None
        self.start_time = 0.0
        self.hparams = None
        self._n_episodes = 0
        self.n_steps = 0

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
        self.run = aim.Run(
            experiment=f"{env_name}-{algorithm_name}",
            log_system_params=self.log_system_params,
        )
        atexit.register(self.run.close)
        self.run["hparams"] = hparams if hparams is not None else {}

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
            Episode which we record the statistic. Will be mapped to epochs
            in the AIM run.

        step : int, optional
            Step at which we record the statistic.

        t : float, optional
            Wallclock time, measured with time.time().

        verbose : int, optional
            Overwrite verbosity level.

        format_str : str, optional
            Format string for stdout logging.
        """
        if episode is None:
            episode = self._n_episodes
        if step is None:
            step = self.n_steps
        if t is None:
            t = time.time() - self.start_time
        s = [episode, step, t][self.counter_idx]
        with contextlib.suppress(TypeError):
            value = float(value)
        self.run.track(value=value, name=key, step=s, epoch=episode)


