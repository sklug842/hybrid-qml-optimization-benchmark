import abc

class AbstractWrapper():
    """
    Abstract base class for dataset- and model-specific wrappers.

    A concrete wrapper is responsible for:
      * storing training and test data (X_train, y_train, X_test, y_test),
      * defining how an "individual" (a flattened parameter vector) is mapped
        to a model and evaluated on given data,
      * exposing metadata needed by optimizers (e.g., batch size, model size).

    Attributes
    ----------
    flag_GF : bool
        Flag indicating whether the current optimization is gradient-free
        (True) or gradient-based (False). This is typically set by the
        optimizer subclasses (e.g., AbstractOptimizer_GF / AbstractOptimizer_GB)
        to allow the wrapper to adapt its behaviour (e.g., how BatchNorm
        parameters are treated).
    """

    def __init__(self) -> None:
        """
        Initialize the wrapper with default settings.
        """
        self.flag_GF = False
    
    @abc.abstractmethod
    def evaluate_single_final(self, X_data, y_labels, individual: list[float]) -> float:
        """
        Evaluate a single candidate solution (individual) on given data.

        Parameters
        ----------
        X_data :
            Input features on which the model defined by `individual` is
            evaluated.
        y_labels :
            Corresponding ground-truth labels for `X_data`.
        individual : list[float]
            Flattened parameter vector representing a model instance.

        Returns
        -------
        float
            Objective value for the given individual on (X_data, y_labels),
            e.g. a loss such as binary cross-entropy.
        """

class AbstractOptimizer():

    def __init__(self, wrapper: AbstractWrapper, n_evals_per_epoch: int, n_epochs: int) -> None:
        """
        Abstract base class for all optimizers.

        This class stores common configuration such as the wrapped problem
        instance and the evaluation budget, and defines the abstract `run`
        interface that all concrete optimizers must implement.

        Parameters
        ----------
        wrapper : AbstractWrapper
            Problem-specific wrapper providing data and evaluation routines for
            candidate solutions.
        n_evals_per_epoch : int
            Number of model evaluations that conceptually correspond to one
            "epoch" of optimization (used for logging / scheduling).
        n_epochs : int
            Total number of optimization epochs to perform.
        """

        self.wrapper           = wrapper
        self.n_evals_per_epoch = n_evals_per_epoch
        self.n_epochs          = n_epochs

    @abc.abstractmethod
    def run(self) -> float:
        """
        Execute the optimization routine.

        Returns
        -------
        float
            Final evaluation value of the best solution found (e.g., accuracy
            or loss on a held-out test set). Concrete subclasses may extend
            this to return richer statistics (e.g., a list of per-epoch
            scores).
        """


class AbstractOptimizer_GB(AbstractOptimizer):

    def __init__(self, wrapper: AbstractWrapper, n_evals_per_epoch: int, n_epochs: int, optimizer: str, eta: float, beta_1: float, beta_2: float, momentum: float) -> None:
        """
        Abstract base class for gradient-based optimizers.

        This subclass extends `AbstractOptimizer` with additional hyperparameters
        used by gradient-based methods such as SGD or Adam, and ensures that the
        associated wrapper is marked as gradient-based (`flag_GF = False`).

        Parameters
        ----------
        wrapper : AbstractWrapper
            Problem-specific wrapper providing data and evaluation routines.
        n_evals_per_epoch : int
            Number of effective evaluations per epoch (may be unused for purely
            gradient-based training loops, but kept for consistency).
        n_epochs : int
            Total number of training epochs.
        optimizer : str
            Name of the underlying gradient-based optimizer (e.g., "SGD" or "Adam").
        eta : float
            Learning rate used by the underlying optimizer.
        beta_1 : float
            First-moment decay rate (e.g. for Adam).
        beta_2 : float
            Second-moment decay rate (e.g. for Adam).
        momentum : float
            Momentum factor (e.g. for SGD with momentum).
        """

        super().__init__(wrapper, n_evals_per_epoch, n_epochs)

        self.optimizer       = optimizer
        self.eta             = eta
        self.beta_1          = beta_1
        self.beta_2          = beta_2
        self.momentum        = momentum
        self.wrapper.flag_GF = False


class AbstractOptimizer_GF(AbstractOptimizer):

    def __init__(self, wrapper: AbstractWrapper, n_evals_per_epoch: int, n_epochs: int) -> None:
        """
        Abstract base class for gradient-free (derivative-free) optimizers.

        This subclass distinguishes gradient-free algorithms such as evolutionary
        strategies, genetic algorithms, swarm methods, etc., and marks the
        associated wrapper as gradient-free (`flag_GF = True`). The wrapper can
        use this information to adapt its parameterization (for example, how
        Batch Normalization statistics are treated).

        Parameters
        ----------
        wrapper : AbstractWrapper
            Problem-specific wrapper providing data and evaluation routines.
        n_evals_per_epoch : int
            Number of model evaluations allotted per optimization epoch.
        n_epochs : int
            Total number of epochs (evaluation blocks) to perform.
        """

        super().__init__(wrapper, n_evals_per_epoch, n_epochs)

        self.wrapper.flag_GF   = True