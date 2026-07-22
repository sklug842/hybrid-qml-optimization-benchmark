# Import the minimization routine used to run COBYLA
from scipy.optimize import minimize
# Import garbage collection utilities for releasing large model objects
import gc
# Import NumPy for numerical operations, random sampling, and vectorized array manipulation
import numpy as np
# Import the abstract base class for gradient-free optimizers (provides wrapper/epoch bookkeeping)
from help_classes.abstract_classes import AbstractOptimizer_GF


# Define the COBYLA baseline optimizer as a gradient-free method
class COBYLA(AbstractOptimizer_GF):
    """
    Gradient-free optimizer based on COBYLA (Constrained Optimization BY Linear Approximations).

    This implementation adapts COBYLA to the hybrid QML setting by optimizing the
    flattened model-parameter vector through objective-function evaluations only.
    The optimizer itself does not use gradients. Instead, the objective function
    evaluates the training loss for a given parameter vector, and COBYLA uses these
    function values to propose new candidate points.

    Algorithm overview
    ------------------
    Let the flattened parameter vector be given by ``x0``. The optimizer proceeds
    for ``n_iterations = n_epochs * n_evals_per_epoch`` objective evaluations.

      1. **Initialization**:
         The hybrid model is initialized and converted into a flattened parameter
         vector ``x0``. This vector serves as the starting point of the COBYLA run.

      2. **Objective evaluation**:
         For each candidate parameter vector ``individual``, the training loss is
         computed on the training set via
         ``wrapper.evaluate_single_final(X_data=wrapper.X_train, y_labels=wrapper.y_train, individual=individual)[0]``.

      3. **Best-solution tracking**:
         The currently best parameter vector is updated whenever a lower training
         loss is observed.

      4. **Epoch logging**:
         Every ``n_evals_per_epoch`` objective evaluations, the current best
         parameter vector is evaluated on the test set and the resulting accuracy is
         appended to ``accuracies``.

      5. **COBYLA optimization**:
         The SciPy ``minimize`` routine is run with method ``COBYLA`` and a maximum
         iteration budget of ``n_iterations``.

    After completion, the method returns the collected test accuracies.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must implement:
          * ``get_model()``:
              Return the initialized hybrid model.
          * ``get_individual(weights)``:
              Convert model weights into a flattened parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)``
             -> (loss, accuracy):
              Evaluate a single parameter vector on the given data.
    n_evals_per_epoch : int
        Number of objective evaluations per epoch. Together with
        ``n_epochs`` this determines the total number of COBYLA iterations.
    n_epochs : int
        Number of epochs (i.e., test-accuracy snapshots). The total number of
        objective evaluations is ``n_epochs * n_evals_per_epoch``.
    rhobeg : float
        Initial trust-region radius passed to COBYLA.
    tol : float
        Termination tolerance passed to COBYLA.

    Returns
    -------
    list[float]
        Test accuracies recorded after each epoch. The list has length
        ``n_epochs``.
    """

    # Constructor: stores hyperparameters and derives the total iteration budget
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, rhobeg, tol=1e-8) -> None:
        # Initialize base class (stores wrapper, n_evals_per_epoch, n_epochs, and sets wrapper.flag_GF=True)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Initial trust-region radius for COBYLA
        self.rhobeg = rhobeg
        # Termination tolerance for COBYLA
        self.tol    = tol

        # Total number of objective evaluations allowed in the run
        self.n_iterations = self.n_epochs * self.n_evals_per_epoch

    # Main optimization routine: runs COBYLA and logs the test accuracy after each epoch
    def run(self):
        """
        Runs the COBYLA baseline on the hybrid QML model.

        :return: returns the recorded test accuracies across epochs
        """

        # Initialize the model and extract the flattened parameter vector
        model = self.wrapper.get_model()
        x0    = self.wrapper.get_individual(model.weights)

        # Initialize the list that stores the test accuracies recorded during the run
        accuracies   = []

        # Initialize the objective-evaluation counter and the next logging marker
        eval_counter = 0
        next_marker  = self.n_evals_per_epoch

        # Track the best parameter vector seen so far according to training loss
        best_individual = np.copy(x0)
        best_loss       = np.inf

        # Define the objective passed to COBYLA
        def objective(individual):
            # Make the outer variables visible inside the nested objective function
            nonlocal eval_counter, next_marker, best_individual, best_loss

            # Evaluate the current candidate on the training set
            loss, _ = self.wrapper.evaluate_single_final(
                X_data=self.wrapper.X_train,
                y_labels=self.wrapper.y_train,
                individual=individual
            )

            # Convert the loss to a Python float for consistent comparisons
            loss = float(loss)

            # Increase the objective-evaluation counter
            eval_counter += 1

            # Update the best-so-far individual if the current candidate improves the loss
            if loss < best_loss:
                best_loss = loss
                best_individual = np.copy(individual)

            # After each epoch-equivalent number of evaluations, record test accuracy
            if eval_counter >= next_marker and len(accuracies) < self.n_epochs:
                _, acc = self.wrapper.evaluate_single_final(
                    X_data=self.wrapper.X_test,
                    y_labels=self.wrapper.y_test,
                    individual=best_individual
                )
                accuracies.append(acc)
                next_marker += self.n_evals_per_epoch

            # Return the training loss to COBYLA
            return loss

        # Run COBYLA on the objective starting from the initial parameter vector
        result = minimize(
            fun=objective,
            x0=x0,
            method="COBYLA",
            options={
                "maxiter": self.n_iterations,
                "rhobeg": self.rhobeg,
                "tol": self.tol,
                "disp": False
            }
        )

        # Release the model object and trigger garbage collection
        del model
        gc.collect()

        # Check whether the expected number of epoch snapshots was collected
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies) != n_epochs")
            print("len(accuracies): ", len(accuracies))
            print("n_epochs:        ", self.n_epochs)
            print("COBYLA nfev:     ", result.nfev if hasattr(result, "nfev") else "unknown", "\n")

        # Return the recorded test accuracies
        return accuracies
