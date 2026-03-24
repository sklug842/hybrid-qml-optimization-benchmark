# Import NumPy for sampling perturbations, population generation, and vectorized NES updates
import numpy as np
# Import tqdm to show a progress bar over NES iterations
from   tqdm import tqdm
# Import the abstract base class for gradient-free optimizers (stores wrapper and evaluation budget)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define the Natural Evolution Strategy optimizer as a gradient-free optimizer
class NES(AbstractOptimizer_GF):
    """
    Natural Evolution Strategy (NES) optimizer for variational quantum circuits.

    This class implements a simple Gaussian Natural Evolution Strategy in the
    spirit of Wierstra et al. (2014) [*Natural Evolution Strategies*, JMLR 15],
    adapted to the gradient-free training of variational quantum circuits via
    the provided ``wrapper`` interface.

    The algorithm maintains a multivariate **isotropic Gaussian search
    distribution** over circuit parameters,
    :math:`\theta \\sim \\mathcal{N}(\\mu, \\sigma^2 I)`, and iteratively updates
    the mean ``mu`` and global standard deviation ``sigma`` using a
    rank-based natural-gradient estimate derived from sampled perturbations.

    Algorithm overview
    ------------------
    Let ``ind_size`` be the dimensionality of the parameter vector (obtained
    from ``wrapper.get_ind_size()``). For each NES iteration:

      1. **Sampling**:
         Draw perturbations :math:`\\epsilon_i \\sim \\mathcal{N}(0, I)` and
         construct a population of candidate parameter vectors
         :math:`\\theta_i = \\mu + \\sigma \\epsilon_i` of size ``pop_size``.

      2. **Evaluation (training set)**:
         Each candidate is evaluated on the training set via

         ``eval_i = wrapper.evaluate_single_final(
                        X_data=wrapper.X_train,
                        y_labels=wrapper.y_train,
                        individual=theta_i
                    )[0]``

         where the first return value is assumed to be a **loss** (lower is
         better).

      3. **Rank-based utilities**:
         The evaluations are converted into **linear ranking weights** using
         ``compute_linear_rank_weights``: the best (lowest-loss) candidate
         receives the largest positive weight and the worst the smallest,
         then the weights are centered to have zero mean. This mitigates the
         influence of outliers and makes the update invariant to affine
         transformations of the loss.

      4. **Natural-gradient update**:

         *Mean update*:
         The gradient estimate for the mean is obtained from the weighted
         perturbations:

         ``grad_mu = (linear_weights @ epsilons) / sigma``

         and the mean is updated as

         ``mu += mu_eta * sigma * grad_mu``,

         where ``mu_eta`` is the learning rate for the mean.

         *Scale (sigma) update*:
         A scalar gradient estimate for the global standard deviation is
         computed from the squared perturbations:

         ``grad_sigma = mean(linear_weights * (epsilons**2 - 1))``

         and the scale is updated multiplicatively:

         ``sigma *= exp((sigma_eta / 2) * grad_sigma)``,

         where ``sigma_eta`` is the learning rate for the exploration scale.
         Afterwards, ``sigma`` is clipped to the interval ``[1e-3, 1e-1]`` to
         avoid premature collapse or unstable explosion of the search
         distribution.

      5. **Epoch logging (test set)**:
         The total number of iterations is
         ``n_iterations = (n_evals_per_epoch * n_epochs) // pop_size``.
         At specific iteration indices (stored in ``self.marker``), the
         algorithm evaluates the *current* population on the test set and
         records the accuracy of the best (lowest-loss) candidate:

         ``accuracy = wrapper.evaluate_single_final(
                          X_data=wrapper.X_test,
                          y_labels=wrapper.y_test,
                          individual=population[argmin(evaluations)]
                      )[1]``

         These accuracies are appended to ``accuracies`` once per epoch,
         yielding a trajectory of test performance over ``n_epochs``.

    After completion, the method checks that exactly ``n_epochs`` test
    accuracies were collected and returns them.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must implement:
          * ``get_ind_size() -> int``:
              Return the dimensionality of the parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``:
              Evaluate a single parameter vector on the given data and
              return a scalar loss (to be minimized) and a corresponding
              accuracy.
    n_evals_per_epoch : int
        Number of model evaluations per epoch. Together with
        ``n_epochs`` and ``pop_size`` this determines the number of NES
        iterations and the spacing of test-accuracy snapshots.
    n_epochs : int
        Number of epochs (i.e., times test accuracy is recorded). The
        total number of NES update steps is
        ``n_iterations = (n_evals_per_epoch * n_epochs) // pop_size``.
    pop_size : int
        Population size, i.e., the number of perturbations (and thus
        candidate solutions) sampled from the Gaussian search
        distribution in each iteration. Larger values improve the
        gradient estimate at higher computational cost.
    mu_eta : float
        Learning rate for the mean ``mu`` of the search distribution,
        controlling the step size of the natural-gradient update in
        parameter space.
    sigma_eta : float
        Learning rate for the global standard deviation ``sigma`` of the
        search distribution, determining how quickly the overall
        exploration scale adapts.
    mu_init : float
        Initialization range for the mean vector. Each component of
        ``mu`` is sampled uniformly from
        ``[-mu_init, mu_init]``, setting the initial center of the
        search in parameter space.
    sigma_init : float
        Initial standard deviation of the Gaussian search distribution.
        This sets the initial exploration radius around ``mu``.

    Returns
    -------
    list[float]
        Test accuracies of the best (lowest-loss) individual at the end
        of each epoch. The list has length ``n_epochs``.
    """

    # Store NES hyperparameters and derive iteration/logging schedule from evaluation budget
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, pop_size, mu_eta, sigma_eta, mu_init, sigma_init) -> None:
        # Initialize base optimizer state (wrapper, evaluation budget per epoch, number of epochs)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Number of samples (perturbations / candidates) per NES iteration
        self.pop_size     = pop_size
        # Learning rate for the mean update
        self.mu_eta       = mu_eta
        # Learning rate for the sigma (scale) update
        self.sigma_eta    = sigma_eta
        # Range parameter for initializing mu uniformly in [-mu_init, mu_init]
        self.mu_init      = mu_init
        # Initial global standard deviation of the search distribution
        self.sigma_init   = sigma_init

        # Number of NES iterations given the evaluation budget (each iteration evaluates pop_size candidates)
        self.n_iterations = (n_evals_per_epoch*n_epochs)//pop_size
        # Dimensionality of the parameter vector (number of trainable parameters)
        self.ind_size     = self.wrapper.get_ind_size()

        # Iteration indices at which to log epoch-level test accuracy
        self.marker       = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // pop_size
        # Ensure the final marker aligns with the final iteration index
        self.marker[-1]   = self.n_iterations


    # Compute centered linear ranking utilities from evaluation scores (best gets largest weight)
    def compute_linear_rank_weights(self, evaluations):
        """
        Compute linear ranking weights for a set of evaluations.
        
        Arguments:
            evaluations (np.array or list): A 1-D array of evaluation scores where a higher value
                                            means a better candidate. (If lower loss is better, 
                                            then use the negative loss as in your NES setup.)
        
        Returns:
            np.array: A 1-D array of centered ranking weights with zero mean.
        """
        # Convert evaluations to a NumPy array for consistent indexing and sorting
        evaluations           = np.array(evaluations)
        
        # Sort indices by ascending evaluation value (NOTE: in this implementation, lower is treated as "better")
        # If you want "higher is better", you would sort by -evaluations instead
        sorted_indices        = np.argsort(evaluations) # np.argsort(-evaluations)
        
        # Allocate array to store rank (position in the sorted order) for each candidate
        ranks                 = np.empty_like(sorted_indices)

        # Assign ranks: best candidate gets rank 0, worst gets rank pop_size - 1
        ranks[sorted_indices] = np.arange(len(evaluations))
        
        # Map ranks to linear weights in [0, 1]: best -> 1, worst -> 0
        linear_weights        = (len(evaluations) - 1 - ranks) / (len(evaluations) - 1)
        
        # Center weights so they sum (approximately) to zero, which stabilizes updates
        linear_weights       -= np.mean(linear_weights)
        
        # Return centered linear ranking weights (utilities)
        return linear_weights


    # Run the NES loop: sample perturbations, evaluate candidates, update mu/sigma, and log test accuracy per marker
    def run(self):

        # Initialize mean vector mu uniformly in [-mu_init, mu_init] for each parameter dimension
        mu              = np.random.uniform(low  = -self.mu_init,
                                            high =  self.mu_init,
                                            size = self.ind_size)

        # Initialize global standard deviation sigma (scalar) for the isotropic Gaussian search distribution
        sigma           = self.sigma_init

        # Initialize list for epoch-level test accuracies
        accuracies      = []

        # Create a progress bar over NES iterations
        pbar            = tqdm(range(1, self.n_iterations+1))
        for generation in pbar:
            # Update progress bar label with current iteration number
            pbar.set_description(f"Currently running generation {generation}")

            # Sample standard-normal perturbations epsilon ~ N(0, I)
            epsilons    = np.random.normal(loc   = 0                             ,
                                           scale = 1                             ,
                                           size  = (self.pop_size, self.ind_size))

            # Construct candidate population: theta_i = mu + sigma * epsilon_i
            population     = mu + sigma * epsilons
            
            # Evaluate each candidate on the training set (first return value interpreted as loss to minimize)
            evaluations    = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                 y_labels   = self.wrapper.y_train,
                                                                 individual = individual          )[0] for individual in population]
            
            # Convert evaluations into centered rank-based utilities (best -> largest utility)
            linear_weights = self.compute_linear_rank_weights(evaluations)

            # Estimate natural-gradient direction for mu using weighted perturbations
            grad_mu        = np.dot(linear_weights, epsilons) / sigma
            # Update mu (note: mu_eta * sigma rescales step into parameter space units)
            mu            += self.mu_eta * sigma * grad_mu

            # Estimate scalar gradient for sigma using squared perturbations (eps^2 - 1 term)
            grad_sigma     = np.mean(linear_weights.reshape(-1, 1) * (epsilons ** 2 - 1))
            # Update sigma multiplicatively to keep sigma positive
            sigma         *= np.exp((self.sigma_eta / 2) * grad_sigma)

            # Clip sigma to avoid collapse (too small) or instability (too large)
            sigma          = np.clip(sigma, 1e-3, 1e-1)


            # If this iteration is a logging marker, evaluate best candidate on the test set and append accuracy
            if generation in self.marker:
                
                # Select best candidate by lowest training evaluation (loss) and record its test accuracy
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test               ,
                                                                     y_labels   = self.wrapper.y_test               ,
                                                                     individual = population[np.argmin(evaluations)])[1])


        # Sanity check: should have exactly one recorded accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit()

        # Return epoch-level test accuracies
        return accuracies