# Import NumPy for sampling from distributions, statistics (mean/std), clipping, and array ops
import numpy                         as     np
# Import tqdm to show a progress bar over generations
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (handles wrapper/epoch bookkeeping)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define the Cross-Entropy Method optimizer as a gradient-free optimizer
class CEM(AbstractOptimizer_GF):
    """
    Cross-Entropy Method (CEM) optimizer for gradient-free training of
    parametrized quantum circuits.

    This class implements a Gaussian Cross-Entropy Method tailored to
    variational quantum models. At each iteration, a population of
    parameter vectors is sampled from a multivariate Gaussian search
    distribution, evaluated on the training set, and the top
    ``best_size`` (elite) individuals are used to update the mean and
    standard deviation of the distribution. Over time, the Gaussian
    narrows around regions of low training loss.

    Algorithm overview
    ------------------
    Let each individual be a real-valued parameter vector of dimension
    ``ind_size`` (given by ``wrapper.get_ind_size()``). The algorithm
    proceeds for ``n_iterations`` generations:

      1. **Sampling**:
         Sample ``pop_size`` candidates from a Gaussian distribution
         with mean ``mu`` and standard deviation ``sigma`` using
         :meth:`generate`. Each sampled vector is clipped elementwise
         to the interval ``[-x_min_max, x_min_max]``.

      2. **Evaluation (training set)**:
         For every individual, compute its loss via
         ``wrapper.evaluate_single_final(X_data=wrapper.X_train,
                                         y_labels=wrapper.y_train,
                                         individual=theta)[0]``.
         Lower values are assumed to correspond to better performance.

      3. **Elite selection and distribution update**:
         Sort individuals by training loss and select the best
         ``best_size`` to form the elite set. Compute the elite-wise
         mean and standard deviation, then update ``mu`` and ``sigma``
         with an exponential moving average controlled by ``beta``:
         new parameters are a convex combination of old and elite-based
         estimates. A minimum standard deviation is enforced to avoid
         premature collapse of the distribution.

      4. **Epoch logging (test set)**:
         At generations indicated by ``marker`` (derived from
         ``n_evals_per_epoch`` and ``n_epochs``), the current
         population is re-evaluated on the training set, the best
         individual (lowest training loss) is selected, and its
         **test** accuracy is computed via
         ``wrapper.evaluate_single_final(X_data=wrapper.X_test,
                                         y_labels=wrapper.y_test,
                                         individual=theta_best)[1]``.
         This accuracy is appended to ``accuracies``.

    The ``run`` method terminates once all generations have been
    processed, checks that exactly ``n_epochs`` test accuracies were
    recorded, and then returns the list of accuracies.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must implement:
          * ``get_ind_size() -> int``:
              Return the dimensionality of the parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``:
              Evaluate a single parameter vector on the given data.
    n_evals_per_epoch : int
        Target number of model evaluations per epoch. Together with
        ``n_epochs`` and ``pop_size`` this determines the number of
        CEM generations and the spacing of logging points.
    n_epochs : int
        Number of epochs (i.e., test-accuracy snapshots) to record.
    pop_size : int
        Number of candidates sampled from the Gaussian search
        distribution per generation.
    mu_init : float or array-like
        Initial mean of the Gaussian search distribution. In the common
        case this is a scalar (e.g., 0) that is broadcast to all
        parameters.
    sigma_init : float or array-like
        Initial standard deviation of the Gaussian search distribution.
        Sets the initial exploration scale around ``mu_init``.
    best_size : int
        Number of elite individuals (with lowest training loss) used to
        update the Gaussian parameters at each generation.
    beta : float
        Smoothing factor in ``[0, 1]`` controlling how strongly the
        distribution tracks the current elite set. Larger values make
        updates more responsive (less inertia), while smaller values
        slow adaptation.
    x_min_max : float
        Symmetric absolute bound for each parameter. All sampled
        individuals are clipped elementwise to
        ``[-x_min_max, x_min_max]``.

    Returns
    -------
    list[float]
        Test accuracies of the best individual (according to training
        loss) at each epoch. The list has length ``n_epochs``.
    """

    # Constructor: store CEM hyperparameters and precompute iteration/logging indices
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, pop_size, mu_init, sigma_init, best_size, beta, x_min_max) -> None:
        # Initialize base class (stores wrapper, n_evals_per_epoch, n_epochs, and sets wrapper.flag_GF=True)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Number of candidates sampled per generation
        self.pop_size     = pop_size
        # Initial Gaussian mean (scalar or vector); used to seed the search distribution
        self.mu_init      = mu_init
        # Initial Gaussian standard deviation (scalar or vector); sets initial exploration scale
        self.sigma_init   = sigma_init
        # Number of elite (best) candidates used to update the search distribution each generation
        self.best_size    = best_size
        # EMA smoothing factor for distribution updates (beta close to 1 updates aggressively)
        self.beta         = beta
        # Symmetric clipping bound for every parameter dimension
        self.x_min_max    = x_min_max
        
        # Total number of generations implied by evaluation budget and population size
        self.n_iterations = (n_evals_per_epoch*n_epochs)//pop_size
         
        # Dimensionality of individuals (number of trainable parameters)
        self.ind_size     = self.wrapper.get_ind_size()
        # Generation indices at which to record epoch-level test accuracy
        self.marker       = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // pop_size
        # Force the last marker to be exactly the last generation index
        self.marker[-1]   = self.n_iterations


    # Sample a population from N(mu, sigma^2) and clip to [-x_min_max, x_min_max]
    def generate(self, mu, sigma):
        
        # Draw pop_size individuals of length ind_size from a (diagonal) Gaussian distribution
        population        = np.random.normal(loc   = mu                            ,
                                             scale = sigma                         ,
                                             size  = (self.pop_size, self.ind_size))
        
        # Enforce parameter bounds by clipping each coordinate to the symmetric range
        return np.clip(a     =  population    ,
                       a_min = -self.x_min_max,
                       a_max =  self.x_min_max)
    

    # Update Gaussian parameters using elite statistics with exponential smoothing and a sigma floor
    def update(self, population_best, mu, sigma):
        
        # Compute the new mean from the elite individuals (coordinate-wise)
        new_mu    = np.mean(population_best, axis=0)
        # Compute the new std from elites (coordinate-wise) and add epsilon for numerical stability
        new_sigma = np.std( population_best, axis=0) + 1e-8  # Avoid division by zero.
        
        # Smooth mean update: convex combination of elite mean and previous mean
        mu        = self.beta*new_mu    + (1-self.beta)*mu
        # Smooth std update: convex combination of elite std and previous std
        sigma     = self.beta*new_sigma + (1-self.beta)*sigma
        
        # Prevent sigma from collapsing to ~0 (maintains exploration capability)
        sigma     = np.maximum(sigma, 1e-2)
        
        # Return updated distribution parameters
        return mu, sigma

    
    # Main optimization routine: sample/evaluate/update iteratively and log test accuracy at markers
    def run(self):

        # Initialize Gaussian mean from user-provided starting value
        mu              = self.mu_init
        # Initialize Gaussian standard deviation from user-provided starting value
        sigma           = self.sigma_init

        # Initialize the population by sampling from the initial Gaussian
        population      = self.generate(mu    = mu   ,
                                        sigma = sigma)
        
        # Initialize array for accuracies (one entry per epoch marker)
        accuracies      = []

        # Start generational process  
        pbar            = tqdm(range(1, self.n_iterations+1))
        for generation in pbar:
            pbar.set_description(f"Currently running generation {generation}")

            # Evaluate population on training set (collect loss values)
            evaluations = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                              y_labels   = self.wrapper.y_train,
                                                              individual = individual          )[0] for individual in population]
            
            # Select elites (indices sorted by loss ascending) and update Gaussian parameters
            mu, sigma   = self.update(population_best = [population[i] for i in np.argsort(evaluations)[0:self.best_size]],
                                      mu              = mu                                                                ,
                                      sigma           = sigma                                                             )

            # Sample a new population from the updated Gaussian
            population  = self.generate(mu    = mu   ,
                                        sigma = sigma)

            # At epoch markers: pick best-by-training-loss and record its test accuracy
            if generation in self.marker:
                    
                # Re-evaluate population on training data to determine the best individual at this snapshot
                evaluations = [   self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train              ,
                                                                     y_labels   = self.wrapper.y_train              ,
                                                                     individual = individual                        )[0] for individual in population]

                # Evaluate best individual on test data and append its accuracy
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test               ,
                                                                     y_labels   = self.wrapper.y_test               ,
                                                                     individual = population[np.argmin(evaluations)])[1])

        # Sanity check: ensure we recorded exactly one accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit()
        
        # Return collected test accuracies (length == n_epochs)
        return accuracies