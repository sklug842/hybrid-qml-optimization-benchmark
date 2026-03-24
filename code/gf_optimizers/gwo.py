# Import NumPy for vectorized population initialization, random sampling, and array algebra
import numpy                         as     np
# Import tqdm to provide a progress bar over optimizer generations
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (stores wrapper and budget settings)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define the Grey Wolf Optimizer (GWO) as a gradient-free optimizer
class GWO(AbstractOptimizer_GF):
    """
    Grey Wolf Optimizer (GWO) for gradient-free training of parametrized
    quantum circuits.

    This class implements a Grey Wolf Optimizer inspired by the original
    formulation of Mirjalili et al. (2014) and adapted to the variational
    quantum circuit setting following Majumder et al. (2020). Each
    candidate solution is interpreted as the position of a wolf in
    parameter space, and the pack collectively searches for low-loss
    configurations by mimicking the hunting behavior and social hierarchy
    of grey wolves.

    Algorithm overview
    ------------------
    Let each individual be a real-valued parameter vector of dimension
    ``ind_size`` (given by ``wrapper.get_ind_size()``). At each
    generation:

      1. **Evaluation (training set)**:
         All wolves in ``population`` are evaluated on the training set
         via

         ``wrapper.evaluate_single_final(X_data=wrapper.X_train,
                                         y_labels=wrapper.y_train,
                                         individual=theta)[0]``.

         Lower values are assumed to correspond to better performance
         (fitness).

      2. **Leadership hierarchy**:
         Wolves are sorted by fitness. The three best individuals define
         the leaders:
           * ``pos_alpha`` – best solution found so far,
           * ``pos_beta``  – second best,
           * ``pos_delta`` – third best.

      3. **Coefficient update**:
         A linearly decreasing coefficient

         ``a = 2 - 2 * (generation - 1) / n_iterations``

         controls the balance between exploration and exploitation.
         Random coefficient vectors ``A_k`` and ``C_k`` (for
         k ∈ {1, 2, 3}) are drawn as in the standard GWO update:

         ``A_k = 2 * a * U(0, 1) - a``,  ``C_k = U(0, 2)``,

         where `U` is a uniform random vector of length ``ind_size``.

      4. **Position update**:
         For each wolf, three candidate positions relative to the
         leaders are computed using :meth:`equations`:

         ``X_k = pos_leader_k - A_k * |C_k * pos_leader_k - pos_wolf|``.

         The new position of the wolf is set to the average of the three
         candidate positions:

         ``population[x] = (X1 + X2 + X3) / 3``,

         and then clipped elementwise to the interval
         ``[x_min, x_max]``.

      5. **Epoch logging (test set)**:
         At generations indicated by ``marker`` (derived from
         ``n_evals_per_epoch`` and ``n_epochs``), the current population
         is re-evaluated on the training set, the best individual
         (lowest training loss) is selected, and its **test** accuracy
         is computed via

         ``wrapper.evaluate_single_final(X_data=wrapper.X_test,
                                         y_labels=wrapper.y_test,
                                         individual=theta_best)[1]``.

         This accuracy is appended to ``accuracies``.

    After all iterations, the ``run`` method checks that exactly
    ``n_epochs`` accuracies were recorded and returns the list.

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
        ``n_epochs`` and ``pop_size`` this determines the number of GWO
        generations and the spacing of logging points.
    n_epochs : int
        Number of epochs (i.e., test-accuracy snapshots) to record.
    pop_size : int
        Number of wolves (candidate solutions) in the pack. Larger
        values increase search diversity but also raise the computational
        cost per generation.
    x_min : float
        Lower bound of the search space for each parameter. All initial
        positions and updates are clipped elementwise to be at least
        ``x_min``.
    x_max : float
        Upper bound of the search space for each parameter. All initial
        positions and updates are clipped elementwise to be at most
        ``x_max``.

    Returns
    -------
    list[float]
        Test accuracies of the best wolf (according to training loss) at
        each epoch. The list has length ``n_epochs``.
    """

    # Initialize GWO hyperparameters and derive iteration/logging schedule from evaluation budget
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, pop_size, x_min, x_max) -> None:
        # Initialize base optimizer state (wrapper, evaluation budget per epoch, number of epochs)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Number of wolves (candidate solutions) in the pack
        self.pop_size     = pop_size
        # Lower clipping bound for each parameter
        self.x_min        = x_min
        # Upper clipping bound for each parameter
        self.x_max        = x_max

        # Dimensionality of an individual (number of parameters)
        self.ind_size     = self.wrapper.get_ind_size()
        # Number of GWO generations given the evaluation budget (each generation evaluates pop_size individuals)
        self.n_iterations = (n_evals_per_epoch*n_epochs)//pop_size

        # Generation indices at which to log epoch-level test accuracy
        self.marker       = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // pop_size
        # Ensure the final marker aligns with the final generation index
        self.marker[-1]   = self.n_iterations


    # Compute the GWO position update component relative to a given leader position
    def equations(self, A, C, pos, member):
        
        # Distance vector between the wolf and the leader (scaled by C), elementwise absolute value
        D = np.abs(C * pos - member)

        # Candidate position update: leader position minus A-scaled distance
        # (matches standard GWO form X = leader - A * D)
        X = pos - A * D

        # Return the candidate position for this leader
        return X


    # Run the GWO loop: evaluate pack, update alpha/beta/delta, move wolves, and log test accuracy per marker
    def run(self):

        # Initialize wolf positions uniformly at random within [x_min, x_max] for each parameter
        population  = np.random.uniform(low  = self.x_min                    ,
                                        high = self.x_max                    ,
                                        size = (self.pop_size, self.ind_size))
        
        # Initialize list for epoch-level test accuracies
        accuracies  = []

        # Create a progress bar over the number of GWO generations
        pbar        = tqdm(range(1, self.n_iterations+1))
        for generation in pbar:
            # Update progress bar label with current generation
            pbar.set_description(f"Currently running generation {generation}")

            # Evaluate training loss ("fitness") for each wolf in the current population
            fitness                        = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                                 y_labels   = self.wrapper.y_train,
                                                                                 individual = individual          )[0] for individual in population]
            
            # Sort indices by ascending fitness (lower loss is better)
            sorted_indices                 = np.argsort(fitness)

            # Select the three best wolves as alpha, beta, and delta leaders
            # (alpha: best, beta: second, delta: third)
            pos_alpha, pos_beta, pos_delta = population[sorted_indices[0:3]]

            # Linearly decreasing control parameter a in [2, 0] over iterations:
            # larger a => more exploration; smaller a => more exploitation
            a                              = 2 - (2 * ((generation-1) / (self.n_iterations)))


            # Update each wolf position based on alpha/beta/delta influences
            for x in range(self.pop_size):

                # Sample A coefficient vectors for alpha/beta/delta:
                # A = 2*a*r - a with r ~ U(0,1), yielding A in [-a, a]
                A1, A2, A3                 = 2 * a * np.random.uniform(low  = 0                ,
                                                                       high = 1                ,
                                                                       size = (3, self.ind_size)) - a
            
                # Sample C coefficient vectors for alpha/beta/delta:
                # C ~ U(0,2) elementwise
                C1, C2, C3                 =         np.random.uniform(low  = 0                ,
                                                                       high = 2                ,
                                                                       size = (3, self.ind_size))
  
                # Compute candidate positions w.r.t. alpha leader
                X1                         = self.equations(A1, C1, pos_alpha, population[x])
                # Compute candidate positions w.r.t. beta leader
                X2                         = self.equations(A2, C2, pos_beta, population[x])
                # Compute candidate positions w.r.t. delta leader
                X3                         = self.equations(A3, C3, pos_delta, population[x])
 
                # Set wolf position to the average of the three candidates and clip to bounds
                population[x]              = np.clip((X1 + X2 + X3) / 3, self.x_min, self.x_max)
            

            # If this generation is a logging marker, compute and store epoch-level test accuracy
            if generation in self.marker:

                # Re-evaluate population on training set to identify the current best wolf
                evaluations = [   self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train              ,
                                                                     y_labels   = self.wrapper.y_train              ,
                                                                     individual = individual                        )[0] for individual in population]

                # Evaluate the best-by-training-loss individual on the test set and append its accuracy
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test               ,
                                                                     y_labels   = self.wrapper.y_test               ,
                                                                     individual = population[np.argmin(evaluations)])[1])


        # Sanity check: should have exactly one accuracy value per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit() 
        
        # Return epoch-level test accuracies
        return accuracies