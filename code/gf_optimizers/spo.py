# Import NumPy for population initialization, random perturbations (roots/runners), stacking, and clipping
import numpy                         as     np
# Import tqdm to show a progress bar over optimizer iterations
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (provides wrapper + eval budget bookkeeping)
from   help_classes.abstract_classes import AbstractOptimizer_GF

 
# Define the Strawberry Plant Optimizer as a gradient-free optimizer
class SPO(AbstractOptimizer_GF):
    """
    Strawberry Plant Optimizer (SPO) for gradient-free training of variational quantum circuits.

    This class implements the Strawberry Plant Optimizer as proposed by
    Merrikh-Bayat (2014) and parametrized following Wang et al. (2019), adapted
    to the gradient-free optimization interface provided by ``wrapper``. Each
    individual in the population represents a candidate parameter vector for a
    variational quantum circuit.

    Algorithm overview
    ------------------
    Let ``ind_size`` be the dimensionality of the parameter vector returned by
    ``wrapper.get_ind_size()``. At each iteration, the algorithm maintains a
    population of ``pop_size`` strawberry plants (candidate solutions) within
    the bounded search space ``[x_min, x_max]^{ind_size}`` and proceeds as:

      1. **Initialization**:
         * The initial population is sampled uniformly from
           ``[x_min, x_max]^{ind_size}``.

      2. **Propagation (roots and runners)**:
         For each plant in the current population:

           * A **root offspring** (local search) is generated as:
             ::
                 population_root = population
                     + d_root * U(-1, 1)^(pop_size × ind_size)

             where ``d_root`` controls the maximum local step size.

           * A **runner offspring** (global search) is generated as:
             ::
                 population_runner = population
                     + d_runner * U(-1, 1)^(pop_size × ind_size)

             where ``d_runner`` controls the maximum global step size and allows
             long-range moves to escape local minima.

         The two sets are concatenated into a combined population of size
         ``2 * pop_size`` and clipped to the search bounds:
         ::
             population_total = clip(population_total, x_min, x_max)

      3. **Fitness evaluation and selection**:
         * Each candidate in ``population_total`` is evaluated on the training
           data using:
           ::
               loss = wrapper.evaluate_single_final(
                   X_data=wrapper.X_train,
                   y_labels=wrapper.y_train,
                   individual=individual
               )[0]

         * **Elitist selection**:
           The best ``pop_size // 2`` candidates (lowest loss) are kept as
           ``population_n1``.

         * **Roulette-wheel selection**:
           A second set ``population_n2`` of size ``pop_size // 2`` is sampled
           from ``population_total`` with probabilities proportional to an
           inverse-rank scheme, so that better individuals are more likely to be
           selected but diversity is preserved.

         * The next generation is formed by stacking:
           ::
               population = vstack(population_n1, population_n2)

      4. **Epoch logging (test set)**:
         The total number of iterations is
         ``n_iterations = (n_evals_per_epoch * n_epochs) // (2 * pop_size)``,
         because each iteration evaluates a population of size ``2 * pop_size``.
         At iteration indices stored in ``self.marker``, the algorithm:

           * Re-evaluates the current population on the training set,
           * Selects the best individual (lowest loss),
           * Evaluates that individual on the test set and records its accuracy:
             ::
                 accuracy = wrapper.evaluate_single_final(
                     X_data=wrapper.X_test,
                     y_labels=wrapper.y_test,
                     individual=best_individual
                 )[1]

         These accuracies are stored once per epoch, giving a trajectory of test
         performance over ``n_epochs``.

    After all iterations, the method verifies that exactly ``n_epochs`` accuracy
    values have been collected and returns them.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must implement:
          * ``get_ind_size() -> int``:
                Return the dimensionality of the parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``:
                Evaluate a single candidate on the given data, returning a scalar
                loss (to be minimized) and a corresponding accuracy.
    n_evals_per_epoch : int
        Number of model evaluations per epoch. Together with ``n_epochs`` and
        ``pop_size``, determines the total number of SPO iterations and the
        spacing of test-accuracy snapshots.
    n_epochs : int
        Number of epochs (i.e., times test accuracy is recorded). The total
        number of SPO iterations is
        ``n_iterations = (n_evals_per_epoch * n_epochs) // (2 * pop_size)``.
    pop_size : int
        Number of strawberry plants (candidate solutions) in the population.
        Larger values increase search diversity and robustness at higher
        computational cost.
    x_min : float or array-like
        Lower bound of the search space for each parameter. Initial candidates
        and all propagated offspring are clipped not to fall below this value.
    x_max : float or array-like
        Upper bound of the search space for each parameter. Initial candidates
        and all propagated offspring are clipped not to exceed this value.
    d_root : float
        Maximum step size for local (root) propagation. Controls the radius of
        fine-grained, exploitative search around each parent solution.
    d_runner : float
        Maximum step size for global (runner) propagation. Controls the radius
        of exploratory, long-range moves allowing the optimizer to escape local
        minima.

    Returns
    -------
    list[float]
        Test accuracies of the best individual (lowest training loss) at the end
        of each epoch. The list has length ``n_epochs``.
    """

    # Initialize SPO hyperparameters and derive iteration/logging schedule from the evaluation budget
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, pop_size, x_min, x_max, d_root, d_runner) -> None: #oder lb_ub?
        # Initialize base optimizer state (wrapper, evaluations per epoch, epochs)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Number of strawberry plants (parents) maintained each iteration
        self.pop_size     = pop_size
        # Lower bound for each parameter dimension
        self.x_min        = x_min
        # Upper bound for each parameter dimension
        self.x_max        = x_max 
        # Local-search step size for "root" propagation
        self.d_root       = d_root
        # Global-search step size for "runner" propagation
        self.d_runner     = d_runner

        # Dimensionality of the parameter vector
        self.ind_size     = self.wrapper.get_ind_size()
        # Number of iterations implied by eval budget; each iteration evaluates 2*pop_size candidates
        self.n_iterations = (n_evals_per_epoch*n_epochs)//(pop_size*2)

        # Iteration indices at which to log epoch-level test accuracy
        self.marker       = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // (pop_size*2)
        # Ensure the last marker aligns with the final iteration
        self.marker[-1]   = self.n_iterations


    # Run the SPO loop: propagate roots/runners, select next population, and log test accuracy
    def run(self):

        # Initialize parent population uniformly in the bounded search space [x_min, x_max]
        # Population alternativ normalverteilt initialisieren: np.random.normal() anstatt x_min, x_max: sigma verwenden
        population                     = np.random.uniform(low  = self.x_min                    ,
                                                           high = self.x_max                    , 
                                                           size = (self.pop_size, self.ind_size))

        # Initialize list to store epoch-level test accuracies
        accuracies                     = []

        # Start generational process with a progress bar over iterations
        pbar = tqdm(range(1, self.n_iterations+1))
        for generation in pbar:
            # Update progress bar label
            pbar.set_description(f"Currently running generation {generation}")

            # Generate "root" offspring (local perturbations) around each parent
            # Random matrices with mxN
            population_root            = population + self.d_root  *np.random.uniform(low  = -1                            ,
                                                                                      high =  1                            ,
                                                                                      size = (self.pop_size, self.ind_size))
            
            # Generate "runner" offspring (global perturbations) around each parent
            population_runner          = population + self.d_runner*np.random.uniform(low  = -1                            ,
                                                                                      high =  1                            ,
                                                                                      size = (self.pop_size, self.ind_size))

            # Combine roots and runners into a candidate pool of size 2*pop_size
            population_total           = np.vstack((population_root, population_runner))

            # Enforce box constraints by clipping each parameter into [x_min, x_max]
            population_total           = np.clip(a     = population_total,
                                                 a_min = self.x_min      ,
                                                 a_max = self.x_max      )

            # Evaluate all candidates on training data (loss is first return value)
            population_total_fitness   = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                             y_labels   = self.wrapper.y_train,
                                                                             individual = individual          )[0] for individual in population_total]

        
            # Elite-selection: keep the best pop_size//2 candidates (lowest loss)
            population_n1              = population_total[np.argsort(population_total_fitness)[:self.pop_size//2]]

            # Compute ranks of candidates (lower loss -> better rank)
            # Lower fitness = lower rank
            rank                       = np.argsort(population_total_fitness) + 1

            # Convert ranks into scores so the best candidate receives the highest score
            # Best individual now gets highest score
            inverse_rank               = (2 * self.pop_size - rank + 1)

            # Normalize scores into a probability distribution for roulette-wheel selection
            # Define probabilities
            probs                      = inverse_rank / np.sum(inverse_rank)

            # Sample pop_size//2 additional candidates with replacement using roulette-wheel selection
            # Selected by roulette wheel
            population_n2              = population_total[np.random.choice(a       = np.arange(2*self.pop_size),
                                                                           replace = True                      , # nicht False?
                                                                           p       = probs                     ,  # mu/mu.sum()               ,
                                                                           size    = self.pop_size//2          )]
            
            # Form the next generation by stacking elites + roulette-selected individuals
            # Population
            population                 = np.vstack(tup = (population_n1, population_n2))

            # At logging markers, evaluate current population, pick the best on train, and record its test accuracy
            if generation in self.marker:

                # Evaluate populataion on training data to identify the current best individual
                evaluations = [   self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train              ,
                                                                     y_labels   = self.wrapper.y_train              ,
                                                                     individual = individual                        )[0] for individual in population]

                # Evaluate the best individual's test accuracy and append it to the trace
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