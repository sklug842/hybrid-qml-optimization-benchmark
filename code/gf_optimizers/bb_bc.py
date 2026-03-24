# Import NumPy for numerical operations, random sampling, and array manipulation
import numpy                         as     np
# Import tqdm to display a progress bar during the optimization loop
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (provides wrapper/epoch bookkeeping)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define the Big Bang–Big Crunch (BB-BC) optimizer as a gradient-free optimizer
class BB_BC(AbstractOptimizer_GF):
    """
    Big Bang--Big Crunch (BB--BC) optimizer following Erol and Eksin (2006) and
    the neural-network adaptation of Wang and Kumbasar (2019).

    This optimizer maintains a population of candidate parameter vectors that is
    repeatedly “exploded” (Big Bang) around a weighted centroid (Big Crunch) to
    balance exploration and exploitation in a continuous search space.

    Algorithm overview
    ------------------
    Let each individual in the population be a real-valued parameter vector of
    dimension ``ind_size`` (obtained from ``wrapper.get_ind_size()``). Each
    generation consists of:

      1. **Evaluation (training set)**:
         Every individual in the current population is evaluated via
         ``wrapper.evaluate_single_final(X_data=wrapper.X_train,
                                         y_labels=wrapper.y_train,
                                         individual=theta)``.
         The first return value is interpreted as a loss to be minimized.

      2. **Big Crunch (centroid update)**:
         A new “center of mass” :math:`x_c` is computed as a fitness-weighted
         average of the population, where weights are proportional to
         :math:`1/(\\text{loss} + \\varepsilon)`. Lower loss therefore yields
         higher influence on the centroid.

      3. **Big Bang (population generation)**:
         A new population is sampled around :math:`x_c` according to
         ::
             
             x_new = x_c + alpha * (x_max - x_min) * U(-1, 1) / generation

         where ``U(-1, 1)`` is i.i.d. uniform noise and ``alpha`` controls the
         exploration radius. As ``generation`` increases, the perturbation
         shrinks, leading to a gradual transition from exploration to
         exploitation.

         In the **first call** (initialization), the population is sampled
         uniformly from the hyper-rectangle ``[x_min, x_max]`` without using
         ``x_c``.

      4. **Clipping**:
         After generation, each parameter is clipped elementwise into the
         interval ``[x_min, x_max]``.

    Training loop and logging
    -------------------------
    The number of generations is derived from the total evaluation budget:
    ``n_iterations = (n_evals_per_epoch * n_epochs) // pop_size - 1``.

    A vector ``marker`` encodes the generation indices at which an “epoch”
    snapshot is taken. At each such generation:

      * The population is re-evaluated on the training set.
      * The best individual (lowest training loss) is evaluated on the **test**
        set.
      * The resulting test accuracy (second return value of
        ``evaluate_single_final``) is appended to ``accuracies``.

    The ``run`` method finishes when all generations are processed and checks
    that exactly ``n_epochs`` accuracies were collected. If not, it prints a
    diagnostic message and terminates the program.

    Parameters
    ----------
    wrapper : object
        Object encapsulating the model and data. It must implement:
          * ``get_ind_size() -> int``: return the dimensionality of the
            parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``: evaluate a single parameter vector on
            given data.
    n_evals_per_epoch : int
        Target number of model evaluations per epoch; together with
        ``n_epochs`` and ``pop_size`` this determines the number of generations
        and logging points.
    n_epochs : int
        Number of epochs (i.e., test-accuracy snapshots) to record.
    pop_size : int
        Population size. Larger values increase search diversity but raise the
        cost per generation.
    alpha : float
        Expansion coefficient in the Big Bang step. Scales the random
        displacement from the centroid and thus controls the exploration
        radius. Larger values encourage broader exploration.
    x_min : float or array-like
        Lower bound of the search space for each parameter. Can be a scalar
        (broadcast to all dimensions) or a vector of length ``ind_size``.
        Used for initialization, sampling, and clipping.
    x_max : float or array-like
        Upper bound of the search space for each parameter. Same shape
        conventions as ``x_min``.

    Returns
    -------
    list[float]
        Test accuracies of the best individual (according to training loss)
        at each epoch. The list has length ``n_epochs``.
    """

    # Constructor: stores BB-BC hyperparameters and precomputes iteration/logging indices
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, pop_size, alpha, x_min, x_max) -> None: # g_step_max
        # Initialize base class (stores wrapper, n_evals_per_epoch, n_epochs, and sets wrapper.flag_GF=True)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Number of candidate solutions sampled per generation
        self.pop_size     = pop_size
        # Expansion factor controlling the exploration radius around the centroid
        self.alpha        = alpha
        # Lower bound(s) for parameter values (scalar or per-dimension array)
        self.x_min        = x_min
        # Upper bound(s) for parameter values (scalar or per-dimension array)
        self.x_max        = x_max

        # Dimensionality of the parameter vector (number of trainable parameters)
        self.ind_size     = self.wrapper.get_ind_size()
        # Flag to indicate whether the next generation is the initial population draw
        self.pop_init     = True
        # Total number of BB-BC generations derived from the evaluation budget
        self.n_iterations = (n_evals_per_epoch*n_epochs)//pop_size - 1

        # Marker indices used to decide when to record an "epoch" accuracy snapshot
        self.marker       = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // pop_size
        # Force the last marker to match the final generation index (ensures n_epochs snapshots)
        self.marker[-1]   = self.n_iterations


    # Generate an initial population (uniform) or a new population around centroid x_c (Big Bang step)
    def generate(self, x_c, generation):
        # On the first call, initialize the population uniformly within [x_min, x_max]
        if self.pop_init:
            # Disable initialization mode after generating the first population
            self.pop_init = False

            # Sample pop_size individuals uniformly within the bounded search space
            return np.random.uniform(low  = self.x_min                    ,
                                     high = self.x_max                    ,
                                     size = (self.pop_size, self.ind_size))
        
        # On subsequent calls, sample around the centroid with a shrinking radius (1/generation)
        else:
            # Create new individuals by perturbing centroid x_c with scaled uniform noise
            return x_c + (self.alpha * (self.x_max-self.x_min) * np.random.uniform(low  = -1                            ,
                                                                                   high =  1                            , 
                                                                                   size = (self.pop_size, self.ind_size)) / generation)
    
    
    # Compute the BB-BC centroid (Big Crunch step) as a fitness-weighted average of the population
    def update(self, population, evaluations):

        # Convert losses to positive weights by inverting them (lower loss => higher weight)
        raw_weights = 1 / (np.array(evaluations) + 1e-10)

        # Normalize weights so they sum to 1 (required by np.average with weights)
        weights     = raw_weights / np.sum(raw_weights)

        # Compute the weighted centroid x_c across the population (dimension-wise average)
        x_c         = np.average(a       = population,
                                 weights = weights   ,
                                 axis    = 0         )

        # Return the updated centroid (center of mass)
        return x_c


    # Main optimization routine: iterate Big Bang / Big Crunch steps and log test accuracy at markers
    def run(self):

        # Initialize population (first call uses uniform sampling in [x_min, x_max])
        population = self.generate(x_c        = 0   ,
                                   generation = None)

        # Initialize array for accuracies
        accuracies = []

        # Start generational process
        pbar       = tqdm(range(1, self.n_iterations+1))

        # Iterate through BB-BC generations
        for generation in pbar:
            pbar.set_description(f"Currently running generation {generation}")

            # Evaluate population (training loss for each individual)
            evaluations = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                              y_labels   = self.wrapper.y_train,
                                                              individual = individual          )[0] for individual in population]

            # Update centroid x_c using fitness-weighted averaging (Big Crunch)
            x_c        = self.update(population  = population ,
                                     evaluations = evaluations)

            # Generate new population around centroid (Big Bang) with shrinking perturbation
            population = self.generate(x_c        = x_c       ,
                                       generation = generation)

            # Clip all parameters back into the feasible search bounds
            population = np.clip(a     = population,
                                 a_min = self.x_min,
                                 a_max = self.x_max)
            
            # If this generation corresponds to an "epoch" marker, evaluate and record test accuracy
            if generation in self.marker:

                # Re-evaluate population on training set to select the best individual by loss
                evaluations = [   self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train              ,
                                                                     y_labels   = self.wrapper.y_train              ,
                                                                     individual = individual                        )[0] for individual in population]

                # Evaluate the best individual on the test set and append its accuracy
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test               ,
                                                                     y_labels   = self.wrapper.y_test               ,
                                                                     individual = population[np.argmin(evaluations)])[1])
        
        # Sanity check: ensure that we recorded exactly one accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit()

        # Return collected test accuracies (length == n_epochs)
        return accuracies