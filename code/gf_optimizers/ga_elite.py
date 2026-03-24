# Import NumPy for random sampling, vectorized crossover/mutation, sorting, and array ops
import numpy                         as     np
# Import tqdm to display a progress bar over generations
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (handles wrapper/epoch bookkeeping)
from   help_classes.abstract_classes import AbstractOptimizer_GF

# Define a truncation-selection Genetic Algorithm variant with an elite offspring pool
class GA_Elite(AbstractOptimizer_GF):
    """
    Genetic optimizer following Castaldo et al.,
    "Quantum optimal control with quantum computers: A hybrid algorithm
    featuring machine learning optimization" (Phys. Rev. A 103, 2021).

    The name **GA_Elite** reflects that this GA uses *elite (best-loss) truncation selection*:
    only the top-performing individuals are retained as the parent pool for reproduction.

    This class implements a truncation-selection genetic algorithm for
    continuous parameter vectors. A population of candidate solutions is
    iteratively evolved via:
      * selection of the best-performing individuals (offspring pool),
      * uniform crossover between randomly chosen parents, and
      * Gaussian mutation applied per gene.

    Algorithm overview
    ------------------
    Let each individual be a real-valued parameter vector of dimension
    ``ind_size`` (given by ``wrapper.get_ind_size()``). For each generation:

      1. **Evaluation (training set)**:
         Every individual in the current population is evaluated via
         ``wrapper.evaluate_single_final(X_data=wrapper.X_train,
                                         y_labels=wrapper.y_train,
                                         individual=theta)``.
         The first return value is interpreted as a loss to be minimized.

      2. **Truncation selection**:
         Individuals are sorted by training loss, and the top
         ``offspring_size`` form the *offspring* pool from which parents
         are sampled for the next generation.

      3. **Uniform crossover**:
         Parent pairs are drawn uniformly at random (without replacement)
         from the offspring pool. For each pair and each gene, with
         probability 0.5, the gene values are swapped between parents.
         Crossover is applied to a pair with probability ``p_cx``; with
         probability ``1 - p_cx`` the parents are copied unchanged.
         The resulting children overwrite the current population.

      4. **Gaussian mutation**:
         For every individual and every gene, mutation is applied with
         probability ``p_mut`` by adding Gaussian noise with zero mean
         and standard deviation ``sigma_gs``. This step injects local
         exploration around the current offspring.

    Initialization and logging
    --------------------------
    * The initial population is drawn i.i.d. from a zero-mean Gaussian
      with standard deviation ``sigma_init``.
    * The total number of generations is
      ``n_iterations = (n_evals_per_epoch * n_epochs) // pop_size``.
    * A vector ``marker`` encodes the generation indices at which an
      “epoch” snapshot is taken. At each such generation:
        - The current population is re-evaluated on the training set.
        - The best individual (lowest training loss) is evaluated on the
          **test** set.
        - The resulting test accuracy (second return value of
          ``evaluate_single_final``) is appended to ``accuracies``.

    The ``run`` method stops after all generations are processed and
    checks that exactly ``n_epochs`` accuracies were collected. If not,
    it prints a diagnostic message and terminates the program.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must implement:
          * ``get_ind_size() -> int``: return the dimensionality of the
            parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``: evaluate a single parameter vector
            on given data.
    n_evals_per_epoch : int
        Target number of model evaluations per epoch; together with
        ``n_epochs`` and ``pop_size`` this determines the number of
        generations and logging points.
    n_epochs : int
        Number of epochs (i.e., test-accuracy snapshots) to record.
    sigma_init : float
        Standard deviation of the Gaussian used to initialize the
        population. Controls the initial spread of candidate solutions
        in parameter space.
    pop_size : int
        Total population size. Larger values increase search diversity
        but raise the computational cost per generation.
    offspring_size : int
        Number of top-performing individuals retained as the parent
        pool for the next generation. Controls selection pressure:
        smaller values increase exploitation but reduce diversity.
    p_cx : float
        Probability of applying uniform crossover to a selected parent
        pair. With probability ``1 - p_cx``, the pair is copied to the
        next generation without recombination.
    p_mut : float
        Per-gene mutation probability. For each parameter of each
        individual, mutation is applied with this probability.
    sigma_gs : float
        Standard deviation of the Gaussian noise added during mutation.
        Sets the typical magnitude of parameter perturbations.

    Returns
    -------
    list[float]
        Test accuracies of the best individual (according to training
        loss) at each epoch. The list has length ``n_epochs``.
    """
    
    # Initialize GA hyperparameters and derive iteration/logging schedule from evaluation budget
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, sigma_init, pop_size, offspring_size, p_cx, p_mut, sigma_gs) -> None:
        # Initialize base class (stores wrapper, n_evals_per_epoch, n_epochs, and sets wrapper.flag_GF=True)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Std-dev for Gaussian initialization of the population
        self.sigma_init     = sigma_init
        # Number of individuals in the population
        self.pop_size       = pop_size
        # Number of elites retained as the mating pool (truncation selection)
        self.offspring_size = offspring_size
        # Probability of performing uniform crossover on a parent pair
        self.p_cx           = p_cx
        # Per-gene probability of mutation
        self.p_mut          = p_mut
        # Std-dev of Gaussian noise used for mutation
        self.sigma_gs       = sigma_gs

        # Dimensionality of an individual (number of parameters)
        self.ind_size       = self.wrapper.get_ind_size()
        # Number of generations implied by total evaluation budget and population size
        self.n_iterations   = (n_evals_per_epoch*n_epochs)//pop_size

        # Generation indices at which to record epoch-level test accuracy
        self.marker         = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // pop_size
        # Ensure last logging happens at the final generation index
        self.marker[-1]     = self.n_iterations
    

    # Main optimization loop: evaluate → select elites → crossover → mutate → log best test accuracy
    def run(self):
        """
        Runs the gradient free optimization algorithm from Castaldo et al.

        :return: returns the final model accuracy in test data
        """

        # Initialize population by sampling each gene from N(0, sigma_init^2)
        population                     = np.random.normal(loc   = 0                             ,
                                                          scale = self.sigma_init               ,
                                                          size  = (self.pop_size, self.ind_size))

        # Initialize array to store test accuracies at epoch markers
        accuracies                     = []
        
        # Start generational process with a progress bar
        pbar                           = tqdm(range(1, self.n_iterations+1))
        for generation in pbar:
            # Update progress-bar label
            pbar.set_description(f"Currently running generation {generation}")

            # Evaluate population on training data and collect losses (to be minimized)
            evaluations                = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                             y_labels   = self.wrapper.y_train,
                                                                             individual = indvidual           )[0] for indvidual in population]

            # Select elites: take the offspring_size individuals with lowest training loss
            offspring                  = population[np.argsort(evaluations)[0:self.offspring_size]]

            # Generate a new population via pairwise reproduction (pop_size must be even for this scheme)
            for y in range(int(self.pop_size//2)):
                
                # Sample two distinct parents from the elite pool (no replacement)
                parents                = offspring[np.random.choice(a       = np.arange(self.offspring_size),
                                                                    size    = 2                             ,
                                                                    replace = False                        )]

                # With probability p_cx, perform uniform crossover (gene-wise mixing)
                if np.random.random() <= self.p_cx:

                    # Boolean mask: True means take gene from the other parent (uniform 0.5 split)
                    p                  = np.random.random(self.ind_size) >= 0.5

                    # Construct child 0 as masked mix of parent 0 and parent 1
                    parents[0]         = (np.ones(self.ind_size)-p)*parents[0] + p*parents[1]
                    # Construct child 1 as masked mix of parent 1 and parent 0
                    parents[1]         = (np.ones(self.ind_size)-p)*parents[1] + p*parents[0]

                # Insert the two children into the new population (overwrite positions 2y and 2y+1)
                population[y*2:y*2+2]  = parents


            # Apply Gaussian mutation to each individual and each gene (independent per gene)
            for i in range(self.pop_size):

                # Sample per-gene mutation probabilities in [0, 1]
                p                            = np.random.uniform(low  = 0            ,
                                                                 high = 1            ,
                                                                 size = self.ind_size)

                # Sample per-gene Gaussian noise N(0, sigma_gs^2)
                gaussian_noise               = np.random.normal(loc   = 0            ,
                                                                scale = self.sigma_gs,
                                                                size  = self.ind_size)

                # Mutate genes where p <= p_mut by adding Gaussian noise
                population[i]         += (p <= self.p_mut) * gaussian_noise


            # At epoch markers: evaluate best-by-training-loss and record its test accuracy
            if generation in self.marker:

                # Re-evaluate population on training set to identify current best individual
                evaluations = [   self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train              ,
                                                                     y_labels   = self.wrapper.y_train              ,
                                                                     individual = individual                        )[0] for individual in population]

                # Evaluate best individual on test set and append the resulting accuracy
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test               ,
                                                                     y_labels   = self.wrapper.y_test               ,
                                                                     individual = population[np.argmin(evaluations)])[1])


        # Sanity check: must have exactly one recorded accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit
        
        # Return list of test accuracies at epoch markers
        return accuracies