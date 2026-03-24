# Import NumPy for population initialization, sampling, crossover/mutation masks, and array ops
import numpy                         as     np
# Import tqdm to display and manually advance a progress bar (based on evaluation counts)
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (handles wrapper/epoch bookkeeping)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define a Genetic Algorithm variant using tournament selection (GA_Tourn)
class GA_Tourn(AbstractOptimizer_GF):
    """
    Genetic optimizer following Acampora et al., *"Training Variational Quantum Circuits
    through Genetic Algorithms"*.

    This optimizer treats the trainable parameters of the underlying model (provided via
    ``wrapper``) as real-valued chromosomes and applies a generational genetic algorithm
    with tournament selection, uniform crossover, and Gaussian mutation.

    Population initialization
    -------------------------
    The initial population is sampled i.i.d. from a normal distribution
    :math:`\mathcal{N}(0, \texttt{sigma_init}^2)` of dimension ``ind_size``, where
    ``ind_size`` is obtained from ``wrapper.get_ind_size()``. Each individual encodes
    a full set of model parameters.

    Selection and variation
    -----------------------
    In each generation, the algorithm repeatedly:
      * Samples ``t_size`` distinct individuals for tournament selection and evaluates
        them on the training set via ``wrapper.evaluate_single_final``. Tournament
        fitness values are cached so that each individual is evaluated at most once per
        generation.
      * Selects the two fittest individuals as parents.
      * Applies **uniform crossover** with probability ``p_cx`` by randomly swapping
        genes between the parents; with probability ``1 - p_cx`` the parents are copied
        unchanged.
      * Applies **Gaussian mutation** to every offspring: for each gene, with probability
        ``p_mut`` the parameter is perturbed by zero-mean Gaussian noise with standard
        deviation ``sigma_gs``.

    Training loop and logging
    -------------------------
    The outer loop runs until approximately ``n_evals_per_epoch * n_epochs`` model
    evaluations have been performed (up to caching effects). At the end of each
    "epoch" (as defined by the cumulative number of evaluations), the current
    population is fully evaluated on the training set and the best individual
    (lowest training loss / highest training fitness) is then evaluated on the
    test set. The corresponding test accuracy is stored.

    The ``run`` method returns a list of length ``n_epochs`` containing the test
    accuracy of the best individual at each epoch. If the number of collected
    accuracies does not match ``n_epochs``, the method prints a diagnostic message
    and terminates the program.

    Parameters
    ----------
    wrapper : object
        Object that encapsulates the quantum/classical model and data. It must
        implement:
          * ``get_ind_size() -> int`` returning the dimensionality of the parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual) -> (loss, accuracy)``,
            used on both training and test sets.
    n_evals_per_epoch : int
        Target number of model evaluations per epoch used to define logging intervals.
    n_epochs : int
        Number of epochs (logging points) to run. Determines the length of the
        returned accuracy trace.
    sigma_init : float
        Standard deviation of the Gaussian distribution used for population
        initialization.
    pop_size : int
        Population size. Larger values increase search diversity but also the
        computational cost per generation.
    t_size : int
        Tournament size, i.e., the number of individuals sampled in each tournament.
        Larger values increase selection pressure but can reduce population diversity.
    p_cx : float
        Probability of applying uniform crossover to a selected pair of parents.
        With probability ``1 - p_cx``, the parents are copied unchanged.
    p_mut : float
        Per-gene mutation probability. For each parameter in each offspring, with
        probability ``p_mut`` a Gaussian perturbation is applied.
    sigma_gs : float
        Standard deviation of the Gaussian noise used in the mutation operator.
        Controls the typical step size of parameter updates.

    Returns
    -------
    list[float]
        Test accuracies of the best individual (according to training fitness)
        at the end of each epoch. The list has length ``n_epochs``.
    """

    # Initialize GA hyperparameters and bookkeeping for evaluation-budget-based stopping/logging
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, sigma_init, pop_size, t_size, p_cx, p_mut, sigma_gs) -> None:
        # Initialize base class (stores wrapper, n_evals_per_epoch, n_epochs, and sets wrapper.flag_GF=True)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Std-dev for Gaussian initialization of population individuals
        self.sigma_init   = sigma_init
        # Number of individuals in the population
        self.pop_size     = pop_size
        # Tournament size (number of contenders sampled per tournament)
        self.t_size       = t_size
        # Probability of applying uniform crossover to a parent pair
        self.p_cx         = p_cx
        # Per-gene mutation probability
        self.p_mut        = p_mut
        # Std-dev of Gaussian noise used for mutation
        self.sigma_gs     = sigma_gs

        # Dimensionality of an individual (number of parameters)
        self.ind_size     = self.wrapper.get_ind_size()
        # Total evaluation budget (in evaluations, not generations)
        self.n_iterations = n_evals_per_epoch*n_epochs

        # Cumulative evaluation-count markers where an epoch snapshot is taken
        self.marker       = n_evals_per_epoch*np.arange(1, n_epochs+1, 1)
    

    # Main loop: create new generations while counting training evaluations; log best test accuracy per epoch marker
    def run(self):
        """
        Runs the gradient free optimization algorithm from Acampora et al.

        :return: returns the final model accuracy in test data
        """

        # Initialize population by sampling each gene from N(0, sigma_init^2)
        population                           = np.random.normal(loc   = 0                             ,
                                                                scale = self.sigma_init               ,
                                                                size  = (self.pop_size, self.ind_size))

        # Initialize list to store epoch-level test accuracies
        accuracies                           = []

        # Track total number of training-set model evaluations performed so far
        n_evals                              = 0

        # Index into marker array indicating the next epoch boundary
        counter = 0

        # Create a progress bar over the total evaluation budget (advanced manually per loop)
        pbar                                 = tqdm(range(1, self.n_evals_per_epoch*self.n_epochs + 1))
        
        # Continue until evaluation budget is exhausted (or until we have logged all epochs)
        while n_evals <= self.n_iterations:
            # Track number of *new* evaluations performed in this generation (due to caching)
            n_evals_new                      = 0
            
            # Set progress description (uses n_evals as a proxy "generation" label)
            pbar.set_description(f"Currently running generation {n_evals}")
            
            # Initialize cache for training losses; 0 means "not evaluated yet" for that index
            evaluations                      = np.zeros(self.pop_size)

            # Allocate array for the next generation population
            population_new                   = np.zeros(shape=(self.pop_size, self.ind_size))            

            # Create pop_size offspring in pairs (two children per iteration)
            for y in range(int(self.pop_size / 2)):

                # Sample distinct indices for tournament contenders
                indices_for_tournament       = np.random.choice(a       = np.arange(self.pop_size),
                                                                size    = self.t_size             ,
                                                                replace = False                   )

                # Fetch contender individuals from current population
                individuals_for_tournament   = [population[i] for i in indices_for_tournament]

                # Allocate array to store tournament losses (fitness values)
                evaluations_from_tournament  = np.zeros(self.t_size)

                # Evaluate contenders (with caching so each individual is evaluated at most once per generation)
                for i in range(self.t_size):
                    
                    # Map local tournament index to global population index
                    x = indices_for_tournament[i]

                    # If not evaluated yet in this generation, evaluate on training set and cache
                    if evaluations[x] == 0:
                        evaluations[x]       = self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                                  y_labels   = self.wrapper.y_train,
                                                                                  individual = population[x]       )[0]
                        # Count a new model evaluation (training-set evaluation)
                        n_evals_new         += 1
                    
                    # Store cached loss into tournament evaluation array
                    evaluations_from_tournament[i] = evaluations[x]
                
                # Select the two best contenders (lowest loss) as parents
                parents                      = np.array([individuals_for_tournament[i] for i in np.argsort(evaluations_from_tournament)[0:2]])

                # Apply uniform crossover with probability p_cx
                if np.random.random() < self.p_cx:

                    # Gene-wise mask (True => swap/take from other parent); 0.5 probability per gene
                    p                        = np.random.random(self.ind_size) > 0.5

                    # Mix genes to form child 0
                    parents[0]               = (np.ones(self.ind_size)-p)*parents[0] + p*parents[1]
                    # Mix genes to form child 1
                    parents[1]               = (np.ones(self.ind_size)-p)*parents[1] + p*parents[0]
               
                # Insert the two children into the new population buffer
                population_new[y*2:y*2+2]    = parents

            # Advance progress bar by the number of *new* evaluations performed this generation
            pbar.update(n_evals_new)

            # Add new evaluations to the global evaluation counter
            n_evals                         += n_evals_new

            # Apply Gaussian mutation to each individual (gene-wise independent)
            for i in range(self.pop_size):
                
                # Per-gene mutation probability samples in [0, 1]
                p                            = np.random.uniform(low  = 0            ,
                                                                 high = 1            ,
                                                                 size = self.ind_size)

                # Per-gene Gaussian noise N(0, sigma_gs^2)
                gaussian_noise               = np.random.normal(loc   = 0            ,
                                                                scale = self.sigma_gs,
                                                                size  = self.ind_size)

                # Add noise where p < p_mut (mutate selected genes)
                population_new[i]           += (p < self.p_mut) * gaussian_noise

            # Replace current population with the newly generated (and mutated) population
            population                       = population_new


            # Check whether we've crossed an epoch boundary (within a window of pop_size evaluations)
            if (n_evals - self.marker[counter]) in np.arange(0, self.pop_size):

                # Move to next epoch marker after logging
                counter += 1

                # Evaluate full population on training set to identify current best individual
                evaluations  = [  self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train              ,
                                                                     y_labels   = self.wrapper.y_train              ,
                                                                     individual = individual                        )[0] for individual in population]

                # Evaluate best-by-training-loss individual on test set and append accuracy
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test               ,
                                                                     y_labels   = self.wrapper.y_test               ,
                                                                     individual = population[np.argmin(evaluations)])[1])
                
                # If we have logged all epochs, stop early
                if counter == self.n_epochs:
                    break


        # Sanity check: must have exactly one accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit()

        # Return list of epoch-level test accuracies
        return accuracies