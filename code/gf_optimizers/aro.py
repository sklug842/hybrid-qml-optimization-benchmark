# Import NumPy for numerical operations, random sampling, and vectorized array manipulation
import numpy                         as     np
# Import tqdm to display a progress bar for the optimization loop
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (provides wrapper/epoch bookkeeping)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define the Asexual Reproduction Optimization (ARO) class as a gradient-free optimizer
class ARO(AbstractOptimizer_GF):
    """
    Gradient-free optimizer inspired by Hashemi et al. (2019) on
    Asexual Reproduction Optimization (ARO) for neural network training.

    This implementation adapts the ARO-style update rule to the
    variational quantum circuit setting. Instead of maintaining a
    population, the algorithm evolves a **single** candidate solution
    (the "individual") over a fixed number of model evaluations. In each
    iteration, a mutated offspring ("larva") is created by perturbing a
    contiguous segment of the parameter vector with Gaussian noise; the
    larva replaces the current individual only if it achieves a better
    training fitness (lower loss), yielding a greedy, hill-climbing
    search in parameter space.

    Algorithm overview
    ------------------
    Let the parameter vector be of dimension ``ind_size``
    (obtained from ``wrapper.get_ind_size()``). The optimizer proceeds
    for ``n_iterations = n_epochs * n_evals_per_epoch`` steps:

      1. **Initialization**:
         The initial individual is drawn from a zero-mean Gaussian:

         ``individual ~ N(0, sigma_init^2 I)``.

         Its training fitness is computed via

         ``fit_individual = wrapper.evaluate_single_final(
                                X_data=wrapper.X_train,
                                y_labels=wrapper.y_train,
                                individual=individual
                            )[0]``.

      2. **Mutation (larva generation)**:
         At each generation, a larva is created as a copy of the current
         individual. A segment length ``g`` and an ending index
         ``ending_point`` are chosen at random, and the segment
         (interpreted on a ring, so it may wrap around the parameter
         vector) is perturbed. A crossover-like probability

         ``p_cx = 1 / (1 + log(g))``

         is computed, and each parameter in the chosen segment is
         mutated independently with probability ``p_cx`` by adding
         Gaussian noise drawn from

         ``N(0, sigma_gs^2)``.

      3. **Selection (greedy acceptance)**:
         The larva is evaluated on the training set via

         ``fit_larva = wrapper.evaluate_single_final(
                           X_data=wrapper.X_train,
                           y_labels=wrapper.y_train,
                           individual=larva
                       )[0]``.

         If ``fit_larva < fit_individual``, the larva replaces the
         current individual:

         ``individual = larva``, ``fit_individual = fit_larva``.

      4. **Epoch logging (test set)**:
         Every ``n_evals_per_epoch`` generations, the current individual
         is evaluated on the test set, and the resulting accuracy is
         appended to ``accuracies``:

         ``accuracy = wrapper.evaluate_single_final(
                          X_data=wrapper.X_test,
                          y_labels=wrapper.y_test,
                          individual=individual
                      )[1]``.

         After ``n_epochs`` such logging points, the run terminates.

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
              Evaluate a single parameter vector on the given data.
    n_evals_per_epoch : int
        Number of model evaluations per epoch. Together with
        ``n_epochs`` this sets the total number of iterations and the
        spacing of test-accuracy snapshots.
    n_epochs : int
        Number of epochs (i.e., times test accuracy is recorded). The
        total number of generations is ``n_evals_per_epoch * n_epochs``.
    sigma_init : float
        Standard deviation of the Gaussian used to initialize the
        individual. Larger values seed a wider initial spread in
        parameter space.
    sigma_gs : float
        Standard deviation of the Gaussian noise used during mutation of
        the selected segment. This controls the typical step size of
        parameter perturbations.

    Returns
    -------
    list[float]
        Test accuracies of the best individual (according to training
        loss) at the end of each epoch. The list has length
        ``n_epochs``.
    """

    # Constructor: stores hyperparameters and derives loop constants from wrapper/problem size
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, sigma_init, sigma_gs) -> None:
        # Initialize base class (stores wrapper, n_evals_per_epoch, n_epochs, and sets wrapper.flag_GF=True)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Standard deviation for the initial Gaussian sampling of the parameter vector
        self.sigma_init   = sigma_init
        # Standard deviation for Gaussian mutation noise applied during larva generation
        self.sigma_gs     = sigma_gs

        # Dimensionality of the optimization vector (number of trainable parameters)
        self.ind_size     = self.wrapper.get_ind_size()
        # Total number of iterations equals total evaluation budget: epochs * evals per epoch
        self.n_iterations = n_epochs*n_evals_per_epoch


    # Main optimization routine: performs n_iterations greedy mutation-selection steps and logs test accuracy each epoch
    def run(self):
        """
        Runs the gradient free optimization algorithm from Hashemi et al.

        :return: returns the final model accuracy in test data
        """

        # Initialize the individual
        individual     = np.random.normal(loc   = 0              ,
                                          scale = self.sigma_init,
                                          size  = self.ind_size  )

        # Evaluate the individuals with an invalid fitness
        fit_individual = self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                            y_labels   = self.wrapper.y_train,
                                                            individual = individual          )[0]
        
        # Initialize array for accuracies     
        accuracies     = []

        # Start generational process  
        pbar = tqdm(range(1, self.n_iterations+1))
        for generation in pbar:
            pbar.set_description(f"Currently running generation {generation}")

            # Copy of individual
            larva                                   = np.copy(individual)    

            # Go beyond the border of the array and start at the beginning, i.e. view array as ring
            g                                       = np.random.choice(a = np.arange(1, self.ind_size+1))
            ending_point                            = np.random.choice(a = np.arange(0, self.ind_size  ))

            # Calculate crossover probability
            p_cx                                    = 1/(1+np.log(g))

            # Only add gaussain noise with probablilty p
            if ending_point-g < 0:

                # Probability
                p1                                 = np.random.uniform(low  = 0             ,
                                                                       high = 1             ,
                                                                       size =   ending_point)
                
                p2                                 = np.random.uniform(low  = 0             ,
                                                                       high = 1             ,
                                                                       size = g-ending_point)
                
                # Gaussian noise
                gaussian_noise1                    = np.random.normal(loc   = 0            ,
                                                                      scale = self.sigma_gs,
                                                                      size  = ending_point)
                
                gaussian_noise2                    = np.random.normal(loc   = 0            ,
                                                                      scale = self.sigma_gs,
                                                                      size  = g-ending_point)
                


                larva[0:ending_point]              += (p1 < p_cx) * gaussian_noise1
                larva[-(g-ending_point):]          += (p2 < p_cx) * gaussian_noise2

            else:

                # Probability
                p                                   = np.random.uniform(low  = 0             ,
                                                                        high = 1             ,
                                                                        size = g)
                
                # Gaussian noise
                gaussian_noise                      = np.random.normal(loc   = 0            ,
                                                                       scale = self.sigma_gs,
                                                                       size  = g)
                


                larva[ending_point-g:ending_point] += (p < p_cx) * gaussian_noise


            # Evaluate larva
            fit_larva                               = self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                                         y_labels   = self.wrapper.y_train,
                                                                                         individual = larva               )[0]
            
            # Check, if larva has the better fitness, if yes: larva becomes new individual
            if fit_larva < fit_individual:
                individual     = np.copy(larva    )
                fit_individual = np.copy(fit_larva)


            if generation % self.n_evals_per_epoch == 0:
            
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test,
                                                                     y_labels   = self.wrapper.y_test,
                                                                     individual = individual         )[1])


        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit

        # Return accuracies
        return accuracies