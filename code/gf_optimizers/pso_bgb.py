# Import NumPy for swarm initialization, velocity/position updates, and biogeography-style recombination
import numpy                         as     np
# Import tqdm to visualize iteration progress with a progress bar
from   tqdm                          import tqdm
# Import the abstract base class for gradient-free optimizers (stores wrapper and evaluation budget)
from   help_classes.abstract_classes import AbstractOptimizer_GF


# Define PSO with optional Biogeography-based recombination as a gradient-free optimizer
class PSO_BGB(AbstractOptimizer_GF):
    """
    Particle Swarm Optimization with optional Biogeography-based recombination (PSO / PSO BGB).

    This class implements a particle-swarm–based gradient-free optimizer for
    variational quantum circuits, using the interface provided by ``wrapper``.
    Depending on the flag ``flag_bgb``, the algorithm behaves either as:

      * **Standard PSO** in the sense of Kennedy & Eberhart (1995) [PSO],
      * or as **Biogeography-based PSO (PSO BGB)** following Mousavirad et al. (2020),
        where personal-best positions are periodically recombined based on
        biogeographic migration probabilities.

    The swarm consists of ``pop_size`` particles, each representing a candidate
    parameter vector for the variational circuit. At each iteration, particle
    velocities and positions are updated using the standard PSO velocity rule,
    and the wrapper is used to evaluate fitness (training loss) and track test
    accuracy over epochs.

    Algorithm overview
    ------------------
    Let ``ind_size`` be the dimension of the parameter vector obtained from
    ``wrapper.get_ind_size()``. For each PSO iteration:

      1. **Initialization**:
         * Positions ``X`` are uniformly initialized in
           ``[x_min, x_max]^{ind_size}``.
         * Velocities ``V`` are initialized uniformly in ``[0, 1]``.
         * Personal-best positions ``personal_best`` are set to the initial
           positions, and their fitness values are computed on the training set.
         * The global-best position ``global_best`` is the personal-best with
           the lowest training loss.

      2. **Velocity update**:
         For each particle, the velocity is updated as
         ::
             V = w * V
                 + c1 * r1 * (personal_best_or_bgb - X)
                 + c2 * use_global_best * r2 * (global_best - X)

         where:
           * ``w`` is the inertia weight,
           * ``c1`` is the cognitive coefficient,
           * ``c2`` is the social coefficient,
           * ``r1`` and ``r2`` are i.i.d. random vectors sampled uniformly from [0, 1],
           * ``personal_best_or_bgb`` is:
                - ``personal_best`` in standard PSO mode (``flag_bgb == False``),
                - a **biogeographically recombined** personal-best population
                  ``personal_best_new`` in PSO BGB mode (``flag_bgb == True``),
           * ``use_global_best`` is typically 0 or 1 and scales or disables the
             global-best term.

         In PSO BGB mode, ``personal_best_new`` is produced by
         :meth:`get_personal_best_new`, which implements a migration-like
         recombination mechanism: genes of each particle’s personal-best are
         probabilistically replaced by genes from other particles’ personal-bests
         according to rank-based migration rates.

      3. **Position update and clipping**:
         Positions are updated via
         ::
             X = X + V

         and are implicitly kept within the search domain by virtue of the
         bounds used during initialization; if additional clipping is required,
         it can be added using ``np.clip`` with ``x_min`` and ``x_max``.

      4. **Evaluation and best updates (training set)**:
         Each particle’s fitness (training loss) is computed as
         ::
             fitness_X = wrapper.evaluate_single_final(
                 X_data=wrapper.X_train,
                 y_labels=wrapper.y_train,
                 individual=particle
             )[0]

         Personal-best positions and fitnesses are updated wherever the new
         fitness is not worse than the stored personal-best fitness. The global
         best is then updated as the particle with the lowest personal-best
         fitness.

      5. **Epoch logging (test set)**:
         The total number of iterations is
         ``n_iterations = (n_evals_per_epoch * n_epochs) // pop_size``.
         At specific iteration indices stored in ``self.marker``, the algorithm
         evaluates the current global-best particle on the test set and records
         its accuracy:

         ::
             accuracy = wrapper.evaluate_single_final(
                 X_data=wrapper.X_test,
                 y_labels=wrapper.y_test,
                 individual=global_best
             )[1]

         These accuracies are appended once per epoch, yielding a trajectory of
         test performance over ``n_epochs``.

    After all iterations, the method checks that exactly ``n_epochs`` test
    accuracies have been collected and returns them.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must implement:
          * ``get_ind_size() -> int``:
                Return the dimensionality of the parameter vector.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``:
                Evaluate a single parameter vector on the given data and return
                a scalar loss (to be minimized) and a corresponding accuracy.
    n_evals_per_epoch : int
        Number of model evaluations per epoch. Together with ``n_epochs`` and
        ``pop_size``, this determines the total number of PSO iterations and
        the spacing of test-accuracy snapshots.
    n_epochs : int
        Number of epochs (i.e., times test accuracy is recorded). The total
        number of swarm updates is
        ``n_iterations = (n_evals_per_epoch * n_epochs) // pop_size``.
    pop_size : int
        Swarm size, i.e., the number of particles. Larger values increase
        exploration and robustness of the search at higher computational cost.
    x_min : float or array-like
        Lower bound of the search space for each parameter. Initial positions
        are sampled from ``[x_min, x_max]`` and, if clipping is applied, are
        never allowed to fall below this value.
    x_max : float or array-like
        Upper bound of the search space for each parameter. Initial positions
        are sampled from ``[x_min, x_max]`` and, if clipping is applied, are
        never allowed to exceed this value.
    w : float
        Inertia weight that scales the previous velocity term. Larger values
        promote exploration by preserving momentum, while smaller values
        encourage exploitation and convergence.
    c1 : float
        Cognitive acceleration coefficient that weights the attraction of each
        particle toward its personal-best position (or its biogeographically
        recombined variant in PSO BGB mode).
    c2 : float
        Social acceleration coefficient that weights the attraction of each
        particle toward the global-best position found by the swarm.
    flag_bgb : bool
        If ``True``, activates the Biogeography-based recombination mechanism
        (PSO BGB) by replacing personal-best positions with
        ``get_personal_best_new`` before the velocity update. If ``False``,
        the algorithm reduces to standard PSO.
    use_global_best : float or None, optional
        Scaling factor for the global-best term in the velocity update. When
        set to 0, the global-best contribution is effectively disabled; when
        set to 1, standard PSO/PSO BGB behavior is recovered. If ``None``, it
        is assumed to be chosen appropriately by the calling code.

    Returns
    -------
    list[float]
        Test accuracies of the global-best particle at the end of each epoch.
        The list has length ``n_epochs``.
    """

    # Initialize PSO/PSO-BGB hyperparameters and derive iteration/logging schedule from evaluation budget
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, pop_size, x_min, x_max, w, c1, c2, flag_bgb, use_global_best=None) -> None:
        # Initialize base optimizer state (wrapper, evaluation budget per epoch, number of epochs)
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs)

        # Number of particles in the swarm
        self.pop_size        = pop_size
        # Lower bound for each parameter dimension
        self.x_min           = x_min
        # Upper bound for each parameter dimension
        self.x_max           = x_max
        # Inertia weight for velocity carry-over
        self.w               = w
        # Cognitive coefficient (pull toward personal best / recombined personal best)
        self.c1              = c1
        # Social coefficient (pull toward global best)
        self.c2              = c2

        # Toggle for biogeography-based recombination of personal bests
        self.flag_bgb        = flag_bgb
        # Scaling for the global-best term (0 disables, 1 enables standard behavior)
        self.use_global_best = use_global_best

        # Dimensionality of the parameter vector
        self.ind_size        = self.wrapper.get_ind_size()
        # Number of swarm iterations implied by evaluation budget (each iteration evaluates pop_size particles)
        self.n_iterations    = (n_evals_per_epoch*n_epochs)//pop_size

        # Iteration indices at which to log epoch-level test accuracy
        self.marker          = n_evals_per_epoch*np.arange(1, n_epochs+1, 1) // pop_size
        # Ensure the last marker aligns with the final iteration
        self.marker[-1]      = self.n_iterations


    # Construct a biogeographically recombined set of personal-best positions (PSO-BGB mode)
    def get_personal_best_new(self, mu, personal_best):

        # Compute immigration rates (lambda) from emigration scores mu (lambda = 1 - mu)
        lam               = 1 - mu 
        # Convert mu into a probability distribution over source habitats (particles) for gene migration
        p                 = mu / mu.sum()
                     
        # Allocate array for recombined personal bests
        personal_best_new = np.zeros((self.pop_size, self.ind_size))

        # For each particle i, build a new personal-best chromosome by migrating genes with probability lam[i]
        for i in range(self.pop_size):
            # Track whether at least one gene was changed; used to enforce at least one migration
            changed = False
            for k in range(self.ind_size):
                # With probability lam[i], replace gene k by sampling it from another particle's personal best
                if np.random.random() <= lam[i]:
                    personal_best_new[i,k] = personal_best[np.random.choice(a    = np.arange(self.pop_size),
                                                                            size = 1                       ,
                                                                            p    = p                       ), k]
                    # Mark that at least one gene has been migrated
                    changed                = True
                # Otherwise keep gene k from particle i's own personal best
                else:
                    personal_best_new[i,k] = personal_best[i,k]
            
            # If nothing changed, force a single random gene migration to avoid an identical copy
            if not changed:
                # Choose a random source particle j
                j                      = np.random.choice(self.pop_size) 
                # Choose a random gene index l
                l                      = np.random.choice(self.ind_size)
                # Replace that single gene to ensure variation
                personal_best_new[i,l] = personal_best[j,l]

        # Return the recombined personal-best population used in the next velocity update
        return personal_best_new



    # Run the PSO / PSO-BGB loop: update velocities/positions, update personal/global bests, and log test accuracy
    def run(self):

        # Initialize swarm positions uniformly within [x_min, x_max] for each parameter dimension
        X                     = np.random.uniform(low  = self.x_min                    ,
                                                  high = self.x_max                    , 
                                                  size = (self.pop_size, self.ind_size))
        
        # Initialize swarm velocities uniformly in [0, 1] (same shape as positions)
        V                     =                           np.random.uniform(low  = 0                            , 
                                                                            high = 1                            , 
                                                                            size = (self.pop_size, self.ind_size))

        # Initialize personal best positions to the initial positions
        personal_best         = X
        # Evaluate initial personal best fitness values (training loss) for each particle
        personal_best_fitness = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                    y_labels   = self.wrapper.y_train,
                                                                    individual = indvidual           )[0] for indvidual in personal_best]
        
        # Initialize global best as the personal best with the lowest training loss
        global_best           = personal_best[np.argmin(personal_best_fitness)]


         # Initialize list to store epoch-level test accuracies of the current global best
        accuracies = []

        # Create a progress bar over PSO iterations
        pbar = tqdm(range(1, self.n_iterations + 1))
        for generation in pbar:
            # Update progress bar label with current iteration number
            pbar.set_description(f"Currently running generation {generation}")

            # Sample two independent random vectors r1 and r2 in [0, 1] for stochastic PSO components
            r         = np.random.uniform(low  = 0                ,
                                          high = 1                ,
                                          size = (2, self.ind_size))

            # If PSO-BGB is enabled, recombine personal-best positions before the velocity update
            if self.flag_bgb:

                # Build recombined personal bests using rank-derived mu (higher rank -> higher migration probability)
                personal_best_new = self.get_personal_best_new(mu            = (np.argsort(personal_best_fitness) + 1) / self.pop_size,
                                                               personal_best = personal_best                                          )

                # Velocity update with recombined personal best and (optionally scaled) global-best attraction
                V                 = self.w*V + self.c1*r[0]*(personal_best_new-X) + self.use_global_best*self.c2*r[1]*(global_best-X)
                
            else:
                # Standard PSO velocity update using personal best and global best
                V                 = self.w*V + self.c1*r[0]*(personal_best    -X) +                      self.c2*r[1]*(global_best-X)
            
            # Position update: move particles according to updated velocities
            X                     = X + V

            # Evaluate current positions on training set (loss to minimize)
            fitness_X             = [self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_train,
                                                                        y_labels   = self.wrapper.y_train,
                                                                        individual = indvidual           )[0] for indvidual in X]
            
            # Update personal best positions where the new loss is better or equal than the stored personal-best loss
            personal_best[(fitness_X <= personal_best_fitness)] =        X[(fitness_X <= personal_best_fitness )]
            # Update personal best fitness values elementwise (take min of old and new fitness)
            personal_best_fitness                               = np.array([fitness_X,   personal_best_fitness ]).min(axis=0)
            # Update global best as the best personal best found so far (lowest personal-best loss)
            global_best                                         = personal_best[np.argmin(personal_best_fitness)]

            # At logging markers, evaluate global best on the test set and store accuracy
            if generation in self.marker:

                # Record test accuracy of the current global best
                accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test,
                                                                     y_labels   = self.wrapper.y_test,
                                                                     individual = global_best        )[1])
        
        # Sanity check: should have exactly one recorded accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit

        # Return epoch-level test accuracies of the global best
        return accuracies