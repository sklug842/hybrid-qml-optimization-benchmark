# Import NumPy for vector/matrix operations, random sampling, and array handling
import numpy                         as     np
# Import TensorFlow for GradientTape autodiff, model execution, and optimizers (Adam/SGD)
import tensorflow                    as     tf
# Import gc to explicitly trigger garbage collection between offspring updates
import gc 
# Import tqdm to show a progress bar over epochs
from   tqdm                          import tqdm
# Import the abstract base class for gradient-based optimizers (provides wrapper + hyperparameter fields)
from   help_classes.abstract_classes import AbstractOptimizer_GB


# Define the Gradient Lexicase Selection (GLS) optimizer with Adam/SGD local updates
class GLS_Adam_SGD(AbstractOptimizer_GB):
    """
    Gradient-based evolutionary optimizer combining Ding et al.'s lexicase-style
    offspring selection with Adam/SGD local updates.

    GLS (Gradient Lexicase Selection) is a selection mechanism that chooses the
    next parent by filtering candidates across multiple “cases” (here: sampled
    mini-batches), emphasizing solutions that are consistently strong on at least
    some cases rather than only on an averaged objective.

    This class implements the optimizer family introduced by Ding & Spector
    (2023) for training neural networks with *gradient lexicase selection*,
    adapted here to variational quantum circuits via the ``wrapper`` interface.
    At each epoch, several independent “offspring” copies of the current
    solution are created and locally updated using a standard gradient-based
    optimizer (Adam or SGD). The best offspring, evaluated on a held-out
    mini-batch, is then selected as the parent for the next epoch.

    Algorithm overview
    ------------------
    Let ``ind_size`` be the dimensionality of the parameter vector returned by
    ``wrapper.get_ind_size()`` and ``X_train_size`` the size of the training
    set. The algorithm proceeds as follows:

      1. **Initialization**:
         * The initial individual (parameter vector) is obtained from
           ``wrapper.get_model().weights`` via ``wrapper.get_individual(...)``.
         * The number of data points used for each offspring's local update is
           precomputed as ``n_dp_train``, chosen such that the total number of
           model evaluations per epoch matches ``n_evals_per_epoch``.

      2. **Per-epoch loop** (for ``epoch = 1, ..., n_epochs``):
         a. **Offspring generation (local gradient updates)**:
            For each of the ``n_copies`` offspring:
              * A random subset of ``n_dp_train`` training examples is sampled
                without replacement.
              * A fresh Keras model is constructed by mapping the current parent
                individual to the model via:
                ``wrapper.map_individual(individual, wrapper.get_model())``.
              * A single gradient-based update step is performed on this subset
                using either:
                  - **Adam**: if ``optimizer == "Adam"`` with learning rate
                    ``eta`` and moment parameters ``beta_1``, ``beta_2``, or
                  - **SGD**: if ``optimizer == "SGD"`` with learning rate
                    ``eta`` and momentum ``momentum``.
              * The updated model weights are converted back into an individual
                via ``wrapper.get_individual(model.weights)``. This becomes one
                offspring candidate.

         b. **Selection on a separate batch**:
            * A separate mini-batch of size ``wrapper.batchsize`` is drawn from
              the training set.
            * Each offspring is evaluated on this mini-batch via
              ``wrapper.evaluate_single_final(...)``, and its scalar loss is
              recorded.
            * The offspring with the lowest loss on this batch is selected as
              the new parent ``individual`` for the next epoch.

         c. **Test-set evaluation (logging)**:
            * The current parent individual is evaluated on the test set via
              ``wrapper.evaluate_single_final(X_test, y_test, individual)``.
            * The resulting test accuracy (second return value) is appended to
              ``accuracies``. This yields one accuracy value per epoch.

      3. **Termination**:
         After ``n_epochs`` iterations, the method checks that the number of
         recorded accuracies matches ``n_epochs`` and returns the list.

    Internally, TensorFlow optimizers and models are constructed and discarded
    within each offspring update. To limit memory usage, the implementation
    clears the Keras backend and triggers garbage collection after each
    offspring mutation.

    Parameters
    ----------
    wrapper : object
        Object encapsulating model and data. It must provide:
          * ``X_train``, ``y_train``, ``X_test``, ``y_test``
                Training and test datasets.
          * ``batchsize`` : int
                Size of the mini-batch used for offspring selection.
          * ``get_ind_size() -> int``
                Returns the dimensionality of the parameter vector.
          * ``get_model() -> tf.keras.Model``
                Returns a (fresh) Keras model with trainable weights matching the
                parameterization.
          * ``map_individual(individual, model) -> tf.keras.Model``
                Assigns the parameter vector ``individual`` to the given model
                and returns the updated model.
          * ``get_individual(weights) -> np.ndarray``
                Extracts a parameter vector from a given list of model weights.
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)``
                Evaluates a single individual on the given data, returning a
                scalar loss (to be minimized) and a corresponding accuracy.

    n_evals_per_epoch : int
        Nominal number of model evaluations per epoch. Together with
        ``n_epochs``, ``n_copies`` and ``X_train_size``, this determines
        ``n_dp_train``, i.e., the size of the mini-subset used for each
        offspring's local gradient update.
    n_epochs : int
        Number of epochs (outer iterations). One round of offspring generation,
        selection, and test-set evaluation is performed per epoch.
    n_copies : int
        Number of parallel offspring generated from the current parent at each
        epoch. Larger values increase the diversity of candidate updates and the
        robustness of selection, at the cost of more gradient computations.
    optimizer : {"Adam", "SGD"}
        Choice of underlying gradient-based optimizer used in ``mutateSubGD``.
        If ``"Adam"``, TensorFlow's Adam optimizer is used; if ``"SGD"``,
        TensorFlow's SGD optimizer with momentum is used.
    eta : float
        Learning rate for the underlying gradient-based optimizer. Larger values
        can speed up learning but may lead to instability; smaller values yield
        more conservative updates.
    beta_1 : float, optional
        Exponential decay rate for the first-moment estimates in Adam
        (``beta_1`` parameter). Ignored if ``optimizer == "SGD"``.
    beta_2 : float, optional
        Exponential decay rate for the second-moment estimates in Adam
        (``beta_2`` parameter). Ignored if ``optimizer == "SGD"``.
    momentum : float, optional
        Momentum factor for SGD. Ignored if ``optimizer == "Adam"``.

    Returns
    -------
    list[float]
        Test accuracies of the selected (best) individual after each epoch.
        The list has length ``n_epochs``.
    """

    # Initialize the GLS optimizer state and precompute per-offspring subset size
    def __init__(self, wrapper, n_evals_per_epoch, n_epochs, n_copies, optimizer, eta, beta_1=0.9, beta_2=0.999, momentum=0.0) -> None:
        # Call base class constructor to register wrapper + evaluation budget + optimizer hyperparameters
        super().__init__(wrapper=wrapper, n_evals_per_epoch=n_evals_per_epoch, n_epochs=n_epochs, optimizer=optimizer, eta=eta, beta_1=beta_1, beta_2=beta_2, momentum=momentum)

        # Number of offspring copies generated per epoch
        self.n_copies               = n_copies

        # Cache training-set size to avoid recomputing len(...) repeatedly
        self.X_train_size           = len(self.wrapper.X_train)
        # Cache parameter vector dimensionality for convenience/consistency checks
        self.ind_size               = self.wrapper.get_ind_size()

        # Compute number of training points used per offspring local update so total evals align with n_evals_per_epoch
        self.n_dp_train             = int(((n_evals_per_epoch*self.X_train_size) - (self.wrapper.batchsize*n_copies)) / (n_evals_per_epoch * n_copies))

    
    # Perform one offspring mutation via a single mini-batch gradient step (Adam or SGD)
    def mutateSubGD(self, individual):
        """
        Perform a single local gradient–descent-based mutation of an individual.

        This method implements the *sub-gradient descent mutation* used inside
        the GLS/Ding-style procedure. It produces one offspring by:

          1. Sampling ``self.n_dp_train`` training examples without replacement.
          2. Instantiating a fresh model and mapping the parent parameter vector
             onto it via ``wrapper.map_individual``.
          3. Computing BCE loss on the sampled subset and taking one optimizer
             step (Adam or SGD) using TensorFlow ``GradientTape``.
          4. Converting the updated model weights back into a flat parameter
             vector via ``wrapper.get_individual``.
          5. Clearing TensorFlow state and forcing garbage collection to reduce
             memory growth when many offspring are created.

        :param individual: Current parent individual (flat parameter vector).
        :return:           Mutated offspring individual (flat parameter vector).
        :rtype:            numpy.ndarray
        """

        # Sample indices for a random subset of the training set (no replacement)
        # Generate subset of X_train and y_train
        idx          = np.random.choice(a       = self.X_train_size,
                                        size    = self.n_dp_train  ,
                                        replace = False            )

        # Slice training data/labels for the selected indices
        # Select data
        X_train_temp = self.wrapper.X_train[idx]
        y_train_temp = self.wrapper.y_train[idx]

        # Instantiate the requested TF optimizer for the local update
        # Define Optimizer
        if   self.optimizer == "SGD":
            # SGD with momentum (classical velocity smoothing)
            optimizer  = tf.keras.optimizers.SGD( learning_rate = self.eta     ,
                                                  momentum      = self.momentum)
        
        elif self.optimizer == "Adam": 
            # Adam with first/second moment estimates
            optimizer  = tf.keras.optimizers.Adam(learning_rate = self.eta   ,
                                                  beta_1        = self.beta_1,
                                                  beta_2        = self.beta_2)

        # Build a fresh model and assign the parent parameters into it
        # Get model by assigning individual to model
        model          = self.wrapper.map_individual(individual = individual              ,
                                                     model      = self.wrapper.get_model())

        # Record operations for autodiff
        with tf.GradientTape() as tape:

            # Forward pass on sampled subset (training=True enables BN/dropout behavior if present)
            # Calculate model prediction for X_train
            y_pred     = model(X_train_temp, training = True)

            # Compute BCE loss on subset (expects probabilities, from_logits=False)
            # Get loss value (BCE)
            loss_value = tf.keras.losses.binary_crossentropy(y_true      = tf.reshape(y_train_temp, [self.n_dp_train,]),
                                                             y_pred      = tf.reshape(y_pred      , [self.n_dp_train,]),
                                                             from_logits = False                                       )
            
            # Compute gradients of the loss w.r.t. trainable weights
            # Get gradients
            grads      = tape.gradient(target  = loss_value             ,
                                       sources = model.trainable_weights)
            
            # Apply one gradient update step
            # Execute gradient descent step
            optimizer.apply_gradients(zip(grads, model.trainable_weights))


        # Extract the updated parameters as a flat vector (offspring representation)
        individual = self.wrapper.get_individual(model.weights)

        # Drop references to large tensors/models to help memory reclamation
        del model, optimizer, grads, loss_value, y_pred, X_train_temp, y_train_temp, idx

        # Clear Keras/TensorFlow backend state to avoid graph accumulation across many offspring
        tf.keras.backend.clear_session()

        # Run garbage collection to free up memory
        gc.collect()
        
        # Return the mutated offspring parameter vector
        # Return individual
        return individual # self.wrapper.get_individual(model.weights)



    # Main optimization loop: generate offspring, select best on a held-out batch, log test accuracy per epoch
    def run(self):
        """
        Runs the GLS (Gradient Lexicase Selection) outer loop for ``n_epochs``.

        Per epoch, the algorithm:
          * creates ``n_copies`` offspring via :meth:`mutateSubGD`,
          * evaluates all offspring on a fresh selection mini-batch,
          * selects the offspring with minimum loss as the next parent,
          * evaluates that parent on the full test set and logs accuracy.

        :return: List of test accuracies (length ``self.n_epochs``).
        """

        # Initialize parent as the current weights of a fresh model (flattened into an individual vector)
        # Get individual
        individual           = self.wrapper.get_individual(self.wrapper.get_model().weights)

        # Store test accuracy after each epoch
        # Initialize array for accuracies
        accuracies           = []

        # Progress bar over epochs
        # Start generational process
        pbar = tqdm(range(1, self.n_epochs + 1))
        for epoch in pbar:
            # Update progress bar label
            pbar.set_description(f"Currently running epoch {epoch}")

            # Allocate container for offspring parameter vectors
            # Initialize array for offspring
            offspring        = [[] for _ in range(self.n_copies)]
           
            # Generate offspring by applying one local gradient step per copy
            # Train offspring
            for i in range(self.n_copies):
                offspring[i] = self.mutateSubGD(individual = individual)
                
            # Sample a fresh mini-batch used only for selection among offspring
            # Generate subset of X_train and y_train
            idx              = np.random.choice(a       = self.X_train_size     ,
                                                size    = self.wrapper.batchsize,
                                                replace = False                 )
            
            # Slice selection batch
            # Select data
            X_train_temp     = self.wrapper.X_train[idx]
            y_train_temp     = self.wrapper.y_train[idx]

            # Evaluate each offspring on the selection batch (loss is index [0])
            # Evaluate offspring
            evaluations      = [self.wrapper.evaluate_single_final(X_data     = X_train_temp,
                                                                   y_labels   = y_train_temp,
                                                                   individual = indvidual   )[0] for indvidual in offspring]
            
            # Select the offspring with minimum loss to become the next parent
            # Choose best individual
            individual       = offspring[np.argmin(evaluations)]

            # Evaluate selected parent on the full test set and log accuracy (index [1])
            # Evaluate best individual
            accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test,
                                                                 y_labels   = self.wrapper.y_test,
                                                                 individual = individual         )[1])


        # Sanity check: we expect exactly one logged accuracy per epoch
        if len(accuracies) != self.n_epochs:
            print("\nlen(accuracies != n_epochs")
            print("len(accuracies): ", len(accuracies)    )
            print("n_epochs:        ", self.n_epochs, "\n")
            exit()
        
        # Return per-epoch test accuracy trace
        return accuracies