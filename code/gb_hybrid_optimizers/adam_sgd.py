import tensorflow                    as     tf
import numpy                         as     np
from   tqdm                          import tqdm
from   help_classes.abstract_classes import AbstractOptimizer_GB
import gc

# Set backend to float64
tf.keras.backend.set_floatx("float64")


class Adam_SGD(AbstractOptimizer_GB):
    """
    Gradient-based training of the hybrid quantum–classical model using Adam or SGD.

    This optimizer performs standard mini-batch gradient-based training of the
    underlying model provided by ``wrapper``, using either Adaptive Moment
    Estimation (Adam) or Stochastic Gradient Descent (SGD with momentum) as the
    update rule. It directly optimizes the model parameters by iterating over
    the full training set for a fixed number of epochs and logging the test
    accuracy after each epoch.

    Algorithm overview
    ------------------
    1. **Model construction**:
       A fresh Keras model is obtained from ``wrapper.get_model()``. Its
       trainable weights define the parameterization that is being optimized.

    2. **Optimizer initialization**:
       Depending on the ``optimizer`` argument:
         * If ``optimizer == "Adam"``:
             - Use ``tf.keras.optimizers.Adam`` with learning rate ``eta``,
               first-moment decay ``beta_1`` and second-moment decay ``beta_2``.
         * If ``optimizer == "SGD"``:
             - Use ``tf.keras.optimizers.SGD`` with learning rate ``eta`` and
               momentum factor ``momentum``.

       These choices reflect the usual interpretations of the hyperparameters:
       ``eta`` controls the global step size, while ``beta_1``, ``beta_2``, and
       ``momentum`` govern the smoothing of first and second moment estimates or
       velocity in parameter space.

    3. **Epoch loop** (for ``epoch = 1, ..., n_epochs``):
       a. **Mini-batch training**:
          * The training data ``wrapper.X_train`` / ``wrapper.y_train`` are
            iterated in contiguous mini-batches of size ``wrapper.batchsize``.
          * For each batch, a forward pass is computed and the binary cross
            entropy loss is evaluated using
            ``tf.keras.losses.binary_crossentropy`` (``from_logits = False``).
          * Gradients of the loss w.r.t. the model's trainable weights are
            obtained via ``tf.GradientTape`` and applied with the selected
            optimizer.

       b. **Test-set evaluation**:
          * After one full pass over the training data, the current model weights
            are converted to an individual via:
            ``wrapper.get_individual(model.weights)``.
          * The wrapper is then used to evaluate this individual on the test
            set via:
            ``wrapper.evaluate_single_final(X_test, y_test, individual)``,
            and the resulting accuracy (second return value) is appended to
            the ``accuracies`` list.

    4. **Cleanup and return**:
       After ``n_epochs`` epochs, TensorFlow resources are released by clearing
       the Keras backend and running garbage collection. The list of test
       accuracies is returned.

    Parameters
    ----------
    wrapper : object
        Object encapsulating the model and data. It must provide:
          * ``X_train``, ``y_train``, ``X_test``, ``y_test`` :
                Training and test datasets.
          * ``batchsize`` : int
                Size of the mini-batches used during training.
          * ``get_model() -> tf.keras.Model`` :
                Returns a (fresh) Keras model whose weights are to be trained.
          * ``get_individual(weights) -> np.ndarray`` :
                Maps a list of model weights to a flat parameter vector
                (individual).
          * ``evaluate_single_final(X_data, y_labels, individual)
             -> (loss, accuracy)`` :
                Evaluates the given individual on the provided data and returns
                a scalar loss (to be minimized) and an accuracy.

    n_epochs : int
        Number of training epochs. Each epoch corresponds to one full pass over
        the training dataset.
    optimizer : {"Adam", "SGD"}
        Choice of underlying gradient-based optimization algorithm:
          * ``"Adam"`` uses Adam with parameters ``eta``, ``beta_1``, ``beta_2``.
          * ``"SGD"`` uses SGD with learning rate ``eta`` and momentum
            ``momentum``.
    eta : float
        Learning rate controlling the global step size of the parameter updates.
        Larger values speed up learning but can cause instability; smaller
        values yield more stable but slower convergence.
    beta_1 : float, optional
        Exponential decay rate for the first-moment estimates in Adam (``beta_1``).
        Ignored if ``optimizer == "SGD"``.
    beta_2 : float, optional
        Exponential decay rate for the second-moment estimates in Adam (``beta_2``).
        Ignored if ``optimizer == "SGD"``.
    momentum : float, optional
        Momentum factor for SGD, controlling the influence of past gradients on
        the current update. Ignored if ``optimizer == "Adam"``.

    Returns
    -------
    list[float]
        A list of test accuracies, one per epoch, obtained by evaluating the
        current model on the test set after each training epoch.
    """

    def __init__(self, wrapper, n_epochs, optimizer, eta, beta_1=0.9, beta_2=0.999, momentum=0.0) -> None:
        super().__init__(wrapper=wrapper, n_evals_per_epoch=0, n_epochs=n_epochs, optimizer=optimizer, eta=eta, beta_1=beta_1, beta_2=beta_2, momentum=momentum)
        
        self.wrapper  = wrapper
        self.n_epochs = n_epochs


    def run(self):
        """
        Runs the gradient based optimization

        :return: returns the model accuray on test data after 5, 10, 15 and 20 epochs
        """

        # Get model
        model                = self.wrapper.get_model()

        # Define Optimizer
        if   self.optimizer == "SGD":
            optimizer  = tf.keras.optimizers.SGD( learning_rate = self.eta     ,
                                                  momentum      = self.momentum)
        
        elif self.optimizer == "Adam": 
            optimizer  = tf.keras.optimizers.Adam(learning_rate = self.eta   ,
                                                  beta_1        = self.beta_1,
                                                  beta_2        = self.beta_2)
        
        # Initialization of array to save the model accuracy for test data after 5, 10, 15, and 20 training epochs
        accuracies = []

        # Start generational process
        pbar = tqdm(range(1, self.n_epochs + 1))
        for epoch in pbar:
            pbar.set_description(f"Currently running epoch {epoch}")
             
            # Iterate over batches
            for i in range(len(self.wrapper.X_train) // self.wrapper.batchsize):

                with tf.GradientTape() as tape:

                    # Calculate model prediction for X_train
                    y_pred     = model(self.wrapper.X_train[i*self.wrapper.batchsize:i*self.wrapper.batchsize+self.wrapper.batchsize], training = True)

                    # Get loss value (BCE)
                    loss_value = tf.keras.losses.binary_crossentropy(y_true      = tf.reshape(self.wrapper.y_train[i*self.wrapper.batchsize:i*self.wrapper.batchsize+self.wrapper.batchsize], [self.wrapper.batchsize,]),
                                                                     y_pred      = tf.reshape(y_pred                                                                                        , [self.wrapper.batchsize,]),
                                                                     from_logits = False                                                                                                                                )
                    
                    # Get gradients
                    grads      = tape.gradient(target  = loss_value             ,
                                               sources = model.trainable_weights)

                    # Execute gradient descent step
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Save accuray 
            accuracies.append(self.wrapper.evaluate_single_final(X_data     = self.wrapper.X_test                       ,
                                                                 y_labels   = self.wrapper.y_test                       ,
                                                                 individual = self.wrapper.get_individual(model.weights))[1])
        
        # Delete
        del model, optimizer, grads, loss_value, y_pred

        # Clear session
        tf.keras.backend.clear_session()

        # Run garbage collection to free up memory
        gc.collect()

        # Return final accuracies
        return accuracies