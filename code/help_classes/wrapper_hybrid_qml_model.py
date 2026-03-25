import itertools
from   multiprocessing               import Pool
from   datetime                      import datetime
import numpy                         as     np
import tensorflow                    as     tf
from   tensorflow.keras.models       import Model
from   tensorflow.keras              import layers
from   tensorflow.keras.layers       import BatchNormalization, Dense, Input
from   tensorflow.keras.regularizers import L2
import pennylane                     as     qml
from   help_classes.abstract_classes import AbstractWrapper
from   help_classes.myquantumlayer   import MyQuantumLayer


class Wrapper_Hybrid_QML_Model(AbstractWrapper):
    """
    Wrapper for a hybrid quantum-classical binary classification model.

    This wrapper encapsulates:
    
    * Loading and preprocessing of several tabular benchmark datasets
      (e.g., Adult, Breast Cancer, Heart Disease, IoT-23, Network).
    * Construction of a classical encoder (stacked autoencoder-like MLP) in Keras.
    * Construction of a PennyLane-based variational quantum circuit (QVC)
      wrapped as a custom Keras layer (:class:`MyQuantumLayer`).
    * A final classical dense output layer for binary classification.
    * Utilities to map between a flattened parameter vector (``individual``)
      and the full Keras model weights, enabling use with evolutionary /
      gradient-free optimizers.
    * Efficient evaluation helpers, including an optional multiprocessing
      path for CPU execution and a GPU path when available.

    The resulting model is a fully differentiable hybrid QML model that can
    also be driven by gradient-free optimizers through the individual-weight
    mapping methods.

    Parameters
    ----------
    dataset : {"adult", "breast cancer", "heart disease", "IoT-23", "network"}
        Name of the dataset to load. Determines the feature dimension and
        batch size as well as which `.npy` files are loaded from disk.
    n_shots : int
        Number of measurement shots used by the PennyLane device.
    max_threads : int
        Maximum number of worker processes used for parallel prediction on
        CPU when ``flag_GPU`` is False and ``max_threads > 1``.
    type_QVC : {"V1", "V2"}
        String flag selecting between two predefined ansatz architectures for
        the variational quantum circuit.
    n_qubits : int
        Number of qubits (wires) in the quantum circuit and output dimension
        of the QVC layer.
    n_layers : int
        Number of repeated variational layers/blocks in the QVC.

    Attributes
    ----------
    X_train, y_train, X_test, y_test : np.ndarray
        Training and test sets loaded from disk.
    shape : int
        Input feature dimension of the classical encoder.
    batchsize : int
        Batch size used in model training and batched prediction.
    flag_GPU : bool
        Whether a GPU device was detected and the PennyLane device is
        configured to use ``"lightning.gpu"`` instead of ``"default.qubit"``.
    """

    def __init__(self, dataset, n_shots, max_threads, type_QVC, n_qubits, n_layers) -> None:
        super().__init__()
        self.dataset                                         = dataset
        self.n_shots                                         = n_shots
        self.max_threads                                     = max_threads

        self.type_QVC                                        = type_QVC
        self.n_qubits                                        = n_qubits
        self.n_layers                                        = n_layers

        self.X_train, self.y_train, self.X_test, self.y_test = self.get_data()

        print("GPU: ", tf.config.list_physical_devices("GPU"))

        # Check if a GPU is avaiable
        if len(tf.config.list_physical_devices("GPU")) == 1:
            self.flag_GPU                                    = True
        else:
            self.flag_GPU                                    = False



    def get_data(self):
        """
        Load the selected dataset from disk and set shape / batch size.

        The method selects the correct file names and input dimensionality
        based on ``self.dataset``, then loads preprocessed NumPy arrays from
        ``./data/final/``. Labels are flattened to 1-D arrays.

        Returns
        -------
        train_data : np.ndarray
            Training feature matrix.
        train_labels : np.ndarray
            Training labels (flattened).
        test_data : np.ndarray
            Test feature matrix.
        test_labels : np.ndarray
            Test labels (flattened).
        """

        if   self.dataset == "adult":
            X_name         = "adult_f12_b256"
            y_name         = "adult"
            self.shape     = 12
            self.batchsize = 256

        elif self.dataset == "breast cancer":
            X_name         = "breast_cancer_f10_b64"
            y_name         = "breast_cancer"
            self.shape     = 10
            self.batchsize = 64

        elif self.dataset == "heart disease":
            X_name         = "heart_disease_f10_b32"
            y_name         = "heart_disease"
            self.shape     = 10
            self.batchsize = 32

        elif self.dataset == "IoT-23":
            X_name         = "iot_f11_b256"
            y_name         = "iot"
            self.shape     = 11
            self.batchsize = 256

        elif self.dataset == "network":
            X_name         = "network_f10_b256"
            y_name         = "network"
            self.shape     = 10
            self.batchsize = 256

        train_data         = np.load("./../data/" + "X_train_" + X_name + ".npy")
        train_labels       = np.load("./../data/" + "y_train_" + y_name + ".npy")
        test_data          = np.load("./../data/" + "X_test_"  + X_name + ".npy")
        test_labels        = np.load("./../data/" + "y_test_"  + y_name + ".npy")

        train_labels       =  train_labels.flatten()
        test_labels        =  test_labels.flatten( )

        return train_data, train_labels, test_data, test_labels


    def get_encoder(self, reg):
        """
        Build the classical encoder network (stacked dense + batch-norm).

        This encoder acts as a feature extractor or autoencoder-like front
        end for the hybrid model. It consists of several dense layers with
        sigmoid activation and L2 regularization, interleaved with
        batch-normalization layers.

        Parameters
        ----------
        reg : float
            L2 regularization factor passed to ``kernel_regularizer=L2(reg)`` 
            for the dense layers.

        Returns
        -------
        tf.keras.Model
            Keras model mapping input features of shape ``(self.shape,)`` to
            a low-dimensional encoded representation of size 6.
        """

        model_input        = Input(shape=(self.shape,))
        model_encoded_1    = Dense(10, activation="sigmoid", kernel_regularizer = L2(reg))(model_input       )
        model_encoded1_bn  = BatchNormalization(momentum=0)(model_encoded_1)
        model_encoded_2    = Dense(8 , activation="sigmoid", kernel_regularizer = L2(reg))(model_encoded1_bn )
        model_encoded2__bn = BatchNormalization(momentum=0)(model_encoded_2)
        model_encoded_3    = Dense(6 , activation="sigmoid", kernel_regularizer = L2(reg))(model_encoded2__bn)
        model_encoded3__bn = BatchNormalization(momentum=0)(model_encoded_3)

        # Return Autoencoder
        return Model(inputs=model_input, outputs=model_encoded3__bn)
    

    def get_model(self):
        """
        Construct the full hybrid quantum-classical Keras model.

        The model architecture is:

        1. Classical encoder built by :meth:`get_encoder`.
        2. Repetition/tiling of the encoded features along the feature axis
           (one block per QVC layer).
        3. A PennyLane-based variational quantum circuit (QVC) wrapped as
           :class:`MyQuantumLayer`, parameterized by ``self.type_QVC`` and
           ``self.n_layers``.
        4. A final dense output layer with a single sigmoid unit for binary
           classification.

        Internally, this method:

        * Configures a PennyLane device (GPU or CPU backend depending on
          ``self.flag_GPU``).
        * Defines helper functions and two ansatz variants (``qnode_V1`` and
          ``qnode_V2``).
        * Wraps the QNode into :class:`MyQuantumLayer`.
        * Assembles and compiles a Keras model with binary cross-entropy loss
          and accuracy metric.

        Returns
        -------
        tf.keras.Model
            Compiled Keras model representing the hybrid QML classifier.
        """

        def _hadamard_helper(n_qubits):
            for x in range(n_qubits):
                qml.Hadamard(wires = x)

        def _encoding_helper(n_qubits, inputs, i):
            for x in range(n_qubits):
                qml.RZ(phi = inputs[:, i*n_qubits*2 + 2*x+0], wires = x)
                qml.RY(phi = inputs[:, i*n_qubits*2 + 2*x+1], wires = x)

        def _cnot_helper_V1():
            qml.CNOT(wires = [0,1])
            qml.CNOT(wires = [0,2])
            qml.CNOT(wires = [1,2])

        def _cnot_helper_V2():
            qml.CNOT(wires = [2,0])
            qml.CNOT(wires = [1,2])
            qml.CNOT(wires = [0,1])


        def qnode_V1(inputs, weights):

            for i in range(self.n_layers):

                # Hadamars
                _hadamard_helper(self.n_qubits)
                qml.Barrier()

                # Encoding
                _encoding_helper(self.n_qubits, inputs, i)
                qml.Barrier()

                # Rotations
                for x in range(self.n_qubits):
                    qml.RZ(phi = weights[i*self.n_qubits*2 + 2*x + 0], wires = x)
                    qml.RY(phi = weights[i*self.n_qubits*2 + 2*x + 1], wires = x)
                qml.Barrier()
                
                if i == 0:
                    # CNOTs
                    _cnot_helper_V1()
                    qml.Barrier()
        

        def qnode_V2(inputs, weights):

            for i in range(self.n_layers):

                # Hadamars
                _hadamard_helper(self.n_qubits)
                qml.Barrier()

                # Encoding
                _encoding_helper(self.n_qubits, inputs, i)
                qml.Barrier()

                # RY rotations
                for x in range(self.n_qubits):
                    qml.RY(phi = weights[i*self.n_qubits*2 + 2*x+0], wires = x)
                qml.Barrier()

                # CNOTs
                _cnot_helper_V2()
                qml.Barrier()

                # RZ rotations
                for x in range(self.n_qubits):
                    qml.RZ(phi = weights[i*self.n_qubits*2 + 2*x+1], wires = x)
                qml.Barrier()
                
                # CNOTs
                _cnot_helper_V2()
                qml.Barrier()


        # If a GPU is available
        if self.flag_GPU:
            dev = qml.device("lightning.gpu"       ,
                             wires  = self.n_qubits,
                             shots  = self.n_shots )

        # If no GPU is available
        else:
            dev = qml.device("default.qubit"       ,
                             wires  = self.n_qubits,
                             shots  = self.n_shots )

        # Define qnode
        @qml.qnode(device = dev)

        def qnode(inputs, weights):
            """
            Quantum node implementing the variational quantum circuit (QVC).

            Parameters
            ----------
            inputs : array_like
                Encoded classical features fed to the QVC.
            weights : array_like
                Trainable quantum parameters for the chosen ansatz.

            Returns
            -------
            list of qml.measurements.ExpectationMP
                Expectation values of Pauli-Z on each qubit, one per wire.
            """

            inputs     = np.asarray(inputs    )
            weights    = np.asarray(weights[0])

            if   self.type_QVC == "V1":
                qnode_V1(inputs, weights)

            elif self.type_QVC == "V2":
                qnode_V2(inputs, weights)

            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # Initialize quantum layer (QVC)
        qlayer            = MyQuantumLayer(n_qubits      = self.n_qubits                     ,
                                           qnode         = qnode                             ,
                                           weight_shapes = (1, self.n_qubits*2*self.n_layers),
                                           output_dim    = self.n_qubits                     )
        
        # build
        qlayer.build((self.n_qubits*2*self.n_layers, self.batchsize))

        # Initialize encoder
        SAE_encoder       = self.get_encoder(reg = 0.1)

        # Repeated output from encoder
        model_encoded__bn = tf.tile(SAE_encoder.output, [1, self.n_layers])

        # Quantum layer (QVC)
        qc                = qlayer( model_encoded__bn         )

        # Output layer
        mlp4              = Dense(units      = 1        ,
                                  activation = "sigmoid")(qc)

        # Define model
        model             = Model(inputs     = SAE_encoder.input,
                                  outputs    = mlp4             )

        # Compile model
        model.compile(loss      = "binary_crossentropy",
                      optimizer = "adam"               ,
                      metrics   = ["accuracy"]         )

        # Return model
        return model


    def get_ind_size(self):
        """
        Compute the dimensionality of the flattened parameter vector.

        This method creates a fresh model via :meth:`get_model` and sums the
        number of trainable parameters across all layers. The result is the
        length of the one-dimensional ``individual`` vector used by
        gradient-free optimizers.

        Returns
        -------
        int
            Total number of trainable parameters in the hybrid model.
        """

        return np.sum([tf.keras.backend.count_params(w) for w in self.get_model().trainable_weights])


    def map_individual(self, individual, model):
        """
        Map a flattened parameter vector onto a Keras model instance.

        The method interprets the entries of ``individual`` in the correct
        order and reshapes them into the weight and bias tensors for each
        layer (Dense, BatchNormalization, and :class:`MyQuantumLayer`),
        updating the model in-place.

        Parameters
        ----------
        individual : sequence of float
            One-dimensional array containing all model parameters in the
            expected order.
        model : tf.keras.Model
            Model instance whose weights are to be overwritten.

        Returns
        -------
        tf.keras.Model
            The same model instance, but with weights set from ``individual``.
        """

        #
        input_dim = model.input_shape[1]
        if isinstance( model.layers[0], layers.InputLayer):
            units         = input_dim
            current_index = 0
        else:
            units         = model.layers[0].units
            model.layers[0].set_weights([np.array(individual[:input_dim*units]).reshape(input_dim,units),np.array(individual[input_dim*units:(input_dim+1)*units])])
            current_index = (input_dim+1)*units

        # Iterate over layers of model
        for i in range(1, len(model.layers)):

            if isinstance(model.layers[i], layers.Dense):

                model.layers[i].set_weights([np.array(individual[current_index:current_index+units*model.layers[i].units]).reshape(units,model.layers[i].units),
                                             np.array(individual[current_index+units*model.layers[i].units:current_index+(units+1)*model.layers[i].units])])

                current_index    +=(units+1)*model.layers[i].units
                units             = model.layers[i].units

            elif isinstance(model.layers[i], layers.BatchNormalization):

                if self.flag_GF:
                    gamma             = np.array(individual[current_index:current_index+units          ])
                    beta              = np.array(individual[current_index+units:current_index+2*units  ])
                    mean              = np.zeros(units)
                    std               = np.ones( units)
                    current_index    += 2*units

                else:
                    gamma             = np.array(individual[current_index        :current_index+1*units])
                    beta              = np.array(individual[current_index+1*units:current_index+2*units])
                    mean              = np.array(individual[current_index+2*units:current_index+3*units])
                    std               = np.array(individual[current_index+3*units:current_index+4*units])
                    current_index    += 4*units

                model.layers[i].set_weights([gamma, beta, mean, std])

            # eins von beiden unnoetig
            elif isinstance(model.layers[i], MyQuantumLayer):
                units             = model.layers[i].weight_shapes["weights"][1]
                model.layers[i].set_weights(np.array(individual[current_index:current_index+units]).reshape(1, self.n_qubits*2*self.n_layers))
                current_index    += units
                units             = self.n_qubits

        # Return the model with weights from individual
        return model


    def get_individual(self, model_weights):
        """
        Flatten model weights into a one-dimensional parameter vector.

        Parameters
        ----------
        model_weights : list of tf.Tensor
            List of trainable weight tensors (e.g., ``model.trainable_weights``).

        Returns
        -------
        list of float
            Flattened list of all parameters, in the same order expected by
            :meth:`map_individual`.
        """

        # Initialize array
        individual = []

        # Iterate over model weights
        for weights in model_weights:

            # Cast to array
            individual.extend(np.hstack(tup = weights.numpy()))

        # Return model weights as array
        return individual


    def _parallel_run(self, kwargs):
        """
        Helper function for parallel prediction in a worker process.

        Parameters
        ----------
        kwargs : dict
            Dictionary containing:
            ``"individual"`` : flattened parameter vector, and
            ``"X_k"``        : sub-batch of input data.

        Returns
        -------
        np.ndarray
            Flattened prediction for the given sub-batch ``X_k``.
        """

        # Map individual onto model
        model = self.map_individual(kwargs["individual"], self.get_model())

        # Return prediction
        return model(kwargs["X_k"], training = False).numpy().flatten()


    def _parallel_helper(self, X_data, individual):
        """
        Compute model predictions with optional multiprocessing.

        If a GPU is available or ``max_threads == 1``, predictions are
        computed in a single process using Keras' ``model.predict``.
        Otherwise, the input data is split into batches and predictions are
        computed in parallel using multiple processes, each invoking
        :meth:`_parallel_run`.

        Parameters
        ----------
        X_data : np.ndarray
            Input data on which to evaluate the model.
        individual : sequence of float
            Flattened parameter vector to be mapped onto a fresh model.

        Returns
        -------
        np.ndarray
            Flattened array of predictions for all samples in ``X_data``.
        """

        # No parallel execution
        if self.flag_GPU or self.max_threads == 1:

            # Map individual onto model
            model         = self.map_individual(individual, self.get_model())

            # Return prediction
            return model.predict(X_data, batch_size=self.batchsize, verbose=0).flatten()

        # Parallel execution
        else:

            # Number of batches
            n_batches     = len(X_data) // self.batchsize

            # Split X data into n_batches subsamples
            X_data        = np.array_split(X_data, n_batches)

            # Create kwargs_list
            kwargs_list   = [{"individual": individual,
                              "X_k"       : X_data[k] } for k in range(n_batches)]

            # Parallel execution of _parallel_run
            with Pool(min(n_batches, self.max_threads)) as pool:
                jobs      = pool.map(self._parallel_run, kwargs_list)
                y_predict = list(itertools.chain.from_iterable(jobs))

            # Return prediction
            return y_predict


    def bce(self, x, y, eps=10 ** -15):
        """
        Compute binary cross-entropy for a single prediction-label pair.

        Numerically safe clipping is applied to the prediction ``x`` using
        ``eps`` to avoid ``log(0)``.

        Parameters
        ----------
        x : float
            Predicted probability (``y_pred``) for a single sample.
        y : int or float
            True label for the sample (0 or 1).
        eps : float, optional
            Small epsilon used for numerical stability, by default 1e-15.

        Returns
        -------
        float
            Binary cross-entropy contribution for this single sample.
        """

        p = max(eps, min(1 - eps, x))
        z = y * np.log(p) + (1 - y) * np.log(1 - p)
        return -z


    def evaluate_single_final(self, X_data, y_labels, individual):
        """
        Evaluate the hybrid model on a given dataset for a given individual.

        This method maps the flattened parameter vector ``individual`` onto
        a model, computes predictions on ``X_data`` (with optional parallel
        execution), and then returns:

        * the mean binary cross-entropy over all samples, and
        * the classification accuracy (threshold 0.5).

        Parameters
        ----------
        X_data : np.ndarray
            Input data on which to evaluate the model.
        y_labels : np.ndarray
            Ground-truth binary labels corresponding to ``X_data``.
        individual : sequence of float
            Flattened parameter vector representing the current candidate
            solution in an optimization algorithm.

        Returns
        -------
        bce : float
            Mean binary cross-entropy over all samples.
        acc : float
            Classification accuracy (fraction of correct predictions).
        """

        start     = datetime.now()

        # Calculate model prediciton for X-data
        y_predict = self._parallel_helper(X_data, individual)

        # Calculate binary cross entropy
        bce       = round(np.mean([self.bce(y_predict[k], y_labels[k]) for k in range(len(y_predict))]), 4)

        # Calculate y-hat
        y_hat     = (np.array(y_predict) > .5).astype(int).flatten()

        # Calculate accuracy
        acc       = round(np.mean((y_labels - y_hat) == 0), 4)

        return bce, acc