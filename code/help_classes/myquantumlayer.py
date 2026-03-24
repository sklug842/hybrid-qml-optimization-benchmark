import numpy      as np
import tensorflow as tf
import pennylane  as qml


class MyQuantumLayer(tf.keras.layers.Layer):
    """
    Custom TensorFlow Keras layer wrapping a PennyLane quantum circuit with
    manually implemented parameter-shift gradients for both inputs and
    trainable quantum parameters.

    This layer embeds a PennyLane ``qml.qnn.KerasLayer`` (stored in
    ``self.qlayer_layer_PS``) and exposes it as a standard Keras layer that
    can be used inside hybrid quantum–classical models. Forward evaluation is
    delegated to the underlying ``KerasLayer``, while the backward pass is
    overridden via ``@tf.custom_gradient`` in :meth:`my_caller`, where
    gradients are computed using the parameter-shift rule.

    In particular:

    * The trainable quantum weights are perturbed by ±π/2 along each
      parameter dimension to compute gradients with respect to the circuit
      parameters.
    * The classical inputs ``X_data`` are also perturbed by ±π/2 along each
      feature dimension to obtain gradients with respect to the layer inputs,
      enabling end-to-end backpropagation through preceding classical layers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used by the underlying quantum circuit, which also
        determines the output dimension of the expectation values.
    qnode : qml.QNode
        PennyLane quantum node implementing the variational quantum circuit
        to be wrapped by this layer.
    weight_shapes : tuple or sequence of int
        Shape of the trainable quantum parameters expected by ``qnode``.
        This is passed to ``qml.qnn.KerasLayer`` under the key ``"weights"``.
    output_dim : int
        Dimension of the output produced by the quantum circuit (number of
        expectation values returned by the QNode).

    Notes
    -----
    * The underlying quantum layer is created as ``qml.qnn.KerasLayer`` with
      ``trainable=True`` and name ``"QVC_PS"``.
    * The quantum weights are initialized uniformly in ``[-π, π]`` via
      :meth:`set_weights`.
    * The forward pass is invoked via :meth:`call`, which dispatches to
      :meth:`my_caller`. The latter returns both the output and a custom
      gradient function implementing parameter-shift for inputs and weights.
    * The output shape is reported as ``(batch_size, n_qubits)`` in
      :meth:`compute_output_shape`.
    """

    def __init__(self, n_qubits, qnode, weight_shapes, output_dim):
        super().__init__()

        self.n_qubits            = n_qubits        
        self.weight_shapes       = {"weights": weight_shapes}
        self.qlayer_layer_PS     = qml.qnn.KerasLayer(qnode         = qnode             ,
                                                      weight_shapes = self.weight_shapes,
                                                      output_dim    = output_dim        ,
                                                      name          = "QVC_PS"          ,
                                                      trainable     = True              )
        
        # Set weights of quantum layer
        self.set_weights(weights = np.random.uniform(-1, 1, self.weight_shapes["weights"]) * np.pi)


    def set_weights(self, weights):
        """
        Set the trainable quantum weights of the underlying PennyLane layer.

        Parameters
        ----------
        weights : array_like
            Array of rotation angles for the quantum circuit with shape
            compatible with ``self.weight_shapes["weights"]``. Typically this
            has shape ``(1, n_params)`` or similar, depending on the QNode
            definition.

        Notes
        -----
        The provided ``weights`` are wrapped in a single-element list to match
        the expected format of ``qml.qnn.KerasLayer.set_weights``.
        """

        # set weights
        self.qlayer_layer_PS.set_weights([weights])


    # Decorator
    @tf.custom_gradient

    # Define my_caller function
    def my_caller(self, X_data):
        """
        Forward pass through the quantum layer with a custom parameter-shift
        gradient.

        This function is wrapped with ``@tf.custom_gradient`` and thus returns
        both the forward output and a custom gradient callback. The forward
        output is obtained by calling the internal ``qml.qnn.KerasLayer``
        on ``X_data``. The gradient callback then implements parameter-shift
        for:

        * the trainable quantum parameters (``variables``), and
        * the classical inputs ``X_data``.

        Parameters
        ----------
        X_data : tf.Tensor or np.ndarray
            Input batch to the quantum layer, typically of shape
            ``(batch_size, n_features)``. The exact shape must match what the
            wrapped QNode expects.

        Returns
        -------
        output : tf.Tensor
            Output of the quantum circuit for the given inputs, typically
            of shape ``(batch_size, n_qubits)``.
        grad_fn : callable
            Custom gradient function used internally by TensorFlow during
            backpropagation. Users should not call this directly.
        """

        # Calculate output of the quantum layer for X_data
        output = self.qlayer_layer_PS(X_data)

        
        def grad_calculations(upstream, variables):
            """
            Custom gradient function implementing the parameter-shift rule.

            This inner function is invoked by TensorFlow during the backward
            pass. It computes gradients w.r.t. both the inputs ``X_data`` and
            the trainable quantum weights contained in ``variables`` by
            shifting each dimension by ±π/2 and applying the standard
            parameter-shift formula.

            Parameters
            ----------
            upstream : tf.Tensor
                Gradient of the loss with respect to the layer outputs
                (i.e., the upstream gradient dL/dOutput).
            variables : list of tf.Tensor
                List containing the trainable variable tensor(s) associated
                with the underlying ``qml.qnn.KerasLayer``. Here we assume
                ``variables[0][0]`` holds the flattened rotation angles.

            Returns
            -------
            input_gradients : tf.Tensor
                Gradient of the loss with respect to the inputs ``X_data``,
                with the same shape as ``X_data``.
            angle_gradients : list of tf.Tensor
                Single-element list containing the gradient of the loss with
                respect to the quantum circuit parameters, with shape
                compatible with ``variables[0][0]``.
            """

            # Shift rotation angles by plus pi/2
            angles_plus_pi_half  = np.ones((len(variables[0][0]), 1)) * variables[0][0] + np.eye(len(variables[0][0]))*(np.pi/2)

            # Shift rotation angles by minus pi/2
            angles_minus_pi_half = np.ones((len(variables[0][0]), 1)) * variables[0][0] - np.eye(len(variables[0][0]))*(np.pi/2)

            # Initialization of array for gradients of rotation angles
            angles_gradients     = np.zeros(len(variables[0][0]))
            
            # Iterate over rotation angles
            for i, _ in enumerate(iterable = variables[0][0]):

                # Set angles for plus pi/2
                self.qlayer_layer_PS.set_weights([np.array(angles_plus_pi_half[i]).reshape(self.weight_shapes["weights"])])

                # Calculate output for X_data
                plus_pi2            = self.qlayer_layer_PS(X_data)

                # Set angles for minus pi/2
                self.qlayer_layer_PS.set_weights([np.array(angles_minus_pi_half[i]).reshape(self.weight_shapes["weights"])])

                # Calculate output for X_data
                minus_pi2           = self.qlayer_layer_PS(X_data)

                # Calculate gradient of angle
                angle_gradient      = 0.5*(plus_pi2 - minus_pi2)

                # Set unshifted angles
                self.qlayer_layer_PS.set_weights([np.array(variables[0][0]).reshape(self.weight_shapes["weights"])])

                # Calculate gradient
                one_gradient        = tf.math.reduce_sum(angle_gradient * upstream)

                # Insert gradient into array
                angles_gradients[i] = one_gradient

 
            # Initialization of array for gradients of input (X_data)
            input_gradients = np.zeros(shape = X_data.shape)

            # Iterate over X_data
            for i, dp in enumerate(X_data):#   range(len(X_data)):

                # Shift data point dp by plus pi/2
                x_plus_pi_half     = np.ones((len(dp), 1)) * dp + np.eye(len(dp))*(np.pi/2)

                # Shift data point dp by minus pi/2
                x_minus_pi_half    = np.ones((len(dp), 1)) * dp - np.eye(len(dp))*(np.pi/2)

                # Calculate output of data point dp shifted by plus pi half
                plus_pi_2          = self.qlayer_layer_PS(x_plus_pi_half )

                # Calculate output of data point dp shifted by minus pi half
                minus_pi_2         = self.qlayer_layer_PS(x_minus_pi_half)

                # Calculate gradient of data point dp (input)
                input_gradient     = 0.5*(plus_pi_2 - minus_pi_2)

                # Caclulcate gradient
                one_gradient       = tf.math.reduce_sum(input_gradient * upstream[i], axis = -1)  # das funktioniert auch _5

                # Insert one_gradient into array
                input_gradients[i] = one_gradient

            # return gradients of input (X_data) and of rotion angles (weights of quantum layer)
            return tf.convert_to_tensor(input_gradients), [tf.convert_to_tensor(angles_gradients)]

        # return output of quantum layer for X_data and function for calculating gradients
        return output, grad_calculations

    
    def call(self, inputs):
        """
        Keras entry point for the forward pass.

        Parameters
        ----------
        inputs : tf.Tensor or np.ndarray
            Batch of input features passed to the quantum layer.

        Returns
        -------
        tf.Tensor
            Output of the quantum circuit for the given inputs, as produced
            by :meth:`my_caller`.
        """

        return self.my_caller(inputs)

    
    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the quantum layer.

        Parameters
        ----------
        input_shape : tf.TensorShape or tuple
            Shape of the layer input, typically ``(batch_size, n_features)``.

        Returns
        -------
        tf.TensorShape
            Output shape of the layer. By construction this is
            ``(batch_size, n_qubits)``, where ``n_qubits`` is the number of
            expectation values returned by the QNode.
        """
         
        return tf.TensorShape((input_shape[0], self.n_qubits))