package nn.layers;

import mathematics.MultiDimObject;
import nn.models.ModelSettings;

import java.util.ArrayList;

/**
 * Abstract base class for all neural network layers. This class defines the essential
 * methods that all neural network layers must implement to process input data,
 * manage layer-specific parameters, and handle various execution modes.
 */
public abstract class Layer {

    /**
     * Enumerates possible activation functions that can be used by neural network layers.
     * Activation functions are critical in neural networks as they introduce non-linearities
     * into the model, allowing the network to learn more complex patterns.
     * Possible activations: [ReLU, LeakyReLU, Sigmoid, Softmax, Identity].
     */
    public static enum Activation { ReLU, LeakyReLU, Sigmoid, Softmax, Identity }

    /**
     * The activation function to be used by this layer. Defaults to Identity,
     * meaning no activation is applied unless explicitly set.
     */
    protected Activation activation_ = Activation.Identity;

    /**
     * Processes the inputs using this layer's specific logic and returns the output.
     * This function is where the core computation of the layer takes place,
     * applying weights, biases, and the configured activation function to the inputs.
     *
     * @param inputs The inputs to the layer, encapsulated in a {@link MultiDimObject}.
     * @return The processed outputs, also encapsulated in a {@link MultiDimObject}.
     */
    public abstract MultiDimObject forward(MultiDimObject inputs);

    /**
     * Retrieves the parameters of this layer that are subject to training and adjustments
     * during the learning process. Typically includes weights and biases.
     *
     * @return An ArrayList of {@link MultiDimObject} containing the trainable parameters of this layer.
     */
    public abstract ArrayList<MultiDimObject> get_parameters();

    /**
     * Sets the execution mode for this layer based on the provided configuration,
     * affecting how operations are performed internally.
     *
     * @param mode The execution mode as defined by {@link ModelSettings.executionMode},
     *             which can be either parallel or serial.
     */
    public abstract void set_execution_mode(ModelSettings.executionMode mode);
}
