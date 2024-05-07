package main.java.nn.models;

import main.java.mathematics.MultiDimObject;

import java.util.ArrayList;

/**
 * Provides a base class for all neural network models. This abstract class enforces a common interface for
 * model operations, including forward pass computations, parameter management, and execution mode settings.
 */
public abstract class Model {
    protected ModelSettings.executionMode mode_ = ModelSettings.executionMode.SERIAL;

    /**
     * Retrieves the current execution mode of the model.
     *
     * @return The execution mode as defined by {@link ModelSettings.executionMode}, indicating whether the model
     *         operates in SERIAL or PARALLEL mode.
     */
    public ModelSettings.executionMode get_execution_mode() {
        return mode_;
    }

    /**
     * Processes the given input through the model's network structure and returns the output. This method is
     * typically used during both training and inference to compute the forward pass.
     *
     * @param input The input {@link MultiDimObject} to be processed by the model.
     * @return A {@link MultiDimObject} representing the output from the model after processing the input.
     */
    public abstract MultiDimObject forward(MultiDimObject input);

    /**
     * Retrieves a list of all parameters within the model that are subject to optimization during training.
     * This typically includes weights and biases of the neural network layers.
     *
     * @return An {@link ArrayList} of {@link MultiDimObject} containing the trainable parameters of the model.
     */
    public abstract ArrayList<MultiDimObject> get_parameters();
}
