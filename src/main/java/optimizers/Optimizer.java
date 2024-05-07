package main.java.optimizers;

import main.java.mathematics.MultiDimObject;
import main.java.nn.models.ModelSettings;
import java.util.ArrayList;

/**
 * Abstract class representing an optimizer for neural network training. Optimizers are used to update the model's
 * parameters (such as weights and biases) based on the gradients computed during backpropagation.
 */
public abstract class Optimizer {
    protected ModelSettings.executionMode mode_ = ModelSettings.executionMode.SERIAL;
    protected ArrayList<MultiDimObject> parameters_;
    protected double alpha_;

    /**
     * Executes an optimization step to update the parameters of the model based on the computed gradients.
     * This method must be implemented by all subclasses to specify the exact optimization algorithm behavior.
     */
    public abstract void step();

    /**
     * Resets all accumulated gradients to zero in preparation for the next update cycle.
     * This is necessary to prevent accumulation of gradients across different batches of data.
     */
    public abstract void set_zero_gradients();
}
