package main.java.nn.losses;

import main.java.autograd.Value;
import main.java.mathematics.MultiDimObject;

/**
 * Abstract base class for loss functions in neural networks. This class provides the structure and necessary methods
 * that all specific loss function implementations must adhere to. Loss functions evaluate how well the model performs
 * by comparing the predicted output to the actual target.
 */
public abstract class Loss {
    protected Value loss_value_;

    /**
     * Returns the current loss value calculated by the loss function.
     *
     * @return The current loss value as a {@link Value}. It can be null if the loss has not yet been calculated or has been reset.
     */
    public Value get_loss() {
        return loss_value_;
    }

    /**
     * Resets the loss value to null. This method is useful for clearing the loss information before a new calculation
     * or after gradients have been propagated.
     */
    public void reset() {
        loss_value_ = null;
    }

    /**
     * Calculates the loss between the predicted output and the target.
     *
     * @param input The predicted output from the model as a {@link MultiDimObject}.
     * @param target The actual target output as a {@link MultiDimObject}.
     * @return The calculated loss as a {@link Value}.
     */
    public abstract Value calculate_loss(MultiDimObject input, MultiDimObject target);

    /**
     * Adds a scalar value to the current loss value.
     *
     * @param value The scalar value to be added to the loss.
     */
    public abstract void add(double value);

    /**
     * Adds a {@link Value} to the current loss value.
     *
     * @param value The {@link Value} to be added to the loss.
     */
    public abstract void add(Value value);

    /**
     * Divides the current loss value by a scalar value.
     *
     * @param value The scalar value by which to divide the loss.
     */
    public abstract void divide(double value);

    /**
     * Divides the current loss value by a {@link Value}.
     *
     * @param value The {@link Value} by which to divide the loss.
     */
    public abstract void divide(Value value);

    /**
     * Performs backpropagation to calculate the gradient of the loss function with respect to the model's parameters.
     * This method is crucial for the training process, allowing the optimizer to adjust the parameters appropriately.
     */
    public abstract void backward();
}
