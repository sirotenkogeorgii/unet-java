package nn.losses;

import autograd.Value;
import mathematics.Matrix;
import mathematics.MultiDimObject;
import nn.layers.LayerFunctions;

/**
 * Implements Cross-Entropy Loss, which is typically used for classification problems.
 */
public class CrossEntropyLoss extends Loss {

    /**
     * Constructs a CrossEntropyLoss object with an initial loss value set to null.
     */
    public CrossEntropyLoss() {
        loss_value_ = null;
    }

    /**
     * Calculates the cross-entropy loss between the predicted outputs and actual targets.
     *
     * @param input The predicted outputs from the model as a {@link Matrix}, typically representing probabilities.
     * @param target The actual target outputs as a {@link Matrix}, typically representing gold distribution.
     * @return A {@link Value} representing the computed cross-entropy loss.
     */
    @Override
    public Value calculate_loss(MultiDimObject input, MultiDimObject target) {
        return LayerFunctions.cross_entropy_loss((Matrix)input, (Matrix)target);
    }

    /**
     * Adds a scalar value to the current loss value.
     *
     * @param value The scalar value to be added to the loss.
     */
    @Override
    public void add(double value) {
        if (loss_value_ == null) loss_value_ = new Value(value);
        else loss_value_.value += value;
    }

    /**
     * Adds another {@link Value} to the current loss value.
     *
     * @param value The {@link Value} to be added to the loss.
     */
    @Override
    public void add(Value value) {
        if (loss_value_ == null) loss_value_ = value;
        else loss_value_.value += value.value;
    }

    /**
     * Divides the current loss value by another {@link Value}.
     *
     * @param value The {@link Value} by which to divide the loss.
     */
    @Override
    public void divide(Value value) {
        if (loss_value_ == null) loss_value_ = new Value(0);
        else loss_value_.value /= value.value;
    }

    /**
     * Divides the current loss value by a scalar value.
     *
     * @param value The scalar value by which to divide the loss.
     */
    @Override
    public void divide(double value) {
        if (loss_value_ == null) loss_value_ = new Value(0);
        else loss_value_.value /= value;
    }

    /**
     * Executes the backward pass of the cross-entropy loss, computing gradients with respect to the model's parameters.
     * This method is critical for training models using backpropagation.
     */
    @Override
    public void backward() {
        loss_value_.backward();
    }
}
