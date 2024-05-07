package autograd;

/**
 * Represents an abstract base class for differentiable variables.
 * This class is designed to be extended by any class that represents a variable which
 * supports automatic differentiation.
 *
 * Variables of this type can hold a scalar value, accumulate gradients, and
 * participate in backpropagation algorithms by defining custom backward computation.
 */
public abstract class Differentiable {

    /**
     * The maximum allowable gradient magnitude for gradient clipping.
     * Gradient values exceeding this threshold will be clipped to this value.
     */
    static double gradient_clip_value = 10;

    /**
     * Indicates whether the variable should be considered during gradient computation.
     * If set to true, gradients for this variable will be computed; otherwise, they will not.
     */
    public boolean requires_grad;

    /**
     * The scalar value of this variable.
     */
    public double value;

    /**
     * The accumulated gradient of this variable as computed during the backward pass.
     */
    public double gradient;

    /**
     * A function (runnable) that encapsulates the backward propagation logic specific to
     * this variable. This function is defined during the forward pass and executed
     * during the backward pass.
     */
    protected Runnable prop_func_;

    /**
     * Triggers the backward computation for this variable, typically propagating the
     * gradient to variables it depends on.
     * Implementations of this method should define how the backward pass should be performed
     * for the specific type of variable, including how gradients are to be handled and propagated.
     */
    public abstract void backward();
}
