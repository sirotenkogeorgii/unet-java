package optimizers;

import autograd.Value;
import mathematics.MultiDimObject;
import nn.models.ModelSettings;
import java.util.ArrayList;
import java.util.stream.StreamSupport;

/**
 * Abstract class representing an optimizer for neural network training. Optimizers are used to update the model's
 * parameters (such as weights and biases) based on the gradients computed during backpropagation.
 */
public abstract class Optimizer {
    protected double gradient_clip_value = 2.0;

    protected ModelSettings.executionMode mode_ = ModelSettings.executionMode.SERIAL;
    protected ArrayList<MultiDimObject> parameters_;
    protected double alpha_;

    /**
     * Clips a gradient value to ensure it remains within a specified range.
     *
     * This method enforces that the gradient remains within the predefined bounds set by {@code gradient_clip_value}.
     * If the gradient exceeds these bounds, it is clipped to the nearest boundary value. This clipping helps in
     * controlling gradient explosion, a common problem in training deep neural networks where large gradients can
     * destabilize the learning process.
     *
     * @param gradient the gradient value to be clipped
     * @return the clipped gradient value. If the gradient is less than the negative of {@code gradient_clip_value},
     *         it will be clipped to this minimum. If it is more than {@code gradient_clip_value}, it will be clipped
     *         to this maximum.
     */
    public double clip_gradient(double gradient) {
        return Math.max(-gradient_clip_value, Math.min(gradient, gradient_clip_value));
    }

    /**
     * Executes an optimization step to update the parameters of the model based on the computed gradients.
     * This method must be implemented by all subclasses to specify the exact optimization algorithm behavior.
     */
    public abstract void step();

    /**
     * Resets the gradients of all parameters to zero. This is necessary before computing gradients
     * for a new batch to avoid accumulating gradients from multiple backward passes.
     */
    public void set_zero_gradients() {
        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            parameters_.parallelStream()
                    .flatMap(param -> StreamSupport.stream(param.spliterator(), false))
                    .forEach(val -> {
                        val.gradient = 0;
                    });
        } else {
            for (MultiDimObject param: parameters_) {
                for (Value val: param) {
                    val.gradient = 0;
                }
            }
        }
    }

}
