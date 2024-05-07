package optimizers;

import autograd.Value;
import mathematics.MultiDimObject;
import nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.stream.StreamSupport;

/**
 * Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
 * SGD updates model parameters by stepping in the direction opposite to the gradient,
 * scaled by a defined learning rate, across potentially all parameters in parallel or sequentially.
 */
public class SGD extends Optimizer {

    /**
     * Constructs an SGD optimizer with specified parameters.
     *
     * @param parameters A list of {@link MultiDimObject} representing the parameters of the model to be optimized.
     * @param alpha The learning rate used to scale the gradient in the update step.
     * @param mode The execution mode (parallel or serial) that dictates how operations are executed.
     */
    public SGD(ArrayList<MultiDimObject> parameters, double alpha, ModelSettings.executionMode mode) {
        parameters_ = parameters;
        alpha_ = alpha;
        mode_ = mode;
    }

    /**
     * Performs a parameter update using stochastic gradient descent. This method updates each parameter by moving
     * in the direction that minimizes the loss, proportional to the gradient and the learning rate.
     */
    @Override
    public void step() {
        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            parameters_.parallelStream()
                    .flatMap(param -> StreamSupport.stream(param.spliterator(), false))
                    .forEach(val -> val.value = val.value - alpha_ * clip_gradient(val.gradient));
        } else {
            for (MultiDimObject param: parameters_) {
                for (Value val: param) {
                    val.value = val.value - alpha_ * clip_gradient(val.gradient);
                }
            }
        }
    }
}
