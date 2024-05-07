package main.java.optimizers;

import main.java.autograd.Value;
import main.java.mathematics.MultiDimObject;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.stream.StreamSupport;

/**
 * Implements the Momentum optimization algorithm, which helps accelerate SGD in the relevant direction and dampens oscillations.
 * It does this by adding a fraction of the update vector of the past step to the current step's gradient vector.
 */
public class Momentum extends Optimizer {
    double momentum_rate_;
    double[] previous_momentum_;

    /**
     * Constructs a Momentum optimizer with specified parameters, learning rate, and momentum rate.
     *
     * @param parameters A list of {@link MultiDimObject} representing the parameters of the model to be optimized.
     * @param alpha The learning rate used to scale the gradient in the update step.
     * @param momentum_rate The rate at which previous momentum is incorporated into the current update.
     * @param mode The execution mode (parallel or serial) that dictates how operations are executed.
     */
    public Momentum(ArrayList<MultiDimObject> parameters, double alpha, double momentum_rate, ModelSettings.executionMode mode) {
        parameters_ = parameters;
        alpha_ = alpha;
        mode_ = mode;
        momentum_rate_ = momentum_rate;

        int param_nums = 0;
        for (var param: parameters_) {
            int[] current_size = param.get_size();
            int elements_num = 1;
            for (var dim: current_size) elements_num *= dim;
            param_nums += elements_num;
        }
        previous_momentum_ = new double[param_nums];
    }

    /**
     * Executes a single optimization step using the momentum method. This method updates each parameter
     * based on the gradient, the learning rate, and the incorporated momentum from the previous steps.
     */
    @Override
    public void step() {

        int i = 0;
        for (MultiDimObject param: parameters_) {
            for (Value val: param) {
                double current_momentum = momentum_rate_ * previous_momentum_[i] + alpha_ * val.gradient;
                previous_momentum_[i] = current_momentum;

//                if (current_momentum > 2 || current_momentum < -2)
//                System.out.println(current_momentum);

                val.value = val.value - current_momentum;
                i++;
            }
        }
    }

    /**
     * Resets the gradients of all parameters to zero. This is necessary before computing gradients
     * for a new batch to avoid accumulating gradients from multiple backward passes.
     */
    @Override
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
