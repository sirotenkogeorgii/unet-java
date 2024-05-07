package optimizers;

import autograd.Value;
import mathematics.MultiDimObject;
import nn.models.ModelSettings;

import java.util.ArrayList;

/**
 * Implements the Adam optimization algorithm, which computes adaptive learning rates for each parameter.
 * Adam combines ideas from Momentum and RMSProp, namely, it uses squared gradients to scale the learning rate and
 * it takes advantage of momentum by using moving averages of the gradients.
 */
public class Adam extends Optimizer {
    double momentum_rate1_;
    double momentum_rate2_;
    double epsilon_;
    int timestep;
    double[] previous_moment1_;
    double[] previous_moment2_;

    /**
     * Constructs an Adam optimizer with specified parameters and hyperparameters.
     *
     * @param parameters A list of {@link MultiDimObject} representing the parameters of the model to be optimized.
     * @param alpha The step size or learning rate.
     * @param momentum_rate1 The decay rate for the first moment estimates (similar to momentum in other optimizers).
     * @param momentum_rate2 The decay rate for the second moment estimates (controls the moving average of the squared gradients).
     * @param mode The execution mode, either SERIAL or PARALLEL, which specifies how operations are executed.
     */
    public Adam(ArrayList<MultiDimObject> parameters, double alpha, double momentum_rate1, double momentum_rate2, ModelSettings.executionMode mode) {
        parameters_ = parameters;
        alpha_ = alpha;
        mode_ = mode;
        momentum_rate1_ = momentum_rate1;
        momentum_rate2_ = momentum_rate2;
        epsilon_ = 0.0000001;
        timestep = 1;

        int param_nums = 0;
        for (var param: parameters_) {
            int[] current_size = param.get_size();
            int elements_num = 1;
            for (var dim: current_size) elements_num *= dim;
            param_nums += elements_num;
        }
        previous_moment1_ = new double[param_nums];
        previous_moment2_ = new double[param_nums];
    }

    /**
     * Performs a single optimization step to update parameters using the Adam method.
     * This involves calculating the first and second moment estimates and using these
     * to adjust the parameters.
     */
    @Override
    public void step() {
        int i = 0;
        for (MultiDimObject param: parameters_) {
            for (Value val: param) {
                double clipped_gradient = clip_gradient(val.gradient);
                double current_moment1 = momentum_rate1_ * previous_moment1_[i] + (1 - momentum_rate1_) * clipped_gradient;
                double current_moment2 = momentum_rate2_ * previous_moment2_[i] + (1 - momentum_rate2_) * clipped_gradient * clipped_gradient;
                previous_moment1_[i] = current_moment1;
                previous_moment2_[i] = current_moment2;
                double corrected_moment1 = current_moment1 / (1 - Math.pow(momentum_rate1_, timestep));
                double corrected_moment2 = current_moment2 / (1 - Math.pow(momentum_rate2_, timestep));
                val.value = val.value - alpha_ * (corrected_moment1 / (Math.sqrt(corrected_moment2) + epsilon_));
                i++;
            }
        }
        timestep++;
    }
}

