package main.java.optimizers;

import main.java.mathematics.Matrix;
import main.java.autograd.Value;

import java.util.ArrayList;

public class SGD implements IOptimizer {
    private ArrayList<Matrix> parameters_;
    private double alpha_;
    public SGD(ArrayList<Matrix> parameters, double alpha) {
        parameters_ = parameters;
        alpha_ = alpha;
    }
    public void step() {
        for (Matrix matrix: parameters_) {
            int[] matrix_size = matrix.get_size();
            for (int i = 0; i < matrix_size[0]; ++i) {
                for (int j = 0; j < matrix_size[1]; ++j) {
                    Value current_param = matrix.get(i, j);
                    double new_value = current_param.get_value() - alpha_ * current_param.get_gradient();
                    current_param.set_value(new_value);
                }
            }
        }
    }

    public void set_zero_gradients() {
        for (Matrix matrix: parameters_) {
            int[] matrix_size = matrix.get_size();
            for (int i = 0; i < matrix_size[0]; ++i) {
                for (int j = 0; j < matrix_size[1]; ++j)
                    matrix.get(i, j).set_gradient(0);
            }
        }
    }
}
