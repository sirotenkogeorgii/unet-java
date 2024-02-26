package main.java.optimizers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.autograd.Value;

import java.util.ArrayList;

public class SGD extends Optimizer {

    public SGD(ArrayList<IMultiDimObject> parameters, double alpha) {
        parameters_ = parameters;
        alpha_ = alpha;
    }
    @Override
    public void step() {
        var names = new String[] {"conv1", "conv1_bias", "conv2", "conv2_bias", "ll1", "ll1_bias", "ll2", "ll2_bias"};
        int i = 0;
        for (IMultiDimObject param: parameters_) {

            System.out.println(names[i]);
//            System.out.printf("Current i: %d\n", i);

            for (Value val: param) {
                System.out.printf("Value %f Gradient %f\n", val.get_value(), val.get_gradient(), names.length);
                val.set_value(val.get_value() - alpha_ * val.get_gradient());
            }

            i++;
            System.out.println();

        }
    }

    @Override
    public void set_zero_gradients() {
        for (IMultiDimObject param: parameters_) {
            for (Value val: param) {
                val.set_gradient(0);
            }
        }
    }
}
