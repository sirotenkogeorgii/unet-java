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
        for (IMultiDimObject param: parameters_) {
            for (Value val: param) {
                System.out.println(val.get_gradient());
                double new_value = val.get_value() - alpha_ * val.get_gradient();
                val.set_value(new_value);
            }
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
