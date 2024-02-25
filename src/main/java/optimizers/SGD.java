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
            int i = 0;
            for (Value val: param) {
//                System.out.println(val.get_value());
                val.set_value(val.get_value() - alpha_ * val.get_gradient());
                i++;
            }
//            System.out.println(i);
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
