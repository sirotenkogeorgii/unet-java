package main.java.optimizers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;

import java.util.ArrayList;

public abstract class Optimizer {
    protected ArrayList<IMultiDimObject> parameters_;
    protected double alpha_;
    public abstract void step();
    public abstract void set_zero_gradients();
}
