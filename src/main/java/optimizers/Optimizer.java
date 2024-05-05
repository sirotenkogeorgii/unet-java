package main.java.optimizers;

import main.java.mathematics.MultiDimObject;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;

public abstract class Optimizer {
    protected ModelSettings.executionMode mode_ = ModelSettings.executionMode.SERIAL;
    protected ArrayList<MultiDimObject> parameters_;
    protected double alpha_;
    public abstract void step();
    public abstract void set_zero_gradients();
}
