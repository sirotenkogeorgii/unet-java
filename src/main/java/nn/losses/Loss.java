package main.java.nn.losses;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;

public abstract class Loss {
    protected Value loss_value_;
    public abstract void calculate_loss(IMultiDimObject input, IMultiDimObject target);
    public abstract void backward();
}
