package main.java.nn.losses;

import main.java.autograd.Value;
import main.java.mathematics.MultiDimObject;

public abstract class Loss {
    protected Value loss_value_;
    public Value get_loss() { return loss_value_; }
    public abstract void calculate_loss(MultiDimObject input, MultiDimObject target);
    public abstract void backward();
}
