package main.java.autograd;

public abstract class Differentiable {
    static double gradient_clip_value = 10;
    public boolean requires_grad;
    public double value;
    public double gradient;
    protected Runnable prop_func_;
    public abstract void backward();
}
