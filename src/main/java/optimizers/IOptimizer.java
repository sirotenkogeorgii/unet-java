package main.java.optimizers;

public interface IOptimizer {
    public void step();
    public void set_zero_gradients();
}
