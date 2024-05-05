package main.java.mathematics.initializers;

import java.util.Random;

public class HeGaussianInitializer implements IInitializer {
    private double std_;
    private Random sampler_;
    public HeGaussianInitializer(double nl) {
        if (nl == 0) throw new RuntimeException("nl to initialize value with He cannot be 0");
        std_ = Math.sqrt(2 / nl);
        sampler_ = new Random();
    }
    public double next() { return sampler_.nextGaussian(0, std_); }
}
