package main.java.mathematics.initializers;

import java.util.Random;

public class RandomInitializer implements IInitializer {
    private Random sampler_;
    private double min_;
    private double max_;
    public RandomInitializer(double min, double max) {
        min_ = min;
        max_ = max;
        sampler_ = new Random();
    }
    public double next() { return sampler_.nextDouble(min_, max_); }
}
