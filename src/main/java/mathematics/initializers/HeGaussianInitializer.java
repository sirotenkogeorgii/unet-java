package main.java.mathematics.initializers;

import java.util.Random;

/**
 * Implements the He initialization strategy with a Gaussian distribution.
 * This initialization approach is designed specifically for layers with ReLU activation
 * functions to address the issue of vanishing gradients in deep networks. The method
 * adjusts the variance of the weights based on the number of incoming nodes to the layer
 * (fan-in), promoting more effective learning in deep architectures.
 */
public class HeGaussianInitializer implements IInitializer {
    private double std_;
    private Random sampler_;

    /**
     * Constructs a HeGaussianInitializer with the number of input nodes (fan-in) to the layer.
     * It calculates the standard deviation as sqrt(2.0 / nl) to use in the Gaussian distribution.
     *
     * @param nl The number of input nodes (fan-in); this value must not be zero.
     * @throws RuntimeException if nl is zero, as this would result in a division by zero in the calculation.
     */
    public HeGaussianInitializer(double nl) {
        if (nl == 0) throw new RuntimeException("nl to initialize value with He cannot be 0");
        std_ = Math.sqrt(2.0 / nl);
        sampler_ = new Random();
    }

    /**
     * Generates the next random weight initialization value using the Gaussian distribution.
     * The value is computed as a normally distributed random number with mean 0 and the calculated
     * standard deviation. This method is suitable for initializing weights in layers using ReLU activations.
     *
     * @return A double representing the next weight initialization value.
     */
    @Override
    public double next() { return std_ * sampler_.nextGaussian(0, 1); }
}
