package mathematics.initializers;

import java.util.Random;

/**
 * Implements {@link IInitializer} to provide random initialization values within a specified range.
 * This class uses Java's {@link Random} class to generate a random double value between a minimum
 * and maximum boundary.
 */
public class RandomInitializer implements IInitializer {
    private Random sampler_;
    private double min_;
    private double max_;

    /**
     * Constructs a {@code RandomInitializer} with specified minimum and maximum values for initialization.
     *
     * @param min The minimum value in the range from which the random initialization value will be drawn.
     * @param max The maximum value in the range from which the random initialization value will be drawn.
     */
    public RandomInitializer(double min, double max) {
        min_ = min;
        max_ = max;
        sampler_ = new Random();
    }

    /**
     * Generates the next random value within the specified range [min, max].
     *
     * @return A double representing the next random value within the specified initialization range.
     */
    @Override
    public double next() { return sampler_.nextDouble(min_, max_); }
}
