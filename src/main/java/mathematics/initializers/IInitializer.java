package mathematics.initializers;

/**
 * Interface defining a method for generating initialization values.
 * Implementations of this interface provide a specific strategy for generating
 * the next value in a sequence.
 */
public interface IInitializer {

    /**
     * Generates the next value in the initialization sequence.
     * This method should be implemented to provide specific initialization values
     * according to the chosen strategy (e.g., uniform distribution, normal distribution).
     *
     * @return A double representing the next value.
     */
    double next();
}
