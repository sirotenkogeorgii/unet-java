package main.java.mathematics.initializers;

/**
 * An initializer that provides a constant value for every initialization call.
 * This is particularly useful for setting all parameters of a layer or model to a uniform value,
 * such as initializing biases to zeros or ones in a neural network.
 */
public class ConstantInitializer implements IInitializer {
    private double constant_;

    /**
     * Constructs a new ConstantInitializer with a specified constant value.
     *
     * @param constant The constant value to be used for all initialization requests.
     */
    public ConstantInitializer(double constant) {
        constant_ = constant;
    }

    /**
     * Returns the constant initialization value set during the construction of this instance.
     * Each call to this method will return the same value.
     *
     * @return A double representing the constant value with which all parameters will be initialized.
     */
    @Override
    public double next() {
        return constant_;
    }
}
