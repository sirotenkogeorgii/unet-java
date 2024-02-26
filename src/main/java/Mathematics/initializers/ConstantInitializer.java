package main.java.mathematics.initializers;

import java.util.Random;

public class ConstantInitializer implements IInitializer {
    private double constant_;
    public ConstantInitializer(double constant) {
        constant_ = constant;
    }
    public double next() { return constant_; }
}
