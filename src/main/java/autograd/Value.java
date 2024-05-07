package autograd;

import autograd.utils.GraphSorter;
import java.util.ArrayList;

/**
 * Represents a node in a computational graph that can perform automatic differentiation.
 * Each node (`Value`) can perform basic arithmetic operations and supports gradient computation
 * necessary for the backpropagation algorithm in neural networks.
 */
public class Value extends Differentiable {
    private ArrayList<Value> parents_;

    /**
     * Constructs a new Value with an initial scalar and marks it to require gradient computation.
     * @param value_ The initial scalar value of this node.
     */
    public Value(double value_) {
        value = value_;
        gradient = 0;
        parents_ = new ArrayList<>();
        prop_func_ = () -> {};
        requires_grad = true;
    }

    /**
     * Constructs a new Value with an initial scalar and a flag indicating whether it requires gradient computation.
     * @param value_ The initial scalar value of this node.
     * @param requires_grad_ Flag indicating whether the node should compute gradients.
     */
    public Value(double value_, boolean requires_grad_) {
        value = value_;
        gradient = 0;
        parents_ = new ArrayList<>();
        prop_func_ = () -> {};
        requires_grad = requires_grad_;
    }

    /**
     * Returns the parents of this value in the computational graph.
     * @return An ArrayList of Value instances that are parents of this value.
     */
    public ArrayList<Value> get_parents() {
        return parents_;
    }

    /**
     * Static method to add a list of Value instances, supporting automatic differentiation.
     * @param values The list of Value instances to be added.
     * @return A new Value instance representing the sum of the provided values.
     * @throws NullPointerException if the provided list of values is null.
     */
    public static Value add(ArrayList<Value> values) {
        if (values == null) throw new NullPointerException("Attempt to sum null array");
        Value new_value = new Value(0);
        for (Value current_value: values) {
            new_value.value += current_value.value;
            if (current_value.requires_grad) new_value.parents_.add(current_value);
        }
        new_value.prop_func_ = () -> {
            for (var current_parent: new_value.parents_) {
                current_parent.gradient += new_value.gradient;
            }
        };
        if (new_value.parents_.isEmpty()) new_value.requires_grad = false;
        return new_value;
    }

    /**
     * Adds another Value instance to this value, supporting automatic differentiation.
     * @param other The Value instance to be added to this value.
     * @return A new Value instance representing the sum.
     * @throws NullPointerException if the other value is null.
     */
    public Value add(Value other) {
        if (other == null) throw new NullPointerException("Attempt to add null value");
        var new_value = new Value(value + other.value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient;
            other.gradient += new_value.gradient;
        };
        if (requires_grad) new_value.parents_.add(this);
        if (other.requires_grad) new_value.parents_.add(other);
        if (new_value.parents_.isEmpty()) new_value.requires_grad = false;
        return new_value;
    }

    /**
     * Adds a constant to this value, supporting automatic differentiation.
     * @param constant The constant that will be converted to Value to be added.
     * @return A new Value instance representing the sum of this value and the constant.
     */
    public Value add(double constant) {
        return add(new Value(constant, false));
    }

    /**
     * Subtracts another Value instance from this value, supporting automatic differentiation.
     * @param other The Value instance to be subtracted from this value.
     * @return A new Value instance representing the difference.
     * @throws NullPointerException if the other value is null.
     */
    public Value sub(Value other) {
        if (other == null) throw new NullPointerException("Attempt to subtract  null value");
        return add(other.multiply(-1));
    }

    /**
     * Subtracts a constant from this value, supporting automatic differentiation.
     * @param constant The constant that will be converted to Value to be subtracted.
     * @return A new Value instance representing the difference after subtracting the constant.
     */
    public Value sub(double constant) {
        return sub(new Value(constant, false));
    }

    /**
     * Multiplies this Value by another Value, supporting automatic differentiation.
     * @param other The Value instance to multiply with this value.
     * @return A new Value instance representing the product.
     * @throws NullPointerException if the other value is null.
     */
    public Value multiply(Value other) {
        if (other == null) throw new NullPointerException("Attempt to multiply by null value");
        var new_value = new Value(value * other.value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * other.value;
            other.gradient += new_value.gradient * value;
        };
        if (requires_grad) new_value.parents_.add(this);
        if (other.requires_grad) new_value.parents_.add(other);
        if (new_value.parents_.isEmpty()) new_value.requires_grad = false;
        return new_value;
    }

    /**
     * Multiplies this value by a constant, supporting automatic differentiation.
     * @param constant The constant that will be converted to Value to multiply with this value.
     * @return A new Value instance representing the product of this value and the constant.
     */
    public Value multiply(double constant) {
        return multiply(new Value(constant, false));
    }

    /**
     * Raises the value of this {@code Value} object to the power of the given exponent.
     *
     * This method computes the power of the current {@code value} using the provided exponent, creating a new
     * {@code Value} object with the result. It also sets up the backpropagation function necessary for
     * gradient computation during the training phase if {@code requires_grad} is true.
     *
     * If {@code requires_grad} is set to true, the current {@code Value} object is added as a parent of the
     * new {@code Value} object to maintain the computation graph for gradient backpropagation.
     *
     * @param exponent the exponent to which the current value is to be raised
     * @return a new {@code Value} object whose value is this object's {@code value} raised to the given exponent
     * @throws IllegalArgumentException if the exponent causes the result to exceed numerical limits
     */
    public Value pow(double exponent) {
        var new_value = new Value(Math.pow(value, exponent));

        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * exponent * Math.pow(value, exponent - 1);
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    /**
     * Divides this {@code Value} object by another {@code Value} object.
     *
     * This method performs division by using the reciprocal of the other {@code Value} object.
     * It throws an exception if an attempt is made to divide by a null or zero value.
     *
     * @param other the {@code Value} object by which this object is to be divided
     * @return a new {@code Value} object representing the quotient of this object divided by the provided {@code Value}
     * @throws NullPointerException if the other {@code Value} object is null
     * @throws RuntimeException if the other {@code Value}'s value is zero, preventing division
     */
    public Value div(Value other) {
        if (other == null) throw new NullPointerException("Attempt to div by null value");
        if (other.value == 0) throw new RuntimeException("Attempt to div by zero");
        return multiply(other.pow(-1));
    }

    /**
     * Divides this {@code Value} object by a constant.
     *
     * This method calculates the division of this object's value by the given constant using the reciprocal of that constant.
     * It throws an exception if an attempt is made to divide by zero.
     *
     * @param constant the double value by which this {@code Value} object is to be divided
     * @return a new {@code Value} object representing the quotient of this object's value divided by the given constant
     * @throws RuntimeException if the constant is zero, as division by zero is undefined
     */
    public Value div(double constant) {
        if (constant == 0) throw new RuntimeException("Attempt to div by zero");
        return multiply((new Value(constant, false)).pow(-1));
    }

    /**
     * Applies the ReLU (Rectified Linear Unit) activation function to this value, supporting automatic differentiation.
     * ReLU function outputs the input itself if it is positive; otherwise, it outputs zero.
     * @return A new Value instance representing the result of the ReLU function.
     */
    public Value relu() {
        var new_value = new Value(value < 0 ? 0 : value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * (new_value.value > 0 ? 1 : 0);
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    /**
     * Applies the Leaky ReLU activation function to this value, supporting automatic differentiation.
     * Leaky ReLU outputs the input itself if it is positive; otherwise, it outputs the input multiplied by 0.01.
     * @return A new Value instance representing the result of the Leaky ReLU function.
     */
    public Value leakyRelu() {
        var new_value = new Value(value < 0 ? 0.01 * value : value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * (new_value.value > 0 ? 1 : 0.01);
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    /**
     * Applies the logarithm function to this value, supporting automatic differentiation.
     * This method uses a small offset for numerical stability (epsilon=1e-15) when the value is zero.
     * @return A new Value instance representing the natural logarithm of this value.
     */
    public Value log() {
        var new_value = value == 0 ? new Value(Math.log(1e-15)) : new Value(Math.log(value)); // Math.log(1e-15) = -34.538776

        new_value.prop_func_ = () -> {
            gradient += value == 0 ? new_value.gradient / 1e-15 : new_value.gradient / value;
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    /**
     * Applies the sigmoid activation function to this value, supporting automatic differentiation.
     * Sigmoid function outputs 1 / (1 + exp(-input)).
     * @return A new Value instance representing the result of the sigmoid function.
     */
    public Value sigmoid() {
        var new_value = new Value(1 / (1 + Math.exp(-value)));
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * new_value.value * (1 - new_value.value);
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    /**
     * Calculates the exponential of the current {@code Value} object.
     *
     * This method performs the exponential operation on the {@code value} field of the current {@code Value} object,
     * creates a new {@code Value} object with the result, and sets up the backpropagation function for gradient
     * computation if required.
     *
     * If {@code requires_grad} is {@code true} for the current object, the new {@code Value} object will have this object
     * as its parent in the computation graph. This relationship is used to propagate gradients back to this object during
     * the backpropagation phase of training.
     *
     * @return A new {@code Value} object whose value is set to the exponential of this object's {@code value}.
     *         The returned object will have its gradient computation and backpropagation settings configured
     *         based on the {@code requires_grad} status of the current object.
     */
    public Value exp() {
        var new_value = new Value(Math.exp(value));
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * Math.exp(value);
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    /**
     * Triggers the backward pass from this Value through the computational graph to compute gradients.
     */
    public void backward() {
        var sorter = new GraphSorter();
        var topological_order = sorter.topSort(this);

        gradient = 1;
        for (var value: topological_order) {
            value.prop_func_.run();
        }
    }
}