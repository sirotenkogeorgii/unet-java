package main.java.mathematics;

import jdk.jshell.spi.ExecutionControl;
import main.java.autograd.Differentiable;
import main.java.autograd.Value;
import main.java.nn.models.ModelSettings;

/**
 * Abstract class defining a multi-dimensional mathematical object that supports
 * operations such as addition and multiplication. This class is designed to be
 * integrated within neural network architectures and supports various initialization
 * modes and execution settings.
 */
public abstract class MultiDimObject implements Iterable<Value> {

    /**
     * Enumerates possible initialization values for instances of multi-dimensional objects.
     */
    public static enum InitValues { ZEROS, ONES, HE, RANDOM }

    /**
     * The execution mode that determines how operations on this object should be handled: parallel or serial execution.
     */
    public ModelSettings.executionMode mode;

    /**
     * Array storing the size of each dimension of the multi-dimensional object.
     */
    protected int[] size_;

    /**
     * Retrieves the size of each dimension of this multi-dimensional object.
     * @return An array of integers where each integer represents the size of a dimension.
     */
    public int[] get_size() { return size_; };

    /**
     * Retrieves a {@link Differentiable} element located at the specified indices within the multi-dimensional object.
     * @param indices A varargs parameter indicating the indices of the element to retrieve.
     * @return The {@link Differentiable} element at the specified indices.
     */
    public abstract Differentiable get(int... indices);

    /**
     * Adds another multi-dimensional object to this object.
     * @param other The {@link MultiDimObject} to add to this object.
     * @return A new {@link MultiDimObject} representing the result of the addition.
     */
    public abstract MultiDimObject add(MultiDimObject other);

    /**
     * Multiplies another multi-dimensional object with this object.
     * @param other The {@link MultiDimObject} to multiply with this object.
     * @return A new {@link MultiDimObject} representing the result of the multiplication.
     * @throws ExecutionControl.NotImplementedException If the operation is not implemented.
     */
    public abstract MultiDimObject multiply(MultiDimObject other) throws ExecutionControl.NotImplementedException;

    /**
     * Checks if another multi-dimensional object has the same size as this object.
     * @param other The {@link MultiDimObject} to compare with this object.
     * @return True if the other object has the same size in all dimensions, false otherwise.
     */
    public abstract boolean has_same_size(MultiDimObject other);

    /**
     * Determines whether this multi-dimensional object is a vector.
     * @return True if this object is a vector, false otherwise.
     */
    public abstract boolean is_vector();

    /**
     * Prints the contents or a representation of the multi-dimensional object.
     */
    public abstract void print();
}
