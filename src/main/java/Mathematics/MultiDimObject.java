package main.java.mathematics;

import jdk.jshell.spi.ExecutionControl;
import main.java.autograd.Differentiable;
import main.java.autograd.Value;
import main.java.nn.models.ModelSettings;

public abstract class MultiDimObject implements Iterable<Value> {
    public static enum InitValues { ZEROS, ONES, HE, RANDOM }
    public ModelSettings.executionMode mode;
    protected int[] size_;
    public int[] get_size() { return size_; };
    public abstract Differentiable get(int... indices);
//    public abstract void set(IDifferentiable value, int... indices);
    public abstract MultiDimObject add(MultiDimObject other);
    public abstract MultiDimObject multiply(MultiDimObject other) throws ExecutionControl.NotImplementedException;
    public abstract boolean has_same_size(MultiDimObject other);
    public abstract boolean is_vector();
    public abstract void print();
}
