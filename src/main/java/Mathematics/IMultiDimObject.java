package main.java.mathematics;

import jdk.jshell.spi.ExecutionControl;
import main.java.autograd.Value;

import java.util.Iterator;

public interface IMultiDimObject extends Iterable<Value> {
    public static enum InitValues { ZEROS, ONES, HE, RANDOM }
    public int[] get_size();
    public IMultiDimObject add(IMultiDimObject other);
    public IMultiDimObject multiply(IMultiDimObject other) throws ExecutionControl.NotImplementedException;
    public boolean has_same_size(IMultiDimObject other);
    public boolean is_vector();
    public void print();
}
