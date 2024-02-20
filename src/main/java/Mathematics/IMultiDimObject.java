package main.java.mathematics;

import jdk.jshell.spi.ExecutionControl;

public interface IMultiDimObject {
    public static enum InitValues { ZEROS, RANDOM }
    public int[] get_size();
    public IMultiDimObject add(IMultiDimObject other);
    public IMultiDimObject multiply(IMultiDimObject other) throws ExecutionControl.NotImplementedException;
    public boolean has_same_size(IMultiDimObject other);
    public boolean is_vector();
}
