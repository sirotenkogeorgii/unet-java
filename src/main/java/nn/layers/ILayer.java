package main.java.nn.layers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;

import java.util.ArrayList;

public interface ILayer {
    public IMultiDimObject forward(IMultiDimObject inputs);
    public ArrayList<IMultiDimObject> get_parameters();
}
