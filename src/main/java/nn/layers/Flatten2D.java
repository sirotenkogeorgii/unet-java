package main.java.nn.layers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.mathematics.Tensor;

import java.util.ArrayList;

public class Flatten2D implements ILayer {
    public Matrix forward(IMultiDimObject input) {
        return LayerFunctions.flatten((Tensor)input);
    }
    public ArrayList<IMultiDimObject> get_parameters() { return new ArrayList<>(); }
}
