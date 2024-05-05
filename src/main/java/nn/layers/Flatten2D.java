package main.java.nn.layers;

import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;

public class Flatten2D implements ILayer {
    public Matrix forward(MultiDimObject input) {
        return LayerFunctions.flatten((Tensor)input);
    }
    public ArrayList<MultiDimObject> get_parameters() { return new ArrayList<>(); }

    public void set_execution_mode(ModelSettings.executionMode mode) { }
}
