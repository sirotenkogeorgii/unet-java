package main.java.nn.models;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.mathematics.Tensor;
import main.java.nn.layers.ILayer;
import main.java.nn.layers.LinearLayer;

import java.util.ArrayList;
import java.util.Arrays;

public class Model {
    private ArrayList<ILayer> layers_;
    public Model(ArrayList<ILayer> layers) {
        if (layers == null || layers.isEmpty()) throw new RuntimeException("List of layers is empty");
        layers_ = layers;
    }

    public IMultiDimObject forward(IMultiDimObject input) {
        IMultiDimObject current_output = input;
        for (var layer: layers_)
            current_output = layer.forward(current_output);
        return current_output;
    }

    public ArrayList<IMultiDimObject> get_parameters() {
        var parameters = new ArrayList<IMultiDimObject>();
        for (var layer: layers_)
            parameters.addAll(layer.get_parameters());
        return parameters;
    }
}
