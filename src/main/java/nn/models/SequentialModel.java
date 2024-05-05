package main.java.nn.models;

import main.java.mathematics.MultiDimObject;
import main.java.nn.layers.ILayer;

import java.util.ArrayList;

public class SequentialModel extends Model {
    private ArrayList<ILayer> layers_;
    public SequentialModel(ArrayList<ILayer> layers, ModelSettings.executionMode mode)  {
        if (layers == null || layers.isEmpty()) throw new RuntimeException("List of layers is empty");
        layers_ = layers;
        mode_ = mode;

        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            for (var layer: layers_)
                layer.set_execution_mode(mode_);
        }
    }

    @Override
    public MultiDimObject forward(MultiDimObject input) {
        MultiDimObject current_output = input;
        for (var layer: layers_)
            current_output = layer.forward(current_output);
        return current_output;
    }

    @Override
    public ArrayList<MultiDimObject> get_parameters() {
        var parameters = new ArrayList<MultiDimObject>();
        for (var layer: layers_)
            parameters.addAll(layer.get_parameters());
        return parameters;
    }
}
