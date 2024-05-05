package main.java.nn.layers;

import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;

public class MaxPool2D implements ILayer {
    private int pool_size_;
    public MaxPool2D(int pool_size) { pool_size_ = pool_size; }
    public Tensor forward(MultiDimObject inputs) {
        return LayerFunctions.maxPool2D((Tensor)inputs, pool_size_);
    }
    public ArrayList<MultiDimObject> get_parameters() { return new ArrayList<>(); }
    public void set_execution_mode(ModelSettings.executionMode mode) { }
}
