package main.java.nn.layers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Tensor;

import java.util.ArrayList;

public class MaxPool2D implements ILayer {
    private int pool_size_;
    public MaxPool2D(int pool_size) { pool_size_ = pool_size; }
    public Tensor forward(IMultiDimObject inputs) {
        return LayerFunctions.maxPool2D((Tensor)inputs, pool_size_);
    }
    public ArrayList<IMultiDimObject> get_parameters() { return new ArrayList<>(); }
}
