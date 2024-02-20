package main.java.nn.layers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Tensor;

public interface ILayer {
    public IMultiDimObject forward(IMultiDimObject inputs);
}
