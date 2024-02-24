package main.java.nn.layers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.mathematics.Tensor;

import java.util.ArrayList;
import java.util.Arrays;

public abstract class Convolution implements ILayer {
    public static enum Activation { ReLU, NO }
    protected Activation activation_;
    protected int stride_;
    protected int padding_;
    protected String padding_mode_;
    protected Tensor[] kernels_;
    protected Matrix bias_;
    public abstract Tensor forward(IMultiDimObject tensor);

    public ArrayList<IMultiDimObject> get_parameters() {
        var parameters = new ArrayList<IMultiDimObject>();
        parameters.addAll(Arrays.asList(kernels_));
        if (bias_ != null) parameters.add(bias_);
        return parameters;
    }
}
