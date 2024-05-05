package main.java.nn.layers;

import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.Arrays;

public abstract class Convolution implements ILayer {
    public static enum Activation { ReLU, NO }
    protected ModelSettings.executionMode mode_;
    protected Activation activation_;
    protected int stride_;
    protected int padding_;
    protected String padding_mode_;
    protected Tensor[] kernels_;
    protected Matrix bias_;
    public abstract Tensor forward(MultiDimObject tensor);

    public void set_execution_mode(ModelSettings.executionMode mode) {
        if (bias_ != null) bias_.mode = mode;
        for (var kernel: kernels_) kernel.mode = mode;
    }

    public ArrayList<MultiDimObject> get_parameters() {
        var parameters = new ArrayList<MultiDimObject>(Arrays.asList(kernels_));
        if (bias_ != null) parameters.add(bias_);
        return parameters;
    }
}
