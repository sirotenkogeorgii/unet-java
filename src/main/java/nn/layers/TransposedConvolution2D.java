package main.java.nn.layers;

import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.mathematics.Tensor;

import static main.java.nn.layers.Convolution.Activation.NO;

public class TransposedConvolution2D extends Convolution {
    public TransposedConvolution2D(int in_channels, int out_channels, int kernel_size, int stride, int padding, boolean bias, String padding_mode, Activation activation) {
        if (stride < 1) throw new RuntimeException("Stride must be at least 1");
        stride_= stride;
        padding_ = padding;
        bias_ = bias ? new Matrix(out_channels, 1, IMultiDimObject.InitValues.ZEROS) : null;
        padding_mode_ = padding_mode;
        activation_ = activation == null ? NO : activation;
        kernels_ = new Tensor[out_channels];
        for (int i = 0; i < out_channels; ++i)
            kernels_[i] = new Tensor(kernel_size, kernel_size, in_channels, Tensor.InitValues.RANDOM);
    }
    public Tensor forward(IMultiDimObject tensor) {
        Matrix[] matrices = new Matrix[kernels_.length];
        Tensor casted_tensor = (Tensor) tensor;
        for (int i = 0; i < kernels_.length; ++i)
            matrices[i] = LayerFunctions.convolveTransposed2D(casted_tensor, kernels_[i], stride_, padding_);

        var result = bias_ == null ? new Tensor(matrices) : (new Tensor(matrices)).add_vector(bias_);
        return switch (activation_) {
            case ReLU -> result.relu();
            default -> result;
        };
    }
}
