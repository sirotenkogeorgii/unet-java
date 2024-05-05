package main.java.nn.layers;

import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.mathematics.Matrix;
import main.java.nn.models.ModelSettings;

import java.util.stream.IntStream;

public class Convolution2D extends Convolution {
    public Convolution2D(int in_channels, int out_channels, int kernel_size, int stride,
                         int padding, boolean bias, String padding_mode, Activation activation, ModelSettings.executionMode mode) {

        if (stride < 1) throw new RuntimeException("Stride must be at least 1");
        mode_ = mode;
        stride_= stride;
        padding_ = padding;
        bias_ = bias ? new Matrix(out_channels, 1, MultiDimObject.InitValues.ZEROS) : null;
        padding_mode_ = padding_mode;
        activation_ = activation == null ? Activation.NO : activation;
        kernels_ = new Tensor[out_channels];
        for (int i = 0; i < out_channels; ++i)
            kernels_[i] = new Tensor(kernel_size, kernel_size, in_channels, Tensor.InitValues.HE);
    }

    public Tensor forward(MultiDimObject tensor) {
        long startTime = System.nanoTime();

        Matrix[] matrices = new Matrix[kernels_.length];
        Tensor casted_tensor = (Tensor)tensor;

//        if (mode_ == ModelSettings.executionMode.PARALLEL) {
        if (false) {
            IntStream.range(0, kernels_.length).parallel().forEach(i ->
                    matrices[i] = LayerFunctions.convolve2D(casted_tensor, kernels_[i], stride_, padding_));
        } else {
            for (int i = 0; i < kernels_.length; ++i)
                matrices[i] = LayerFunctions.convolve2D(casted_tensor, kernels_[i], stride_, padding_);
        }

        var result = bias_ == null ? new Tensor(matrices) : (new Tensor(matrices)).add_vector(bias_);

        var activated =  switch (activation_) {
            case ReLU -> result.relu();
            default -> result;
        };

        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        System.out.println("Execution time in convolution in milliseconds: " + executionTime / 1_000_000);

        return activated;
    }
}