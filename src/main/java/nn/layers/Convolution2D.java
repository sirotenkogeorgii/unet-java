package main.java.nn.layers;

import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.mathematics.Matrix;
import main.java.nn.models.ModelSettings;

import java.util.stream.IntStream;

/**
 * A 2D convolutional layer that applies a set of learned filters to the input data.
 * This layer is typically used in convolutional neural networks for feature extraction
 * from two-dimensional data such as images.
 */
public class Convolution2D extends Convolution {

    /**
     * Constructs a Convolution2D layer with specified parameters.
     *
     * @param in_channels The number of channels in the input tensor.
     * @param out_channels The number of filters to use in the convolution, determining the number of output channels.
     * @param kernel_size The size of each convolution filter.
     * @param stride The stride of the convolution operation.
     * @param padding The amount of padding applied to the input tensor.
     * @param bias Whether to include a bias term in the convolution.
     * @param activation The activation function to apply after the convolution.
     * @param mode The execution mode to use (sequential or parallel) which can affect performance.
     * @throws RuntimeException If the stride is less than 1.
     */
    public Convolution2D(int in_channels, int out_channels, int kernel_size, int stride,
                         int padding, boolean bias, Activation activation, ModelSettings.executionMode mode) {

        if (stride < 1) throw new RuntimeException("Stride must be at least 1");
        mode_ = mode;
        stride_= stride;
        padding_ = padding;
        bias_ = bias ? new Matrix(out_channels, 1, MultiDimObject.InitValues.ZEROS) : null;
        activation_ = activation;
        kernels_ = new Tensor[out_channels];
        for (int i = 0; i < out_channels; ++i)
            kernels_[i] = new Tensor(kernel_size, kernel_size, in_channels, Tensor.InitValues.HE);
    }

    /**
     * Applies the convolution operation to the input tensor, adds bias if configured, and passes the result through
     * the specified activation function.
     *
     * @param tensor The input tensor to be convolved.
     * @return A {@link Tensor} that is the result of applying the convolution, bias, and activation function.
     * @throws ClassCastException If the input is not an instance of {@link Tensor}.
     */
    @Override
    public Tensor forward(MultiDimObject tensor) {
        Matrix[] matrices = new Matrix[kernels_.length];
        Tensor casted_tensor = (Tensor)tensor;

        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            IntStream.range(0, kernels_.length).parallel().forEach(i ->
                    matrices[i] = LayerFunctions.convolve2D(casted_tensor, kernels_[i], stride_, padding_));
        } else {
            for (int i = 0; i < kernels_.length; ++i)
                matrices[i] = LayerFunctions.convolve2D(casted_tensor, kernels_[i], stride_, padding_);
        }

        var result = bias_ == null ? new Tensor(matrices) : (new Tensor(matrices)).add_vector(bias_);

        return switch (activation_) {
            case ReLU -> result.relu();
            case LeakyReLU -> result.leakyRelu();
            case Sigmoid -> result.sigmoid();
            case Identity -> result;
            default -> throw new RuntimeException("Unknown activation function");
        };
    }
}