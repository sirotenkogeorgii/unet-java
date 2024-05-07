package nn.layers;

import mathematics.Matrix;
import mathematics.MultiDimObject;
import mathematics.Tensor;
import nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Abstract base class for convolution layers in neural networks. This class defines the common
 * functionalities for convolution operations, including the handling of kernels and biases.
 */
public abstract class Convolution extends Layer {
    protected ModelSettings.executionMode mode_;
    protected int stride_;
    protected int padding_;
    protected String padding_mode_;
    protected Tensor[] kernels_;
    protected Matrix bias_;

    /**
     * Processes the input through the convolution layer using configured kernels and strides.
     * This method must be implemented by subclasses to define specific convolution behaviors.
     *
     * @param tensor The input {@link MultiDimObject}, typically a {@link Tensor}, to be convolved.
     * @return A {@link Tensor} representing the output of the convolution operation.
     */
    @Override
    public abstract Tensor forward(MultiDimObject tensor);

    /**
     * Sets the execution mode for the layer, adjusting how tensors within this layer, such as kernels and biases,
     * work (parallel of serial execution).
     *
     * @param mode The execution mode as defined in {@link ModelSettings.executionMode}.
     */
    @Override
    public void set_execution_mode(ModelSettings.executionMode mode) {
        if (bias_ != null) bias_.mode = mode;
        for (var kernel: kernels_) kernel.mode = mode;
    }

    /**
     * Retrieves all trainable parameters of this convolution layer, including kernels and potentially biases.
     *
     * @return An {@link ArrayList} of {@link MultiDimObject} representing the trainable parameters of this layer.
     */
    @Override
    public ArrayList<MultiDimObject> get_parameters() {
        var parameters = new ArrayList<MultiDimObject>(Arrays.asList(kernels_));
        if (bias_ != null) parameters.add(bias_);
        return parameters;
    }
}
