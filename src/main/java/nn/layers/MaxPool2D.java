package main.java.nn.layers;

import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;

/**
 * Implements a max pooling layer for two-dimensional spatial data. This layer
 * reduces the spatial dimensions (height and width) of an input tensor by applying
 * a max pooling operation, which slides a window across the input tensor and
 * outputs the maximum value in each window.
 */
public class MaxPool2D extends Layer {
    private int pool_size_;

    /**
     * Constructs a MaxPool2D layer with a specified size for the pooling window.
     *
     * @param pool_size The size of the window to use for the max pooling operation,
     *                  typically a single integer that specifies the height and width
     *                  of a square window.
     */
    public MaxPool2D(int pool_size) {
        pool_size_ = pool_size;
    }

    /**
     * Applies the max pooling operation to the input tensor.
     *
     * @param inputs The input {@link MultiDimObject} expected to be a {@link Tensor}
     *               representing the data to which max pooling will be applied.
     * @return A new {@link Tensor} representing the result of the max pooling operation,
     *         with reduced dimensions based on the pool size.
     * @throws ClassCastException if the inputs are not an instance of {@link Tensor}.
     */
    public Tensor forward(MultiDimObject inputs) {
        return LayerFunctions.maxPool2D((Tensor)inputs, pool_size_);
    }

    /**
     * Retrieves the parameters of this layer. Since a max pooling layer does not have trainable
     * parameters, this method returns an empty list.
     *
     * @return An {@link ArrayList} of {@link MultiDimObject} which is empty, as there are no parameters.
     */
    public ArrayList<MultiDimObject> get_parameters() {
        return new ArrayList<>();
    }

    /**
     * Sets the execution mode of the layer.
     *
     * @param mode The execution mode as defined in {@link ModelSettings.executionMode}.
     */
    public void set_execution_mode(ModelSettings.executionMode mode) { }
}
