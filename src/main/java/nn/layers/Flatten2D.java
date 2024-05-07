package nn.layers;

import mathematics.Matrix;
import mathematics.MultiDimObject;
import mathematics.Tensor;
import nn.models.ModelSettings;

import java.util.ArrayList;

/**
 * A layer that flattens a 3D tensor into a 2D matrix. This is commonly used in neural networks
 * to convert the output from convolutional layers to a format suitable for fully connected layers.
 */
public class Flatten2D extends Layer {

    /**
     * Takes a multidimensional input, typically a {@link Tensor}, and flattens it into a {@link Matrix}.
     * This operation is essential when transitioning from convolutional layers to dense layers within a neural network.
     *
     * @param input The {@link MultiDimObject}, expected to be a {@link Tensor}, to be flattened.
     * @return A {@link Matrix} representing the flattened version of the input tensor.
     * @throws ClassCastException if the input is not an instance of {@link Tensor}.
     */
    public Matrix forward(MultiDimObject input) {
        return LayerFunctions.flatten((Tensor)input);
    }
    public ArrayList<MultiDimObject> get_parameters() {
        return new ArrayList<>();
    }

    /**
     * Sets the execution mode of the layer. Since the flatten operation does not depend on the mode of execution,
     * this method implementation is empty.
     *
     * @param mode The execution mode as defined in {@link ModelSettings.executionMode}.
     */
    public void set_execution_mode(ModelSettings.executionMode mode) { }
}
