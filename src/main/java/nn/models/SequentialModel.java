package nn.models;

import mathematics.MultiDimObject;
import nn.layers.Layer;

import java.util.ArrayList;

/**
 * Represents a neural network model structured as a sequence of layers. Each layer's output is fed sequentially
 * as input to the next layer. This is one of the most common architectures for neural networks.
 */
public class SequentialModel extends Model {
    private ArrayList<Layer> layers_;

    /**
     * Constructs a sequential neural network model with the specified layers and execution mode.
     *
     * @param layers A list of {@link Layer} objects that make up the model. The layers are processed in the order they appear in the list.
     * @param mode The execution mode, either SERIAL or PARALLEL, that specifies how the model should be executed.
     * @throws RuntimeException If the provided list of layers is empty or null.
     */
    public SequentialModel(ArrayList<Layer> layers, ModelSettings.executionMode mode)  {
        if (layers == null || layers.isEmpty()) throw new RuntimeException("List of layers is empty");
        layers_ = layers;
        mode_ = mode;

        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            for (var layer: layers_)
                layer.set_execution_mode(mode_);
        }
    }

    /**
     * Executes a forward pass through the entire sequence of layers using the given input.
     *
     * @param input The input {@link MultiDimObject} to be processed by the model starting at the first layer.
     * @return A {@link MultiDimObject} representing the output of the last layer after processing the initial input.
     */
    @Override
    public MultiDimObject forward(MultiDimObject input) {
        MultiDimObject current_output = input;
        for (var layer: layers_) {
            current_output = layer.forward(current_output);
        }
        return current_output;
    }

    /**
     * Retrieves all trainable parameters from each layer in the model. This is typically used for gradient
     * calculation and model updating during training.
     *
     * @return An {@link ArrayList} of {@link MultiDimObject} containing all trainable parameters of all layers in the model.
     */
    @Override
    public ArrayList<MultiDimObject> get_parameters() {
        var parameters = new ArrayList<MultiDimObject>();
        for (var layer: layers_)
            parameters.addAll(layer.get_parameters());
        return parameters;
    }
}
