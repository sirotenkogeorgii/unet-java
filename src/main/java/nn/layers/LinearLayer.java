package nn.layers;

import mathematics.Matrix;
import mathematics.MultiDimObject;
import nn.models.ModelSettings;

import java.util.ArrayList;

/**
 * Implements a linear or fully connected layer within a neural network. This layer
 * applies a linear transformation to the incoming data followed by an optional bias and
 * an activation function.
 */
public class LinearLayer extends Layer {
    private Matrix weights_;
    private Matrix bias_;

    /**
     * Constructs a LinearLayer with specified dimensions and optional bias.
     *
     * @param in_features The number of input features.
     * @param out_features The number of output features.
     * @param bias Whether or not to include a bias term.
     * @param activation The activation to be applied after the linear transformation.
     */
    public LinearLayer(int in_features, int out_features, boolean bias, Activation activation) {
        weights_ = new Matrix(out_features, in_features, MultiDimObject.InitValues.HE);
        bias_ = bias ? new Matrix(out_features, 1, MultiDimObject.InitValues.ZEROS) : null;
        activation_ = activation;
    }

    /**
     * Applies a linear transformation to the input data, adds bias if configured, and passes the result
     * through the specified activation function.
     *
     * @param input The input {@link MultiDimObject} expected to be a {@link Matrix} representing a vector.
     * @return A {@link Matrix} that is the result of the linear transformation, bias addition, and activation.
     * @throws RuntimeException If the input is null or not a vector as required.
     */
    @Override
    public Matrix forward(MultiDimObject input) {
        if (input == null) throw new RuntimeException("Input is null");
        if (!input.is_vector()) throw new RuntimeException("Input of the linear layer must be a vector");

        var output_matrix = weights_.multiply(input);
        if (bias_ != null) output_matrix = output_matrix.add(bias_);

//        print();

        return switch (activation_) {
            case Identity -> output_matrix;
            case ReLU -> output_matrix.relu();
            case LeakyReLU -> output_matrix.leakyRelu();
            case Sigmoid -> output_matrix.sigmoid();
            case Softmax -> output_matrix.softmax();
            default -> throw new RuntimeException("Unknown activation function for matrix");
        };
    }

    /**
     * Sets the execution mode for this layer.
     *
     * @param mode The execution mode as defined in {@link ModelSettings.executionMode}.
     */
    @Override
    public void set_execution_mode(ModelSettings.executionMode mode) {
        if (bias_ != null) bias_.mode = mode;
        weights_.mode = mode;
    }

    /**
     * Retrieves all trainable parameters of this layer, typically the weights and possibly the biases.
     *
     * @return An {@link ArrayList} of {@link MultiDimObject} representing the trainable parameters of this layer.
     */
    @Override
    public ArrayList<MultiDimObject> get_parameters() {
        var parameters = new ArrayList<MultiDimObject>();
        parameters.add(weights_);
        if (bias_ != null) parameters.add(bias_);
        return parameters;
    }

    /**
     * Prints the weights of the linear layer to the standard output.
     */
    public void print() {
        weights_.print();
    }
}