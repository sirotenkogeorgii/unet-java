package main.java.nn.layers;

import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;

public class LinearLayer implements ILayer {
    private Matrix weights_;
    private Matrix bias_;
    private String activation_;
    public LinearLayer(int in_features, int out_features, boolean bias, String activation) {
        weights_ = new Matrix(out_features, in_features, MultiDimObject.InitValues.HE);
        bias_ = bias ? new Matrix(out_features, 1, MultiDimObject.InitValues.ZEROS) : null;
        activation_ = activation;
    }

    public Matrix forward(MultiDimObject input) {
        long startTime = System.nanoTime();

        if (input == null) throw new RuntimeException("Input is null");
        if (!input.is_vector()) throw new RuntimeException("Input of the linear layer must be a vector");
        var output_matrix = weights_.multiply(input);
        if (bias_ != null) output_matrix = output_matrix.add(bias_);

        var result = switch (activation_) {
            case "identity" -> output_matrix;
            case "relu" -> output_matrix.relu();
            case "sigmoid" -> output_matrix.sigmoid();
            default -> throw new RuntimeException("Unknown activation function");
        };

        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        System.out.println("Execution time in linear in milliseconds: " + executionTime / 1_000_000);
        return result;
    }

    public void set_execution_mode(ModelSettings.executionMode mode) {
        if (bias_ != null) bias_.mode = mode;
        weights_.mode = mode;
    }

    public ArrayList<MultiDimObject> get_parameters() {
        var parameters = new ArrayList<MultiDimObject>();
        parameters.add(weights_);
        if (bias_ != null) parameters.add(bias_);
        return parameters;
    }

    public void print() { weights_.print(); }
}