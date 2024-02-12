package main.java.nn.layers;

import main.java.mathematics.Matrix;
import main.java.autograd.Value;
import main.java.nn.models.IModel;

import java.util.ArrayList;

public class LinearLayer implements IModel {
    private Matrix weights_;
    private Matrix bias_;
    private String activation_;
    public LinearLayer(int in_features, int out_features, boolean bias, String activation) {
        weights_ = new Matrix(out_features, in_features);
        bias_ = bias ? new Matrix(out_features, 1) : null;
        activation_ = activation;
    }

    public Matrix forward(Matrix input) {
        if (!input.is_vector()) throw new RuntimeException("Input of the linear layer must be a vector");
        var output_matrix = weights_.multiply(input);
        if (bias_ != null) output_matrix = output_matrix.add(bias_);
        return switch (activation_) {
            case "identity" -> output_matrix;
            case "relu" -> output_matrix.relu();
            case "sigmoid" -> output_matrix.sigmoid();
            default -> throw new RuntimeException("Unknown activation function");
        };
    }

    public ArrayList<Matrix> get_parameters() {
        var parameters = new ArrayList<Matrix>();
        parameters.add(weights_);
        if (bias_ != null) parameters.add(bias_);
        return parameters;
    }

    public void print() { weights_.print(); }
}