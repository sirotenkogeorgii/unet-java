package main.java.nn.models;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.nn.layers.LinearLayer;
import main.java.optimizers.IOptimizer;
import main.java.optimizers.SGD;

import java.util.ArrayList;

public class MLP implements IModel {
    private ArrayList<LinearLayer> layers_;

    public MLP(ArrayList<LinearLayer> layers) {
        if (layers == null || layers.isEmpty()) throw new RuntimeException("List of layers is empty");
        layers_ = layers;
    }

    public Matrix forward(Matrix input) {
        Matrix current_output = input;
        for (var layer: layers_)
            current_output = layer.forward(current_output);
        return current_output;
    }

    public ArrayList<Matrix> get_parameters() {
        var parameters = new ArrayList<Matrix>();
        for (var layer: layers_)
            parameters.addAll(layer.get_parameters());
        return parameters;
    }
}

class Program {
    static Value mse(Matrix matrix1, Matrix matrix2) {
        Value result = new Value(0);
        int[] matrix_shape = matrix1.get_size();
        for (int i = 0; i < matrix_shape[0]; ++i) {
            for (int j = 0; j < matrix_shape[1]; ++j) {
                Value temp = matrix1.get(i, j).sub(matrix2.get(i, j));
                result = result.add(temp.multiply(temp));
            }
        }
        return  result;
    }
    public static void main(String[] args) {
        var layers = new ArrayList<LinearLayer>();
//        layers.add(new LinearLayer(10, 10, true, "identity"));

//        layers.add(new LinearLayer(10, 20, true, "relu"));
//        layers.add(new LinearLayer(20, 30, true, "relu"));
//        layers.add(new LinearLayer(30, 10, false, "identity"));

        layers.add(new LinearLayer(10, 10, true, "relu"));
//        layers.add(new LinearLayer(10, 10, false, "identity"));

        var input_vector = new Matrix(10, 1, IMultiDimObject.InitValues.RANDOM);
        var target_vector = new Matrix(new double[][] {{1, 1, 1, 1, 1, 0, 0, 0, 0, 0}}).transpose();

        var mlp = new MLP(layers);
        IOptimizer optimizer = new SGD(mlp.get_parameters(), 0.01);

        for (int i = 0; i < 100; ++i) {
            Matrix output = mlp.forward(input_vector);

            output.transpose().print();

            Value mse_value = mse(output, target_vector);
            System.out.println();
            System.out.println(mse_value.get_value());

            mse_value.backward();
            optimizer.step();
            optimizer.set_zero_gradients();
        }
    }
}

