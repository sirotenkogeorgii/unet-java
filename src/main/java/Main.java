package main.java;

import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.layers.*;
import main.java.nn.losses.BCELoss;
import main.java.nn.losses.Loss;
import main.java.nn.models.ModelSettings;
import main.java.nn.models.SequentialModel;
import main.java.optimizers.Optimizer;
import main.java.optimizers.SGD;

import java.util.ArrayList;

public class Main {
        public static void main(String[] args) {
            ModelSettings.executionMode mode = ModelSettings.executionMode.PARALLEL;
//        ModelSettings.executionMode mode = ModelSettings.executionMode.SERIAL;
//
            ArrayList<ILayer> layers = new ArrayList<>();

            layers.add(new Convolution2D(3, 3, 3, 1, 0, true, null, Convolution.Activation.ReLU, mode));
            layers.add(new Convolution2D(3, 1, 3, 1, 0, true, null, Convolution.Activation.ReLU, mode));
            layers.add(new MaxPool2D(2));
            layers.add(new Flatten2D());
            layers.add(new LinearLayer(12*12*1, 1000, true, "relu"));
            layers.add(new LinearLayer(1000, 10, true, "relu"));
            layers.add(new LinearLayer(10, 1, true, "sigmoid"));

            var input_image = new Tensor(28, 28, 3, MultiDimObject.InitValues.RANDOM);
            input_image.set_requires_grad(false);
            var target = new Matrix(new double[][] {{1}}).transpose();

            var model = new SequentialModel(layers, mode);

            Optimizer optimizer = new SGD(model.get_parameters(), 0.01, mode);
            Loss loss = new BCELoss();

            for (int j = 0; j < 1; ++j) {
                long startTime = System.nanoTime();
                for (int i = 0; i < 10; ++i) {

                    var output = model.forward(input_image);

                    loss.calculate_loss(output, target);
                    loss.backward();

                    optimizer.step();
                    optimizer.set_zero_gradients();

                    System.out.printf("Loss: %f\n", loss.get_loss().value);
                    ((Matrix)output).transpose();
                }
                long endTime = System.nanoTime();
                long executionTime = endTime - startTime;
                System.out.println("Execution time in milliseconds: " + executionTime / 1_000_000);
            }

        }
}
