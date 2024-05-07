package main.java;

import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.layers.*;
import main.java.nn.losses.BCELoss;
import main.java.nn.losses.Loss;
import main.java.nn.models.ModelSettings;
import main.java.nn.models.SequentialModel;
import main.java.optimizers.Adam;
import main.java.optimizers.Momentum;
import main.java.optimizers.SGD;
import main.java.optimizers.Optimizer;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {

        // executionMode means whether some operations will be executed in parallel or serial
        ModelSettings.executionMode mode = ModelSettings.executionMode.SERIAL; // ModelSettings.executionMode.PARALLEL;

        /** An example of building a neural network.
         * In this example, a regular ArrayList of layers is created, where layers are added.
         * The sequential model is selected, so the operations will be executed in the order in which they were added to the ArrayList.
         */
        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new Convolution2D(3, 1, 3, 1, 0, true,  Layer.Activation.LeakyReLU, mode));
        layers.add(new Convolution2D(1, 1, 3, 1, 0, true,  Layer.Activation.LeakyReLU, mode));
        layers.add(new MaxPool2D(2));
        layers.add(new Flatten2D());
        layers.add(new LinearLayer(12 * 12 * 1, 50, true, Layer.Activation.LeakyReLU));
        layers.add(new LinearLayer(50, 10, true, Layer.Activation.LeakyReLU));
        layers.add(new LinearLayer(10, 1, true, Layer.Activation.Sigmoid));
        var model = new SequentialModel(layers, mode);

        /**
         * For example, let's generate some data.
         */
        ArrayList<MultiDimObject> data = new ArrayList<>();
        ArrayList<MultiDimObject> targets = new ArrayList<>();
        int data_samples = 40; // 40 data samples
        for (int i = 0; i < data_samples; ++i) {
            double target_value = i % 2; // target value is binary
            var sampler = target_value == 0 ? MultiDimObject.InitValues.HE: MultiDimObject.InitValues.ONES;
            var image = new Tensor(28, 28, 3, sampler); // image 28x28x3
            var target = new Matrix(new double[][] {{target_value}}).transpose();
            image.set_requires_grad(false); // we don't need to propagate gradients for the inputs

            data.add(image);
            targets.add(target);
        }

        double learning_rate = 0.01;
        double momentum_rate = 0.9;
        Optimizer optimizer = new Momentum(model.get_parameters(), learning_rate, momentum_rate, mode);
        Loss loss = new BCELoss();  // binary cross entropy is an obvious choice for the binary classification problem

        int epochs = 100;
        for (int j = 0; j < epochs; ++j) {
            for (int i = 0; i < data.size(); ++i) {
                var output  = model.forward(data.get(i));

                var current_loss  = loss.calculate_loss(output, targets.get(i));
                loss.reset(); // we don't want to accumulate the loss values
                loss.add(current_loss);
                loss.backward(); // compute the gradients of the model with respect to the loss

                optimizer.step(); // update the weights
                optimizer.set_zero_gradients(); // set the gradients to zero
            }

            // every 10 epochs we check the predictions
            if (j % 10 == 0) {
                for (int k = 0; k < data.size(); ++k) {
                        var output  = model.forward(data.get(k));
                        System.out.printf("Predict: %f. True: %f\n", ((Matrix) output).get(0, 0).value, ((Matrix) targets.get(k)).get(0, 0).value);
                }
                System.out.println();
            }
        }
    }
}
