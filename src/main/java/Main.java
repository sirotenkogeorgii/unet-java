package main.java;

import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.layers.*;
import main.java.nn.losses.BCELoss;
import main.java.nn.losses.Loss;
import main.java.nn.models.Model;
import main.java.nn.models.ModelSettings;
import main.java.nn.models.SequentialModel;
import main.java.optimizers.Optimizer;
import main.java.optimizers.SGD;

import java.util.ArrayList;
import java.util.Spliterator;

public class Main {

    public static void main(String[] args) {
//        ModelSettings.executionMode mode = ModelSettings.executionMode.PARALLEL;
        ModelSettings.executionMode mode = ModelSettings.executionMode.SERIAL;
//
        ArrayList<ILayer> layers = new ArrayList<>();

//        layers.add(new Convolution2D(3, 1, 3, 1, 0, true, null, Convolution.Activation.ReLU, mode));
//        layers.add(new Convolution2D(1, 1, 3, 1, 0, true, null, Convolution.Activation.ReLU, mode));
//        layers.add(new MaxPool2D(2));
//        layers.add(new Flatten2D());
//        layers.add(new LinearLayer(12 * 12 * 1, 10, true, "relu"));
//        layers.add(new LinearLayer(28 * 28 * 3, 50, true, "identity"));
//        layers.add(new LinearLayer(50, 10, true, "identity"));
//        layers.add(new LinearLayer(10, 1, true, "identity"));

        var parameters = new ArrayList<MultiDimObject>();
        var flatten_layer = new Flatten2D();
        var fc1 = new LinearLayer(3 * 3 * 3, 10, true, "relu");
        var fc2 = new LinearLayer(10, 10, true, "relu");
        var output_layer = new LinearLayer(10, 1, true, "sigmoid");
        parameters.addAll(flatten_layer.get_parameters());
        parameters.addAll(fc1.get_parameters());
        parameters.addAll(fc2.get_parameters());
        parameters.addAll(output_layer.get_parameters());

//        var model = new SequentialModel(layers, mode);

        ArrayList<MultiDimObject> data = new ArrayList<>();
        ArrayList<MultiDimObject> targets = new ArrayList<>();

        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{0}}).transpose());
        data.add(new Tensor(3, 3, 3, MultiDimObject.InitValues.HE));
        targets.add(new Matrix(new double[][] {{1}}).transpose());

        Optimizer optimizer = new SGD(parameters, 1, mode);
        Loss loss = new BCELoss();

        long startTime = System.nanoTime();
        for (int j = 0; j < 1; ++j) {
            for (int i = 0; i < data.size(); ++i) {
                System.out.println("fc1 weights:");
                fc1.get_parameters().get(0).print();
                System.out.println("fc2 weights:");
                fc2.get_parameters().get(0).print();
                System.out.println("output layer weights:");
                output_layer.get_parameters().get(0).print();

                var o1 = flatten_layer.forward(data.get(i));
                var o2 = fc1.forward(o1);
                var o3 = fc2.forward(o2);
                var output = output_layer.forward(o3);
                output.print();

                loss.calculate_loss(output, targets.get(i));
                loss.backward();

                optimizer.step();
                optimizer.set_zero_gradients();

                System.out.printf("Loss: %f. Target: %f\n", loss.get_loss().value, ((Matrix) targets.get(i)).get(0, 0).value);
//                System.out.printf("Predict: %f. True: %f\n", ((Matrix) output).get(0, 0).value, ((Matrix) targets.get(i)).get(0, 0).value);

                for (int k = 0; k < data.size(); ++k) {
                    o1 = flatten_layer.forward(data.get(i));
                    o2 = fc1.forward(o1);
                    o3 = fc2.forward(o2);
                    output = output_layer.forward(o3);
                    System.out.printf("Predict: %f. True: %f\n", ((Matrix) output).get(0, 0).value, ((Matrix) targets.get(k)).get(0, 0).value);
//                }
                }
            }

            long endTime = System.nanoTime();
            long executionTime = endTime - startTime;
            System.out.println("Execution time in milliseconds: " + executionTime / 1_000_000);

            // Serial: 6091
            // Parallel: 3665

        }
    }
}


















//fc1 weights:
//0,426846 0,617812 -0,309809 0,155588 -0,190879 -0,235563 -0,594186 -0,844664
//fc2 weights:
//        0,408558 0,033439 -0,745401 -0,223262 -0,010225 -0,065636 0,084750 0,885020 1,013647 -0,183578
//        -0,329990 -0,120139 -0,483478 0,100550 -0,059146 0,173878 1,077098 -0,640287 0,324492 0,634194
//        0,201563 0,081192 -0,372524 0,988278 0,721148 0,254484 0,263936 -0,740939 -0,234788 0,628343
//        0,390737 -0,577315 0,354070 0,507422 0,328321 -0,416216 -0,770995 -0,138623 0,020665 -0,200413
//        0,654762 -0,125233 0,718571 -0,634499 -0,242773 -0,230210 0,779309 0,497666 -0,205921 0,544460
//        0,085707 -0,003617 0,334382 0,387991 0,224764 -0,270456 0,311618 -0,625818 -0,327996 0,319747
//        0,534482 -0,624946 0,405821 0,071932 0,063628 1,346364 -0,532955 0,033911 -0,677612 -0,159769
//        0,156387 -0,428534 0,234463 -0,048713 0,081884 0,352695 0,116759 0,090998 0,069190 0,612800
//        -0,030473 1,023508 -0,416743 0,170116 0,689898 -0,014374 0,572994 0,553082 -0,353935 0,867081
//        0,265158 -0,039194 -0,355099 -0,637246 -0,249695 -0,101573 0,250322 -0,307262 -0,250296 0,587505
//output layer weights:
//0,061719 0,024538 0,606539 -0,656058 0,851914 -0,625288 0,577173 0,212202 0,379121 -0,096785
//








