package main.java.optimizers;

import main.java.autograd.Value;
import main.java.mathematics.MultiDimObject;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.stream.StreamSupport;

public class SGD extends Optimizer {

    public SGD(ArrayList<MultiDimObject> parameters, double alpha, ModelSettings.executionMode mode) {
        parameters_ = parameters;
        alpha_ = alpha;
        mode_ = mode;
    }
    @Override
    public void step() {
        long startTime = System.nanoTime();

        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            parameters_.parallelStream() // Process each MultiDimObject in parallel
//                    .flatMap(param -> StreamSupport.stream(param.spliterator(), true)) // Flatten to stream of Values
                    .flatMap(param -> StreamSupport.stream(param.spliterator(), false)) // Flatten to stream of Values
                    .forEach(val -> val.value = val.value - alpha_ * val.gradient);
        } else {
            for (MultiDimObject param: parameters_) {
                for (Value val: param) {
//                System.out.printf("Value %f Gradient %f\n", val.get_value(), val.get_gradient(), names.length);
                    val.value = val.value - alpha_ * val.gradient;
                }
            }
        }

        long endTime = System.nanoTime();
        long executionTime = endTime - startTime;
        System.out.println("Execution time in sgd_step in milliseconds: " + executionTime / 1_000_000);

    }

    @Override
    public void set_zero_gradients() {
        if (mode_ == ModelSettings.executionMode.PARALLEL) {
            parameters_.parallelStream() // Process each MultiDimObject in parallel
                    .flatMap(param -> StreamSupport.stream(param.spliterator(), false)) // Flatten to stream of Values
//                    .flatMap(param -> StreamSupport.stream(param.spliterator(), true)) // Flatten to stream of Values
                    .forEach(val -> {
                        // Set each value in parallel
                        val.gradient = 0;
                    });
        } else {
            for (MultiDimObject param: parameters_) {
                for (Value val: param) {
                    val.gradient = 0;
                }
            }
        }
    }
}
