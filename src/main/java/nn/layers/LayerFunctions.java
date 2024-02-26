package main.java.nn.layers;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.mathematics.Tensor;
import main.java.nn.losses.BCELoss;
import main.java.nn.losses.Loss;
import main.java.nn.models.Model;
import main.java.optimizers.Optimizer;
import main.java.optimizers.SGD;

import java.util.ArrayList;

public class LayerFunctions {
    public static Matrix convolveTransposed2D(Tensor tensor, Tensor kernel, int stride, int padding) {
        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
        if (kernel == null) throw new NullPointerException("Attempt to convolve with a null kernel");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size(); // we assume height = width

        int output_height = (tensor_size[0] - 1) * stride - 2 * padding + kernel_size[0];
        int output_width = (tensor_size[1] - 1) * stride - 2 * padding + kernel_size[1];

        Matrix output_matrix = LayerFunctions.padding2D(new Matrix(output_height, output_width, IMultiDimObject.InitValues.ZEROS), padding);

//        System.out.printf("Output matrix is [%d, %d]\n", output_height, output_width);

        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {

//                System.out.printf("Current outer index is [%d, %d]\n", i, j);

                Matrix current_input_vector = tensor.get_channel_vector(i, j);
                Matrix multiplied_kernel = kernel.multiply_vector(current_input_vector).sum_channels();

//                System.out.printf("Multiplied matrix is [%d, %d]\n", multiplied_kernel.get_size()[0], multiplied_kernel.get_size()[1]);

                for (int kh = 0, h = i * stride; h < kernel_size[0] + i * stride; ++kh, ++h) {
                    for (int kw = 0, w = j * stride; w < kernel_size[1] + j * stride; ++kw, ++w) {
//                        System.out.printf("Current inner index is [%d, %d]\n", h, w);
                        output_matrix.set(new int[]{h, w}, multiplied_kernel.get(kh, kw));
                    }
                }
            }
        }

        return output_matrix;
    }
    public static Matrix convolve2D(Tensor tensor, Tensor kernel, int stride, int padding) {
        // TODO: check arguments
        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
        if (!is_valid_kernel(tensor, kernel)) throw new ArrayIndexOutOfBoundsException("Input tensor has incorrect size");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size();

        int output_height = (tensor_size[0] + 2 * padding - kernel_size[0]) / stride + 1;
        int output_width = (tensor_size[1] + 2 * padding - kernel_size[1]) / stride + 1;
        var output_matrix = new Matrix(output_height, output_width, IMultiDimObject.InitValues.ZEROS);

        var padded_tensor = padding2D(tensor, padding);
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                Tensor sliced_tensor = padded_tensor.slice(
                        new int[] {i * stride, kernel_size[0] + i * stride},
                        new int[] {j * stride, kernel_size[1] + j * stride}
                );
//                sliced_tensor.print();
                output_matrix.set(new int[] {i, j}, kernel.pw_multiply(sliced_tensor).sum());
            }
        }

        return output_matrix;
    }
    private static boolean is_valid_kernel(Tensor tensor, Tensor kernel) {
        if (tensor == null) throw new RuntimeException("Tensor is null");
        if (kernel == null) throw new RuntimeException("Tensor kernel is null");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size();

        return tensor_size[2] == kernel_size[2] && tensor_size[0] >= kernel_size[0] && tensor_size[1] >= kernel_size[1];
    }
    public static Tensor padding2D(Tensor tensor, int padding) {
        if (tensor == null) throw new RuntimeException("Input tensor is null");
        if (padding < 0) throw new RuntimeException("Padding must be at least 0");
        if (padding == 0) return tensor;

        int[] tensor_size = tensor.get_size();
        var padded_tensor = new Tensor(tensor_size[0] + 2 * padding, tensor_size[1] + 2 * padding, tensor_size[2], IMultiDimObject.InitValues.ZEROS);
        // new Value[tensor_size[0] + 2 * padding][tensor_size[1] + 2 * padding][tensor_size[2]];

        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {
                for (int k = 0; k < tensor_size[3]; ++k) {
                    padded_tensor.set(new int[] {i + padding, j + padding, k}, tensor.get(new int[] {i, j, k}));
                }
            }
        }

        return padded_tensor;
    }
    public static Matrix padding2D(Matrix matrix, int padding) {
        if (matrix == null) throw new RuntimeException("Input matrix is null");
        if (padding < 0) throw new RuntimeException("Padding must be at least 0");
        if (padding == 0) return matrix;

        int[] matrix_size = matrix.get_size();
        var padded_matrix = new Matrix(matrix_size[0] + 2 * padding, matrix_size[1] + 2 * padding, IMultiDimObject.InitValues.ZEROS);

        for (int i = 0; i < matrix_size[0]; ++i) {
            for (int j = 0; j < matrix_size[1]; ++j) {
                padded_matrix.set(new int[] {i + padding, j + padding}, matrix.get(i, j));
            }
        }

        return padded_matrix;
    }

    public static Matrix flatten(Tensor tensor) {
        if (tensor == null) throw new NullPointerException("Attempt to flat a null tensor");
        int[] tensor_size = tensor.get_size();

        int flatten_size = tensor_size[0] * tensor_size[1] * tensor_size[2];
        var values = new Matrix(flatten_size, 1, IMultiDimObject.InitValues.ZEROS);

        int current_index = 0;
        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {
                for (int k = 0; k < tensor_size[2]; ++k) {
                    values.set(new int[] {current_index++, 0}, tensor.get(new int[] {i, j, k}));
                }
            }
        }
        return values;
    }

    public static Tensor maxPool2D(Tensor tensor, int size) {
        if (tensor == null) throw new NullPointerException("Attempt to max pool a null tensor");

        int[] tensor_size = tensor.get_size();
        if (tensor_size[0] % size != 0 || tensor_size[1] % size != 0)
            throw new RuntimeException("Tensor size is invalid to be max pooled");

        int output_height = tensor_size[0] / size;
        int output_width = tensor_size[1] / size;
        var output_tensor = new Tensor(output_height, output_width, tensor_size[2], IMultiDimObject.InitValues.ZEROS);

        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                Tensor sliced_tensor = tensor.slice(
                        new int[] {i * size, size + i * size},
                        new int[] {j * size, size + j * size}
                );
                Value[] max_values = LayerFunctions.maxTensor(sliced_tensor);
                for (int k = 0; k < tensor_size[2]; ++k)
                    output_tensor.set(new int[] {i, j, k}, max_values[k]);
            }
        }

        return output_tensor;
    }

    public static Value[] maxTensor(Tensor tensor) {
        if (tensor == null) throw new RuntimeException("Attempt to take a max of a null tensor");
        int[] tensor_size = tensor.get_size();
        Value[] max_values = new Value[tensor_size[2]];
        for (int channel_i = 0; channel_i < tensor_size[2]; ++channel_i)
            max_values[channel_i] = maxMatrix(tensor.get_dim(channel_i));
        return max_values;
    }

    public static Value maxMatrix(Matrix matrix) {
        if (matrix == null) throw new RuntimeException("Attempt to take a max of a null matrix");
        Value current_value = matrix.get(0, 0);
        int[] matrix_size = matrix.get_size();
        for (int i = 0; i < matrix_size[0]; ++i) {
            for (int j = 0; j < matrix_size[1]; ++j) {
                if (current_value.get_value() < matrix.get(i, j).get_value())
                    current_value = matrix.get(i, j);
            }
        }
        return current_value;
    }

    public static Tensor concatenate(ArrayList<Tensor> tensors) {
        if (tensors == null || tensors.isEmpty()) throw new RuntimeException("Tensor to concatenate is empty or null");
        Tensor current_tensor = tensors.get(0);
        for (int i = 1; i < tensors.size(); ++i)
            current_tensor = current_tensor.concatenate(tensors.get(i));
        return current_tensor;
    }

    public static Value bce_loss(Matrix matrix1, Matrix matrix2) {
        Value result = new Value(0);
        int[] matrix_shape = matrix1.get_size();
        for (int i = 0; i < matrix_shape[0]; ++i) {
            for (int j = 0; j < matrix_shape[1]; ++j) {
                Value temp = bce_value(matrix1.get(i, j), matrix2.get(i, j));
                result = result.add(temp);
            }
        }
        return  result;
    }

    private static Value bce_value(Value pred, Value target) {
        return target.multiply(-1).multiply(pred.log()).sub(         target.multiply(-1).add(1).multiply(        pred.multiply(-1).add(1).log()           )                 );
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

    static Value bce_value(Value pred, Value target) {
        return target.multiply(-1).multiply(pred.log()).sub(         target.multiply(-1).add(1).multiply(        pred.multiply(-1).add(1).log()           )                 );
    }

    static Value bce_loss(Matrix matrix1, Matrix matrix2) {
        Value result = new Value(0);
        int[] matrix_shape = matrix1.get_size();
        for (int i = 0; i < matrix_shape[0]; ++i) {
            for (int j = 0; j < matrix_shape[1]; ++j) {
                Value temp = bce_value(matrix1.get(i, j), matrix2.get(i, j));
                result = result.add(temp);
            }
        }
        return  result;
    }

    public static void main(String[] args) {
        ArrayList<ILayer> layers = new ArrayList<>();

        layers.add(new Convolution2D(3, 1, 3, 1, 0, true, null, Convolution.Activation.ReLU));
        layers.add(new Convolution2D(1, 1, 3, 1, 0, true, null, Convolution.Activation.ReLU));
        layers.add(new MaxPool2D(2));
        layers.add(new Flatten2D());
        layers.add(new LinearLayer(12*12*1, 10, true, "identity"));
        layers.add(new LinearLayer(10, 1, true, "sigmoid"));


        var input_image = new Tensor(28, 28, 3, IMultiDimObject.InitValues.RANDOM);
        var target = new Matrix(new double[][] {{1}}).transpose();

        var model = new Model(layers);

        Optimizer optimizer = new SGD(model.get_parameters(), 0.01);
        Loss loss = new BCELoss();

        for (int i = 0; i < 100; ++i) {
            var output = model.forward(input_image);

            loss.calculate_loss(output, target);
            loss.backward();

            optimizer.step();
            optimizer.set_zero_gradients();

            System.out.printf("Loss: %f\n", loss.get_loss().get_value());
//            ((Matrix)output).transpose().print();
        }
    }
}