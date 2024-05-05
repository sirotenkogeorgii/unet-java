package main.java.nn.layers;

import main.java.autograd.Value;
import main.java.mathematics.Matrix;
import main.java.mathematics.MultiDimObject;
import main.java.mathematics.Tensor;
import main.java.nn.losses.BCELoss;
import main.java.nn.losses.Loss;
import main.java.nn.models.ModelSettings;
import main.java.nn.models.SequentialModel;
import main.java.optimizers.Optimizer;
import main.java.optimizers.SGD;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.ArrayList;
import java.util.stream.IntStream;

public class LayerFunctions {
    public static Matrix convolveTransposed2D(Tensor tensor, Tensor kernel, int stride, int padding) {
        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
        if (kernel == null) throw new NullPointerException("Attempt to convolve with a null kernel");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size(); // we assume height = width

        int output_height = (tensor_size[0] - 1) * stride - 2 * padding + kernel_size[0];
        int output_width = (tensor_size[1] - 1) * stride - 2 * padding + kernel_size[1];

        Matrix output_matrix = LayerFunctions.padding2D(new Matrix(output_height, output_width, MultiDimObject.InitValues.ZEROS), padding);

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
                        output_matrix.set(multiplied_kernel.get(kh, kw), h, w);
                    }
                }
            }
        }

        return output_matrix;
    }



    public static Matrix convolve2D(Tensor tensor, Tensor kernel, int stride, int padding) {
//        // TODO: check arguments
//        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
//        if (!is_valid_kernel(tensor, kernel)) throw new ArrayIndexOutOfBoundsException("Input tensor has incorrect size");
//
//        int[] tensor_size = tensor.get_size();
//        int[] kernel_size = kernel.get_size();
//
//        int output_height = (tensor_size[0] + 2 * padding - kernel_size[0]) / stride + 1;
//        int output_width = (tensor_size[1] + 2 * padding - kernel_size[1]) / stride + 1;
//        var output_matrix_array = new Value[output_height][output_width];
//
//        var padded_tensor = padding2D(tensor, padding);
//        for (int i = 0; i < output_height; ++i) {
//            for (int j = 0; j < output_width; ++j) {
//
//                Tensor sliced_tensor = padded_tensor.slice(
//                        new int[] {i * stride, kernel_size[0] + i * stride},
//                        new int[] {j * stride, kernel_size[1] + j * stride}
//                );
//
//                output_matrix_array[i][j] = kernel.pw_multiply(sliced_tensor).sum();
//            }
//        }
//
//        return new Matrix(output_matrix_array);

        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
        if (!is_valid_kernel(tensor, kernel)) throw new ArrayIndexOutOfBoundsException("Input tensor has incorrect size");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size();

        int output_height = (tensor_size[0] + 2 * padding - kernel_size[0]) / stride + 1;
        int output_width = (tensor_size[1] + 2 * padding - kernel_size[1]) / stride + 1;
        Value[][] output_matrix_array = new Value[output_height][output_width];

        Tensor padded_tensor = padding2D(tensor, padding);

//        int num_threads = output_height * output_width;
//        ExecutorService executor = Executors.newFixedThreadPool(num_threads);
//
//        for (int i = 0; i < output_height; i++) {
//            for (int j = 0; j < output_width; j++) {
//                int finalI = i;
//                int finalJ = j;
//                int depth = kernel_size[2];
//                executor.submit(() -> {
//                    int startX = finalI * stride;
//                    int endX = finalI * stride + kernel_size[0];
//                    int startY = finalJ * stride;
//                    int endY = finalJ * stride + kernel_size[1];
//
//                    Value window_sum = new Value(0);
//                    for (int x = startX; x < endX; x++) {
//                        for (int y = startY; y < endY; y++) {
//                            for (int z = 0; z < depth; z++) {
//                                var result = tensor.get(x, y, z).multiply(kernel.get(x - startX, y - startY, z));
//                                window_sum = window_sum.add(result);
//                            }
//                        }
//                    }
//                    output_matrix_array[finalI][finalJ] = window_sum;
//                });
//            }
//        }
//
//        executor.shutdown();
//        while (!executor.isTerminated()) {
//            // Wait for all tasks to complete
//        }

        // Parallelize over the output matrix's dimensions
        IntStream.range(0, output_height).parallel().forEach(i -> {
            IntStream.range(0, output_width).forEach(j -> {

                Tensor sliced_tensor = padded_tensor.slice(
                        new int[]{i * stride, i * stride + kernel_size[0]},
                        new int[]{j * stride, j * stride + kernel_size[1]}
                );

                output_matrix_array[i][j] = kernel.pw_multiply(sliced_tensor).sum();

//                    int startX = i * stride;
//                    int endX = i * stride + kernel_size[0];
//                    int startY = j * stride;
//                    int endY = j * stride + kernel_size[1];
//
//                    Value window_sum = new Value(0);
//                    for (int x = startX; x < endX; x++) {
//                        for (int y = startY; y < endY; y++) {
//                            for (int z = 0; z < kernel_size[2]; z++) {
//                                var result = padded_tensor.get(x, y, z).multiply(kernel.get(x - startX, y - startY, z));
//                                window_sum = window_sum.add(result);
//                            }
//                        }
//                    }
//                    output_matrix_array[i][j] = window_sum;
            });
        });

        return new Matrix(output_matrix_array);
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
        var padded_tensor = new Tensor(tensor_size[0] + 2 * padding, tensor_size[1] + 2 * padding, tensor_size[2], MultiDimObject.InitValues.ZEROS);
        // new Value[tensor_size[0] + 2 * padding][tensor_size[1] + 2 * padding][tensor_size[2]];
        var padded_tensor_array = new Value[tensor_size[0] + 2 * padding][tensor_size[1] + 2 * padding][tensor_size[2]];

        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {
                for (int k = 0; k < tensor_size[3]; ++k) {
                    padded_tensor_array[i + padding][j + padding][k] = tensor.get(i, j, k);
                }
            }
        }

        return new Tensor(padded_tensor_array);
    }
    public static Matrix padding2D(Matrix matrix, int padding) {
        if (matrix == null) throw new RuntimeException("Input matrix is null");
        if (padding < 0) throw new RuntimeException("Padding must be at least 0");
        if (padding == 0) return matrix;

        int[] matrix_size = matrix.get_size();
//        var padded_matrix = new Matrix(matrix_size[0] + 2 * padding, matrix_size[1] + 2 * padding, IMultiDimObject.InitValues.ZEROS);
        var padded_matrix_array = new Value[matrix_size[0] + 2 * padding][matrix_size[1] + 2 * padding];

        for (int i = 0; i < matrix_size[0]; ++i) {
            for (int j = 0; j < matrix_size[1]; ++j) {
                padded_matrix_array[i + padding][j + padding] = matrix.get(i, j);
            }
        }

        return new Matrix(padded_matrix_array);
    }

    public static Matrix flatten(Tensor tensor) {
        if (tensor == null) throw new NullPointerException("Attempt to flat a null tensor");
        int[] tensor_size = tensor.get_size();

        int flatten_size = tensor_size[0] * tensor_size[1] * tensor_size[2];
//        var values = new Matrix(flatten_size, 1, IMultiDimObject.InitValues.ZEROS);
        var values_array = new Value[flatten_size][1];

        int current_index = 0;
        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {
                for (int k = 0; k < tensor_size[2]; ++k) {
                    values_array[current_index++][0] = tensor.get(new int[] {i, j, k});
                }
            }
        }
        return new Matrix(values_array);
    }

    public static Tensor maxPool2D(Tensor tensor, int size) {
        if (tensor == null) throw new NullPointerException("Attempt to max pool a null tensor");

        int[] tensor_size = tensor.get_size();
        if (tensor_size[0] % size != 0 || tensor_size[1] % size != 0)
            throw new RuntimeException("Tensor size is invalid to be max pooled");

        int output_height = tensor_size[0] / size;
        int output_width = tensor_size[1] / size;
        var output_tensor = new Tensor(output_height, output_width, tensor_size[2], MultiDimObject.InitValues.ZEROS);

        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                Tensor sliced_tensor = tensor.slice(
                        new int[] {i * size, size + i * size},
                        new int[] {j * size, size + j * size}
                );
                Value[] max_values = LayerFunctions.maxTensor(sliced_tensor);
                for (int k = 0; k < tensor_size[2]; ++k)
                    output_tensor.set(max_values[k], i, j, k);
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
                if (current_value.value < matrix.get(i, j).value)
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
//        Value result = new Value(0);
        int[] matrix_shape = matrix1.get_size();
        var values_array = new ArrayList<Value>();
        for (int i = 0; i < matrix_shape[0]; ++i) {
            for (int j = 0; j < matrix_shape[1]; ++j) {
                Value temp = bce_value(matrix1.get(i, j), matrix2.get(i, j));
//                result = result.add(temp);
                values_array.add(temp);
            }
        }
//        return  result;
        return Value.add(values_array);
    }

    private static Value bce_value(Value pred, Value target) {
        var left_part = target.multiply(pred.log());
        var right_part = target.multiply(-1).add(1).multiply(pred.multiply(-1).add(1).log());
        return left_part.add(right_part).multiply(-1);
    }
}