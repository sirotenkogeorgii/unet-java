package nn.layers;

import autograd.Value;
import mathematics.Matrix;
import mathematics.MultiDimObject;
import mathematics.Tensor;

import java.util.ArrayList;
import java.util.stream.IntStream;

/**
 * Provides utility functions for common neural network layer operations such as convolution,
 * pooling, and flattening. These functions are typically used within custom neural network layers.
 */
public class LayerFunctions {

    /**
     * Applies a 2D convolution operation to a given tensor using a specified kernel.
     *
     * @param tensor The input tensor to be convolved.
     * @param kernel The convolution kernel (filter).
     * @param stride The stride of the convolution.
     * @param padding The padding size applied to the tensor.
     * @return A new Matrix representing the result of the convolution.
     * @throws NullPointerException if the input tensor is null.
     * @throws ArrayIndexOutOfBoundsException if the input tensor's dimensions do not match the kernel's requirements.
     */
    public static Matrix convolve2D(Tensor tensor, Tensor kernel, int stride, int padding) {
        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
        if (!is_valid_kernel(tensor, kernel)) throw new ArrayIndexOutOfBoundsException("Input tensor has incorrect size");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size();

        int output_height = (tensor_size[0] + 2 * padding - kernel_size[0]) / stride + 1;
        int output_width = (tensor_size[1] + 2 * padding - kernel_size[1]) / stride + 1;
        Value[][] output_matrix_array = new Value[output_height][output_width];

        Tensor padded_tensor = padding2D(tensor, padding);

        IntStream.range(0, output_height).parallel().forEach(i -> {
            IntStream.range(0, output_width).forEach(j -> {

                Tensor sliced_tensor = padded_tensor.slice(
                        new int[]{i * stride, i * stride + kernel_size[0]},
                        new int[]{j * stride, j * stride + kernel_size[1]}
                );

                output_matrix_array[i][j] = kernel.pw_multiply(sliced_tensor).sum();
            });
        });

        return new Matrix(output_matrix_array);
    }

    /**
     * Verifies if the kernel dimensions are suitable for convolving with the given tensor.
     *
     * @param tensor The tensor to be convolved.
     * @param kernel The convolution kernel.
     * @return true if the kernel can be applied to the tensor; false otherwise.
     * @throws RuntimeException if either the tensor or kernel is null.
     */
    private static boolean is_valid_kernel(Tensor tensor, Tensor kernel) {
        if (tensor == null) throw new RuntimeException("Tensor is null");
        if (kernel == null) throw new RuntimeException("Tensor kernel is null");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size();

        return tensor_size[2] == kernel_size[2] && tensor_size[0] >= kernel_size[0] && tensor_size[1] >= kernel_size[1];
    }

    /**
     * Applies padding to a 2D tensor.
     *
     * @param tensor The tensor to pad.
     * @param padding The amount of padding to apply to each side.
     * @return A new Tensor with padding applied.
     * @throws RuntimeException if the tensor is null or the padding is negative.
     */
    public static Tensor padding2D(Tensor tensor, int padding) {
        if (tensor == null) throw new RuntimeException("Input tensor is null");
        if (padding < 0) throw new RuntimeException("Padding must be at least 0");
        if (padding == 0) return tensor;

        int[] tensor_size = tensor.get_size();
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

    /**
     * Applies padding to a Matrix.
     *
     * @param matrix The matrix to pad.
     * @param padding The amount of padding to apply to each side.
     * @return A new Matrix with padding applied.
     * @throws RuntimeException if the matrix is null or the padding is negative.
     */
    public static Matrix padding2D(Matrix matrix, int padding) {
        if (matrix == null) throw new RuntimeException("Input matrix is null");
        if (padding < 0) throw new RuntimeException("Padding must be at least 0");
        if (padding == 0) return matrix;

        int[] matrix_size = matrix.get_size();
        var padded_matrix_array = new Value[matrix_size[0] + 2 * padding][matrix_size[1] + 2 * padding];

        for (int i = 0; i < matrix_size[0]; ++i) {
            for (int j = 0; j < matrix_size[1]; ++j) {
                padded_matrix_array[i + padding][j + padding] = matrix.get(i, j);
            }
        }

        return new Matrix(padded_matrix_array);
    }

    /**
     * Flattens a tensor into a single-column matrix.
     *
     * @param tensor The tensor to flatten.
     * @return A Matrix where each element of the tensor is laid out in a single column.
     * @throws NullPointerException if the tensor is null.
     */
    public static Matrix flatten(Tensor tensor) {
        if (tensor == null) throw new NullPointerException("Attempt to flat a null tensor");
        int[] tensor_size = tensor.get_size();

        int flatten_size = tensor_size[0] * tensor_size[1] * tensor_size[2];
        var values_array = new Value[flatten_size][1];

        int current_index = 0;
        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {
                for (int k = 0; k < tensor_size[2]; ++k) {
                    values_array[current_index++][0] = tensor.get(i, j, k);
                }
            }
        }
        return new Matrix(values_array);
    }

    /**
     * Applies a max pooling operation to a tensor.
     *
     * @param tensor The tensor to apply max pooling to.
     * @param size The size of the window to use for max pooling.
     * @return A new Tensor representing the max pooled output.
     * @throws NullPointerException if the tensor is null.
     * @throws RuntimeException if the tensor's dimensions are not divisible by the pooling size.
     */
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

    /**
     * Extracts the maximum value from each channel of a tensor.
     *
     * @param tensor The tensor from which to extract maximum values.
     * @return An array of Values, each representing the maximum value from a tensor channel.
     * @throws RuntimeException if the tensor is null.
     */
    public static Value[] maxTensor(Tensor tensor) {
        if (tensor == null) throw new RuntimeException("Attempt to take a max of a null tensor");
        int[] tensor_size = tensor.get_size();
        Value[] max_values = new Value[tensor_size[2]];
        for (int channel_i = 0; channel_i < tensor_size[2]; ++channel_i)
            max_values[channel_i] = maxMatrix(tensor.get_dim(channel_i));
        return max_values;
    }

    /**
     * Finds the maximum value in a matrix.
     *
     * @param matrix The matrix to find the maximum value in.
     * @return The maximum value found in the matrix.
     * @throws RuntimeException if the matrix is null.
     */
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

    /**
     * Computes the binary cross-entropy loss between two matrices.
     *
     * @param matrix1 The first matrix (usually predictions).
     * @param matrix2 The second matrix (usually targets).
     * @return A Value representing the computed binary cross-entropy loss.
     */
    public static Value bce_loss(Matrix matrix1, Matrix matrix2) {
        int[] matrix_shape = matrix1.get_size();
        var values_array = new ArrayList<Value>();
        for (int i = 0; i < matrix_shape[0]; ++i) {
            for (int j = 0; j < matrix_shape[1]; ++j) {
                Value temp = bce_value(matrix1.get(i, j), matrix2.get(i, j));
                values_array.add(temp);
            }
        }
        return Value.add(values_array);
    }

    /**
     * Computes the binary cross-entropy loss value for individual prediction-target pairs.
     *
     * @param pred The predicted value.
     * @param target The target value.
     * @return The computed loss Value for the given prediction and target.
     */
    private static Value bce_value(Value pred, Value target) {
        var left_part = target.multiply(pred.log());
        var right_part = target.multiply(-1).add(1).multiply(pred.multiply(-1).add(1).log());
        return left_part.add(right_part).multiply(-1);
    }

    /**
     * Calculates the cross-entropy loss between two matrices.
     *
     * This method computes the cross-entropy loss, a common loss function for classification problems,
     * between two matrices representing predicted probabilities and true labels. The loss is calculated
     * as the negative sum of the element-wise product of the logarithm of elements from the first matrix
     * (typically prediction probabilities) and elements from the second matrix (typically true labels).
     *
     * Each element from the first matrix is taken as the predicted log-probability and is multiplied by
     * the corresponding element from the second matrix, which represents the true label. The cross-entropy
     * loss is useful for measuring the performance of a classification model where the output is a probability
     * value between 0 and 1.
     *
     * @param matrix1 the matrix representing predicted log-probabilities; should not contain zero values
     *                since the logarithm of zero is undefined.
     * @param matrix2 the matrix representing true labels, typically as one-hot encoded vectors.
     * @return a {@code Value} object representing the total cross-entropy loss, which is a single scalar value.
     * @throws IllegalArgumentException if the sizes of matrix1 and matrix2 do not match.
     */
    public static Value cross_entropy_loss(Matrix matrix1, Matrix matrix2) {
        int[] matrix_shape = matrix1.get_size();
        var values_array = new ArrayList<Value>();
        for (int i = 0; i < matrix_shape[0]; ++i) {
            for (int j = 0; j < matrix_shape[1]; ++j) {
                Value temp = matrix1.get(i, j).log().multiply(matrix2.get(i, j));
                values_array.add(temp);
            }
        }
        return Value.add(values_array).multiply(-1);
    }
}