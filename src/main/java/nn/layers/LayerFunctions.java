package main.java.nn.layers;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.mathematics.Tensor;

import java.util.ArrayList;

public class LayerFunctions {
    public static Matrix convolveTransposed2D(Tensor tensor, Tensor kernel, int stride, int padding) {
        if (tensor == null) throw new NullPointerException("Attempt to convolve null tensor");
        if (kernel == null) throw new NullPointerException("Attempt to convolve with a null kernel");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size(); // we assume height = width

        int output_height = (tensor_size[0] - 1) * stride - 2 * padding + kernel_size[0];
        int output_width = (tensor_size[1] - 1) * stride - 2 * padding + kernel_size[1];

        Matrix output_matrix = new Matrix(output_height, output_width, IMultiDimObject.InitValues.ZEROS);

        System.out.printf("Output matrix is [%d, %d]\n", output_height, output_width);

        for (int i = 0; i < tensor_size[0]; ++i) {
            for (int j = 0; j < tensor_size[1]; ++j) {

                System.out.printf("Current outer index is [%d, %d]\n", i, j);

                Matrix current_input_vector = tensor.get_channel_vector(i, j);
                Matrix multiplied_kernel = kernel.multiply_vector(current_input_vector).sum_channels();

                System.out.printf("Multiplied matrix is [%d, %d]\n", multiplied_kernel.get_size()[0], multiplied_kernel.get_size()[1]);

                for (int kh = 0, h = i * stride; h < kernel_size[0] + i * stride; ++kh, ++h) {
                    for (int kw = 0, w = j * stride; w < kernel_size[1] + j * stride; ++kw, ++w) {
                        System.out.printf("Current inner index is [%d, %d]\n", h, w);
                        output_matrix.set(new int[]{h, w}, multiplied_kernel.get(kh, kw));
                    }
                }
            }
        }

        return output_matrix;
    }
    public static Matrix convolve2D(Tensor tensor, Tensor kernel, int padding, int stride) {
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
    public static Tensor concatenate(ArrayList<Tensor> tensor) {
        return null;
    }

//    public static Matrix padding2D(Matrix matrix, int padding) {
//        if (matrix == null) throw new RuntimeException("Input matrix is null");
//        if (padding < 1) throw new RuntimeException("Padding must be at least 1");
//
//        int[] matrix_size = matrix.get_size();
//        var padded_matrix = new Matrix(matrix_size[0] + 2 * padding, matrix_size[1] + 2 * padding, IMultiDimObject.InitValues.ZEROS);
//        // new Value[tensor_size[0] + 2 * padding][tensor_size[1] + 2 * padding][tensor_size[2]];
//
//        for (int i = 0; i < tensor_size[0]; ++i) {
//            for (int j = 0; j < tensor_size[1]; ++j) {
//                for (int k = 0; k < tensor_size[3]; ++k) {
//                    padded_tensor.set(new int[] {i + padding, j + padding, k}, tensor.get(new int[] {i, j, k}));
//                }
//            }
//        }
//
//        return padded_tensor;
//    }
}

class Program {
    public static void main(String[] args) {
        Tensor input_tensor = new Tensor(2, 2, 3, IMultiDimObject.InitValues.RANDOM);
        Tensor kernel = new Tensor(3, 3, 3, IMultiDimObject.InitValues.RANDOM);

        Matrix output_matrix = LayerFunctions.convolveTransposed2D(input_tensor, kernel, 2, 1);

        output_matrix.print();
    }
}x