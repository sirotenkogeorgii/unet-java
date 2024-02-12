package main.java.nn.layers;

import main.java.autograd.Value;
import main.java.mathematics.Tensor;
import main.java.mathematics.Matrix;

public class Convolution2D {
    private int stride_;
    private int padding_;
    private boolean bias_;
    private String padding_mode_;
    private Tensor[] kernels;

    public Convolution2D(int in_channels, int out_channels, int kernel_size, int stride,
                         int padding, boolean bias, String padding_mode) {

        if (stride < 1) throw new RuntimeException("Stride must be at least 1");
        stride_= stride;
        padding_ = padding;
        bias_ = bias;
        padding_mode_ = padding_mode;

        kernels = new Tensor[out_channels];
        for (int i = 0; i < out_channels; ++i)
            kernels[i] = new Tensor(kernel_size, kernel_size, in_channels);
    }

    public Tensor convolve(Tensor tensor) {
        Matrix[] matrices = new Matrix[kernels.length];
        for (int i = 0; i < kernels.length; ++i)
            matrices[i] = convolve(tensor, kernels[i]);

        return new Tensor(matrices);
    }
    private Matrix convolve(Tensor tensor, Tensor kernel) {
        if (tensor == null)
            throw new NullPointerException("Attempt to convolve null tensor");

        int[] tensor_size = tensor.get_size();
        int[] kernel_size = kernel.get_size();

        if (!check_sizes(tensor.get_size(), kernel))
            throw new ArrayIndexOutOfBoundsException("Input tensor has incorrect size");

        int output_height = (tensor_size[0] + 2 * padding_ - kernel_size[0]) / stride_ + 1;
        int output_width = (tensor_size[1] + 2 * padding_ - kernel_size[1]) / stride_ + 1;
        var output_matrix = new Matrix(output_height, output_width);

        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                Tensor sliced_tensor = tensor.slice(
                        new int[] {i * stride_, kernel_size[0] + i * stride_},
                        new int[] {j * stride_, kernel_size[1] + j * stride_}
                );
                output_matrix.set(new int[] {i, j}, kernel.pw_multiply(sliced_tensor).sum());
            }
        }

        return output_matrix;
    }

    public void backward() {

    }

    private boolean check_sizes(int[] tensor_size, Tensor kernel) {
        int[] kernel_size = kernel.get_size();
        return tensor_size[2] == kernel_size[2] &&
                tensor_size[0] >= kernel_size[0] &&
                tensor_size[1] >= kernel_size[1];
    }
}

//class Program {
//    public static void main(String[] args) {
//        var input_tensor = new Tensor(
//                new Matrix[]{
//                        new Matrix(new double[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}),
//                        new Matrix(new double[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}),
//                        new Matrix(new double[][]{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}})
//                }
//        );
//
//        var conv1 = new Convolution2D(3, 9, 2, 1,  0, false, "zeros");
//        var conv2 = new Convolution2D(9, 3, 2, 1,  0, false, "zeros");
//        var conv3 = new Convolution2D(3, 1, 1, 1,  0, false, "zeros");
//
//        var r1 = conv1.convolve(input_tensor);
//        int[] output_tensor_size = r1.get_size();
//        System.out.printf("[DEBUG] out tensor has size: [%d, %d, %d]\n", output_tensor_size[0], output_tensor_size[1], output_tensor_size[2]);
//
//        var r2 = conv2.convolve(r1);
//        output_tensor_size = r2.get_size();
//        System.out.printf("[DEBUG] out tensor has size: [%d, %d, %d]\n", output_tensor_size[0], output_tensor_size[1], output_tensor_size[2]);
//
//        var r3 = conv3.convolve(r2);
//        output_tensor_size = r3.get_size();
//        System.out.printf("[DEBUG] out tensor has size: [%d, %d, %d]\n", output_tensor_size[0], output_tensor_size[1], output_tensor_size[2]);
//
//        r3.backward();
//
////        r3.print();
////        System.out.println();
////        r2.print();
////        System.out.println();
////        r1.print();
////        System.out.println();
//    }
//}
