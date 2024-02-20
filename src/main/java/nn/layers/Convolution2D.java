package main.java.nn.layers;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Tensor;
import main.java.mathematics.Matrix;
import main.java.nn.models.IModel;

public class Convolution2D implements ILayer {
    private int stride_;
    private int padding_;
    private String padding_mode_;
    private Tensor[] kernels;
    private Matrix bias_;

    public Convolution2D(int in_channels, int out_channels, int kernel_size, int stride,
                         int padding, boolean bias, String padding_mode) {

        if (stride < 1) throw new RuntimeException("Stride must be at least 1");
        stride_= stride;
        padding_ = padding;
        bias_ = bias ? new Matrix(in_channels, 1, IMultiDimObject.InitValues.ZEROS) : null;
        padding_mode_ = padding_mode;

        kernels = new Tensor[out_channels];
        for (int i = 0; i < out_channels; ++i)
            kernels[i] = new Tensor(kernel_size, kernel_size, in_channels, Tensor.InitValues.RANDOM);
    }

    public Tensor forward(IMultiDimObject tensor) {
        Matrix[] matrices = new Matrix[kernels.length];
        Tensor casted_tensor = (Tensor) tensor;
        for (int i = 0; i < kernels.length; ++i)
            matrices[i] = LayerFunctions.convolve2D(casted_tensor, kernels[i], padding_, stride_);

        var result = new Tensor(matrices);
        return bias_ == null ? result : result.add_vector(bias_);
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
