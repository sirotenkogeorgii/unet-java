package main.java.mathematics;

import main.java.autograd.Value;
import java.util.Random;

public class Tensor {
    private Value[][][] values_;
    private int[] size_; // [height, width, depth]
    public Tensor(int height, int width, int depth) {
        if (height < 1 || width < 1 || depth < 1)
            throw new RuntimeException("Tensor has non-positive dimensions");

        size_ = new int[] { height, width, depth };
        var random = new Random();
        values_ = new Value[height][width][depth];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < depth; ++k)
                    values_[i][j][k] = new Value(random.nextGaussian(0, 1));
            }
        }
    }

    public Tensor(Matrix[] matrices) {
        if (matrices.length == 0) throw new RuntimeException("Attempt to create tensor from zero matrices");

        int[] matrix_size = matrices[0].get_size();
        size_ = new int[] { matrix_size[0], matrix_size[1], matrices.length };
        values_= new Value[matrix_size[0]][matrix_size[1]][matrices.length];
        for (int k = 0; k < matrices.length; ++k) {
            for (int i = 0; i < matrix_size[0]; ++i) {
                for (int j = 0; j < matrix_size[1]; ++j)
                    values_[i][j][k] = matrices[k].get(i, j);
            }
        }
    }

    public int[] get_size() { return size_; }
    private boolean index_is_valid(int index, int comparison) {
        return index >= 0 && index < comparison;
    }
    public Tensor slice(int[] x, int[] y) {
        System.out.printf("[DEBUG] x = [%d, %d], y = [%d, %d], size_ = [%d, %d, %d]\n", x[0], x[1], y[0], y[1], size_[0], size_[1], size_[2]);
        if (x.length != 2 || y.length != 2) throw new ArrayIndexOutOfBoundsException("Slicing size is invalid");
        if (x[0] > x[1] || y[0] > y[1]) throw new ArrayIndexOutOfBoundsException("Start is larger than end");
        if (!index_is_valid(x[0], size_[0]) || !index_is_valid(x[1] - 1, size_[0]) ||
                !index_is_valid(y[0], size_[1]) || !index_is_valid(y[1] - 1, size_[1]))
            throw new ArrayIndexOutOfBoundsException("Slicing bounds are invalid");

        int new_height = x[1] - x[0];
        int new_width = y[1] - y[0];
        var view_tensor = new Tensor(new_height, new_width, size_[2]);
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    view_tensor.values_[i][j][k] = values_[i][j][k];
            }
        }
        return view_tensor;
    }

    private boolean has_same_size(Tensor other) {
        if (other == null) throw new NullPointerException("Comparison with the null tensor");
        int[] tenor_size = other.get_size();
        return tenor_size[0] == size_[0] && tenor_size[1] == size_[1] && tenor_size[2] == size_[2];
    }

    public Tensor pw_multiply(Tensor other) {
        if (!has_same_size(other))
            throw new RuntimeException("Tensor has invalid size for the pairwise mul");
        var output_tensor = new Tensor(size_[0], size_[1], size_[2]);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    output_tensor.values_[i][j][k] = other.values_[i][j][k].multiply(values_[i][j][k]);
            }
        }
        return output_tensor;
    }

    public Value sum() {
        var value = new Value(0);

        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    value = value.add(values_[i][j][k]);
            }
        }
        return value;
    }

    public void backward() {
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    values_[i][j][k].backward();
            }
        }
    }

    public void print() {
        System.out.printf("size_ = [%d, %d, %d]\n", size_[0], size_[1], size_[2]);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    System.out.println(values_[i][j][k].get_gradient());
            }
        }
    }
}
