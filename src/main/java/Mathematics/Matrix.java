package main.java.mathematics;

import main.java.autograd.Value;
import java.util.Random;

public class Matrix {
    private Value[][] values_;
    private int[] size_;

//    public Matrix(Value[] values) { values_ = values; }

    public Matrix(int height, int width) {
        if (height < 1 || width < 1) throw new RuntimeException("Matrix has non-positive dimensions");
        size_ = new int[] { height, width };
        values_ = new Value[height][width];

        var random = new Random();
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j)
                values_[i][j] = new Value(random.nextGaussian(0, 1));
        }
    }

    public Matrix(double[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create matrix from the empty arrays");
        size_ = new int[] { matrix.length, matrix[0].length };
        values_ = new Value[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j)
                values_[i][j] = new Value(matrix[i][j]);
        }
    }

    public Matrix(Value[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create matrix from the empty arrays");
        size_ = new int[] { matrix.length, matrix[0].length };
        values_ = new Value[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j)
                values_[i][j] = matrix[i][j];
        }
    }

    public void set(int[] index, Value value) {
        if (index.length != 2 || !index_is_valid(index[0], size_[0]) || !index_is_valid(index[1], size_[1]))
                throw new ArrayIndexOutOfBoundsException("Invalid index to set");
        if (value == null)
                throw new NullPointerException("Attempt to set null value");
        values_[index[0]][index[1]] = value;
    }

    public Value get(int i, int j) {
        if (!index_is_valid(i, size_[0]) && index_is_valid(j, size_[1]))
            throw new ArrayIndexOutOfBoundsException("Attempt to get matrix value that is out of bounds");
        return values_[i][j];
    }
    public int[] get_size() { return size_; }

    private boolean index_is_valid(int index, int comparison) {
        return index >= 0 && index < comparison;
    }

    public Matrix multiply(Matrix other) {
        if (other == null) throw new NullPointerException("Attempt to multiply by the null matrix");
        if (size_[1] != other.size_[0]) throw new NullPointerException("Matrices have incompatible sizes to multiply");

        var matrix_array = new Value[size_[0]][other.size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < other.size_[1]; ++j) {
                matrix_array[i][j] = new Value(0);
                for (int k = 0; k < size_[1]; ++k)
                    matrix_array[i][j] = matrix_array[i][j].add(values_[i][k].multiply(other.values_[k][j]));
            }
        }

        return new Matrix(matrix_array);
    }

    private boolean has_same_size(Matrix other) {
        if (other == null) throw new NullPointerException("Comparison with the null matrix");
        int[] matrix_size = other.get_size();
        return matrix_size[0] == size_[0] && matrix_size[1] == size_[1];
    }

    public Matrix add(Matrix other) {
        if (!has_same_size(other)) throw new RuntimeException("Matrix has invalid size for the addition");

        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = values_[i][j].add(other.values_[i][j]);
            }
        }
        return new Matrix(matrix_array);
    }

    public Matrix relu() {
        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = values_[i][j].relu();
            }
        }
        return new Matrix(matrix_array);
    }

    public Matrix sigmoid() {
        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = values_[i][j].sigmoid();
            }
        }
        return new Matrix(matrix_array);
    }

    public boolean is_vector() { return size_[1] == 1; }

    public Matrix transpose() {
        var matrix_array = new Value[size_[1]][size_[0]];
        for (int i = 0; i < size_[1]; ++i) {
            for (int j = 0; j < size_[0]; ++j) {
                matrix_array[i][j] = values_[j][i];
            }
        }
        return new Matrix(matrix_array);
    }

    public void print() {
        for (int i = 0; i < size_[0]; ++i) {
            System.out.println();
            for (int j = 0; j < size_[1]; ++j) {
                System.out.printf("%f ", values_[i][j].get_value());
            }
        }
    }
}

