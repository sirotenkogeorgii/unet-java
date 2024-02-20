package main.java.mathematics;

import main.java.autograd.Value;
import java.util.Random;

public class Matrix implements IMultiDimObject {
    private Value[][] values_;
    private int[] size_;

    public Matrix(int height, int width, InitValues init_values) {
        if (height < 1 || width < 1) throw new RuntimeException("Matrix has non-positive dimensions");
        size_ = new int[] { height, width };
        values_ = new Value[height][width];

        var random = new Random();
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                double current_value = switch (init_values) { case ZEROS -> 0; case RANDOM -> random.nextGaussian(0, 1); default -> throw new RuntimeException("Unknown value to fill"); };
                values_[i][j] = new Value(current_value);
            }
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
        if (matrix == null) throw new RuntimeException("Array to create a matrix is null");
        if (matrix.length == 0 || matrix[0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create a matrix from the empty arrays");
        values_ = matrix;
        size_ = new int[] { matrix.length, matrix[0].length };
//        values_ = new Value[matrix.length][matrix[0].length];
//        for (int i = 0; i < matrix.length; ++i) {
//            for (int j = 0; j < matrix[0].length; ++j)
//                values_[i][j] = matrix[i][j];
//        }
    }

    public Matrix slice(int[] x, int[] y) {
//        System.out.printf("[DEBUG] x = [%d, %d], y = [%d, %d], size_ = [%d, %d, %d]\n", x[0], x[1], y[0], y[1], size_[0], size_[1], size_[2]);
        if (x.length != 2 || y.length != 2) throw new ArrayIndexOutOfBoundsException("Slicing size is invalid");
        if (x[0] > x[1] || y[0] > y[1]) throw new ArrayIndexOutOfBoundsException("Start is larger than end");
        if (!index_is_valid(x[0], size_[0]) || !index_is_valid(x[1] - 1, size_[0]) ||
                !index_is_valid(y[0], size_[1]) || !index_is_valid(y[1] - 1, size_[1]))
            throw new ArrayIndexOutOfBoundsException("Slicing bounds are invalid");

        int new_height = x[1] - x[0];
        int new_width = y[1] - y[0];
        var view_matrix = new Matrix(new_height, new_width, InitValues.ZEROS);
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                view_matrix.values_[i][j] = values_[i][j];
            }
        }
        return view_matrix;
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

    public Matrix multiply(IMultiDimObject other) {
        if (other == null) throw new NullPointerException("Attempt to multiply by the null matrix");
        Matrix other_matrix = (Matrix)other;
        if (size_[1] != other_matrix.size_[0]) throw new NullPointerException("Matrices have incompatible sizes to multiply");

        var matrix_array = new Value[size_[0]][other_matrix.size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < other_matrix.size_[1]; ++j) {
                matrix_array[i][j] = new Value(0);
                for (int k = 0; k < size_[1]; ++k)
                    matrix_array[i][j] = matrix_array[i][j].add(values_[i][k].multiply(other_matrix.values_[k][j]));
            }
        }

        return new Matrix(matrix_array);
    }

    public Matrix multiply(double constant) {
        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = values_[i][j].multiply(constant);
            }
        }

        return new Matrix(matrix_array);
    }

    public boolean has_same_size(IMultiDimObject other) {
        if (other == null) throw new NullPointerException("Comparison with the null matrix");
        int[] matrix_size = other.get_size();
        return matrix_size.length == size_.length && matrix_size[0] == size_[0] && matrix_size[1] == size_[1];
    }

    public Matrix add(IMultiDimObject other) {
        if (other == null) throw new NullPointerException("Attempt to add the null matrix");
        if (!has_same_size(other)) throw new RuntimeException("Matrix has invalid size for the addition");

        Matrix other_matrix = (Matrix)other;
        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = values_[i][j].add(other_matrix.values_[i][j]);
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

