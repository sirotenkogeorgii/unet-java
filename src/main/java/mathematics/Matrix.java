package main.java.mathematics;

import main.java.autograd.Value;
import main.java.mathematics.initializers.ConstantInitializer;
import main.java.mathematics.initializers.HeGaussianInitializer;
import main.java.mathematics.initializers.IInitializer;
import main.java.mathematics.initializers.RandomInitializer;
import main.java.nn.layers.Layer;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.IntStream;

/**
 * Represents a matrix of {@link Value} objects. This class provides functionality for
 * various matrix operations such as addition, multiplication, and applying activation functions,
 * supporting both sequential and parallel execution modes.
 */
public class Matrix extends MultiDimObject {
    private final Value[][] values_;

    /**
     * Initializes a matrix with given dimensions and initialization settings.
     *
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param init_values The initialization method for the matrix elements.
     */
    public Matrix(int height, int width, InitValues init_values) {
        if (height < 1 || width < 1) throw new RuntimeException("Matrix has non-positive dimensions");
        size_ = new int[] { height, width };
        values_ = new Value[height][width];

        IInitializer sampler = switch (init_values) {
            case ZEROS -> new ConstantInitializer(0);
            case ONES -> new ConstantInitializer(1);
            case HE -> new HeGaussianInitializer(width);
            case RANDOM -> new RandomInitializer(-0.25, 0.25);
            default -> throw new RuntimeException("Unknown sampler");
        };

        if (mode == ModelSettings.executionMode.PARALLEL) {
            IntStream.range(0, height).parallel().forEach(i -> {
                IntStream.range(0, width).forEach(j -> {
                    values_[i][j] = new Value(sampler.next());
                });
            });
        } else {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    values_[i][j] = new Value(sampler.next());
                }
            }
        }
    }

    /**
     * Initializes a matrix from a 2D array of doubles. Each element in the array
     * represents a matrix element.
     *
     * @param matrix The 2D array of doubles to initialize the matrix.
     */
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

    /**
     * Initializes a matrix directly from a 2D array of {@link Value} objects.
     *
     * @param matrix The 2D array of {@link Value} objects to initialize the matrix.
     */
    public Matrix(Value[][] matrix) {
        if (matrix == null) throw new RuntimeException("Array to create a matrix is null");
        if (matrix.length == 0 || matrix[0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create a matrix from the empty arrays");
        values_ = matrix;
        size_ = new int[] { matrix.length, matrix[0].length };
    }

    /**
     * Sets the value at the specified indices in the matrix.
     *
     * @param value The value to set at the specified indices.
     * @param indices The row and column indices where the value should be set.
     */
    public void set(Value value, int... indices) {
        if (indices.length != 2) throw new RuntimeException("Insufficient number of indices to access the matrix");
        if (!index_is_valid(indices[0], size_[0]) && index_is_valid(indices[1], size_[1])) throw new ArrayIndexOutOfBoundsException("Invalid index to set");
        if (value == null) throw new NullPointerException("Attempt to set null value");
        values_[indices[0]][indices[1]] = value;
    }

    /**
     * Retrieves the value from the matrix at the specified indices.
     *
     * @param indices The row and column indices of the value to retrieve.
     * @return The value at the specified indices.
     */
    public Value get(int... indices) {
        if (indices.length != 2) throw new RuntimeException("Insufficient number of indices to access the matrix");
        if (!index_is_valid(indices[0], size_[0]) && index_is_valid(indices[1], size_[1])) throw new ArrayIndexOutOfBoundsException("Attempt to get matrix value that is out of bounds");
        return values_[indices[0]][indices[1]];
    }

    /**
     * Checks if the specified index is within the valid range of [0, comparison).
     * This method is used internally to ensure that matrix access operations do not exceed
     * the dimensions of the matrix, helping to prevent runtime errors due to invalid index access.
     *
     * @param index The index to check for validity.
     * @param comparison The maximum allowable value for the index, exclusive. This is typically
     *                   the size of the dimension being accessed.
     * @return true if the index is within the valid range; false otherwise.
     */
    private boolean index_is_valid(int index, int comparison) {
        return index >= 0 && index < comparison;
    }

    /**
     * Checks if another {@link MultiDimObject} has the same dimensions as this matrix.
     *
     * @param other The other {@link MultiDimObject} to compare against.
     * @return true if the other object has the same dimensions; false otherwise.
     */
    public boolean has_same_size(MultiDimObject other) {
        if (other == null) throw new NullPointerException("Comparison with the null matrix");
        int[] matrix_size = other.get_size();
        return matrix_size.length == size_.length && matrix_size[0] == size_[0] && matrix_size[1] == size_[1];
    }

    /**
     * Adds another matrix to this matrix.
     *
     * @param other The matrix to add to this one.
     * @return A new matrix representing the sum of this matrix and the other matrix.
     */
    public Matrix add(MultiDimObject other) {
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

    /**
     * Applies the Rectified Linear Unit (ReLU) activation function to each element of the matrix.
     * The ReLU function is defined as f(x) = max(0, x), setting all negative elements to zero,
     * and keeping positive values unchanged.
     *
     * @return A new {@link Matrix} with the ReLU activation function applied to each element.
     */
    public Matrix relu() {
        return activation(Layer.Activation.ReLU);
    }

    /**
     * Applies the Leaky Rectified Linear Unit (LeakyReLU) activation function to each element of the matrix.
     * The LeakyReLU function is defined as f(x) = x for x > 0, and f(x) = 0.01 * x for x less than 0.
     * This allows a small, non-zero gradient when the unit is not active and prevents neurons from dying.
     *
     * @return A new {@link Matrix} with the LeakyReLU activation function applied to each element.
     */
    public Matrix leakyRelu() {
        return activation(Layer.Activation.LeakyReLU);
    }

    /**
     * Applies the Sigmoid activation function to each element of the matrix.
     * The Sigmoid function is defined as f(x) = 1 / (1 + exp(-x)), which squashes each element
     * to be between 0 and 1, making it suitable for models where we need to predict probabilities.
     *
     * @return A new {@link Matrix} with the Sigmoid activation function applied to each element.
     */
    public Matrix sigmoid() {
        return activation(Layer.Activation.Sigmoid);
    }

    /**
     * Applies a specified activation function to each element of the matrix.
     *
     * @param activation The activation function name to apply.
     * @return A new matrix with the activation function applied to each element.
     */
    protected Matrix activation(Layer.Activation activation) {
        java.util.function.Function<Value, Value> activation_function = switch (activation) {
            case ReLU -> Value::relu;
            case LeakyReLU -> Value::leakyRelu;
            case Sigmoid -> Value::sigmoid;
            default -> throw new RuntimeException("Unknown activation function");
        };

        if (mode == ModelSettings.executionMode.PARALLEL) {
            Value[][] result = Arrays.stream(values_)
                    .parallel()
                    .map(row -> Arrays.stream(row)
                            .map(activation_function)
                            .toArray(Value[]::new))
                    .toArray(Value[][]::new);
            return new Matrix(result);
        }

        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = activation_function.apply(values_[i][j]);
            }
        }

        return new Matrix(matrix_array);
    }

    /**
     * Determines if the matrix is a vector (i.e., 2nd dimension is 1).
     *
     * @return true if the matrix is a vector; false otherwise.
     */
    public boolean is_vector() {
        return size_[1] == 1;
    }

    /**
     * Transposes this matrix, switching its rows and columns.
     *
     * @return A new matrix that is the transpose of this matrix.
     */
    public Matrix transpose() {
        var matrix_array = new Value[size_[1]][size_[0]];
        for (int i = 0; i < size_[1]; ++i) {
            for (int j = 0; j < size_[0]; ++j) {
                matrix_array[i][j] = values_[j][i];
            }
        }
        return new Matrix(matrix_array);
    }

    /**
     * Returns an iterator for the elements of the matrix.
     *
     * @return An iterator over the elements of the matrix.
     */
    public Iterator<Value> iterator() {
        return new Iterator<Value>() {
            int current_index = 0;
            int values_num = size_[0] * size_[1];
            @Override
            public boolean hasNext() {
                return current_index != values_num;
            }

            @Override
            public Value next() {
                var value = values_[current_index / size_[1]][current_index % size_[1]];
                current_index++;
                return value;
            }
        };
    }

    /**
     * Multiplies this matrix by another matrix.
     *
     * @param other The matrix to multiply with this one.
     * @return A new matrix representing the multiplication of this matrix and the other matrix.
     */
    public Matrix multiply(MultiDimObject other) {
        if (other == null) throw new NullPointerException("Attempt to multiply by the null matrix");
        Matrix other_matrix = (Matrix)other;
        if (size_[1] != other_matrix.size_[0]) throw new NullPointerException("Matrices have incompatible sizes to multiply");

        if (mode == ModelSettings.executionMode.PARALLEL) {
            var matrix_array = new Value[size_[0]][other_matrix.size_[1]];
            for (int i = 0; i < size_[0]; ++i) {
                for (int j = 0; j < other_matrix.size_[1]; ++j) {
                    var values_array = new ArrayList<Value>();
                    for (int k = 0; k < size_[1]; ++k) {
                        values_array.add(values_[i][k].multiply(other_matrix.values_[k][j]));
                    }
                    matrix_array[i][j] = Value.add(values_array);
                }
            }
            return new Matrix(matrix_array);
        }

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

    /**
     * Prints the matrix to the standard output.
     */
    public void print() {
        System.out.printf("size_ = [%d, %d]", size_[0], size_[1]);
        for (int i = 0; i < size_[0]; ++i) {
            System.out.println();
            for (int j = 0; j < size_[1]; ++j) {
                System.out.printf("%f ", values_[i][j].value);
            }
        }
        System.out.println();
    }
}