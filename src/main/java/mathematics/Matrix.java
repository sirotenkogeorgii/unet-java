package main.java.mathematics;

import main.java.autograd.Value;
import main.java.mathematics.initializers.ConstantInitializer;
import main.java.mathematics.initializers.HeGaussianInitializer;
import main.java.mathematics.initializers.IInitializer;
import main.java.mathematics.initializers.RandomInitializer;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.ForkJoinPool;

import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import java.util.Arrays;
import java.util.Iterator;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.stream.IntStream;

public class Matrix extends MultiDimObject {
    private final Value[][] values_;
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

//        if (mode == ModelSettings.executionMode.PARALLEL) {
        if (false) {
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
    }

    public void set(Value value, int... indices) {
        if (indices.length != 2) throw new RuntimeException("Insufficient number of indices to access the matrix");
        if (!index_is_valid(indices[0], size_[0]) && index_is_valid(indices[1], size_[1])) throw new ArrayIndexOutOfBoundsException("Invalid index to set");
        if (value == null) throw new NullPointerException("Attempt to set null value");
        values_[indices[0]][indices[1]] = value;
    }

    public Value get(int... indices) {
        if (indices.length != 2) throw new RuntimeException("Insufficient number of indices to access the matrix");
        if (!index_is_valid(indices[0], size_[0]) && index_is_valid(indices[1], size_[1])) throw new ArrayIndexOutOfBoundsException("Attempt to get matrix value that is out of bounds");
        return values_[indices[0]][indices[1]];
    }

    private boolean index_is_valid(int index, int comparison) {
        return index >= 0 && index < comparison;
    }

    public boolean has_same_size(MultiDimObject other) {
        if (other == null) throw new NullPointerException("Comparison with the null matrix");
        int[] matrix_size = other.get_size();
        return matrix_size.length == size_.length && matrix_size[0] == size_[0] && matrix_size[1] == size_[1];
    }

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

    public Matrix relu() {

        if (mode == ModelSettings.executionMode.PARALLEL) {
            Value[][] result = Arrays.stream(values_)
                    .parallel() // Process rows in parallel
                    .map(row -> Arrays.stream(row)
                            .map(Value::relu) // Square each Value element
                            .toArray(Value[]::new)) // Collect into a new row
                    .toArray(Value[][]::new); // Collect into a new 2D array
            return new Matrix(result);
        }

        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                matrix_array[i][j] = values_[i][j].relu();
            }
        }
        return new Matrix(matrix_array);
    }

    public Matrix sigmoid() {
        if (mode == ModelSettings.executionMode.PARALLEL) {
            Value[][] result = Arrays.stream(values_)
                    .parallel() // Process rows in parallel
                    .map(row -> Arrays.stream(row)
                            .map(Value::sigmoid) // Square each Value element
                            .toArray(Value[]::new)) // Collect into a new row
                    .toArray(Value[][]::new); // Collect into a new 2D array
            return new Matrix(result);
        }

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

    public Matrix multiply(MultiDimObject other) {
        if (other == null) throw new NullPointerException("Attempt to multiply by the null matrix");
        Matrix other_matrix = (Matrix)other;
        if (size_[1] != other_matrix.size_[0]) throw new NullPointerException("Matrices have incompatible sizes to multiply");

        if (mode == ModelSettings.executionMode.PARALLEL) {
//            System.out.println("Is Parallel");
//            var result = ParallelMatrixMultiplier.multiply(values_, other_matrix.values_);
//            return new Matrix(result);

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
//        System.out.printf("[DEBUG] matrix size [%d, %d]\n", size_[0], size_[1]);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < other_matrix.size_[1]; ++j) {
                matrix_array[i][j] = new Value(0);
                for (int k = 0; k < size_[1]; ++k)
                    matrix_array[i][j] = matrix_array[i][j].add(values_[i][k].multiply(other_matrix.values_[k][j]));
            }
        }
        return new Matrix(matrix_array);
    }

    public void print() {
        System.out.printf("size_ = [%d, %d]\n", size_[0], size_[1]);
        for (int i = 0; i < size_[0]; ++i) {
            System.out.println();
            for (int j = 0; j < size_[1]; ++j) {
                System.out.printf("%f ", values_[i][j].value);
            }
        }
        System.out.println();
    }
}

class ParallelMatrixMultiplier extends RecursiveAction {
        private final Value[][] result;
        private final Value[][] matrix1;
        private final Value[][] matrix2;
        private final int startRow;
        private final int endRow;
        private static final int THRESHOLD = 1; // Adjust based on your system's performance

        // Constructor for the task
        public ParallelMatrixMultiplier(Value[][] result, Value[][] matrix1, Value[][] matrix2, int startRow, int endRow) {
            this.result = result;
            this.matrix1 = matrix1;
            this.matrix2 = matrix2;
            this.startRow = startRow;
            this.endRow = endRow;
        }

        @Override
        protected void compute() {
//            Value[] current_column = new Value[matrix2.length];
            if (endRow - startRow <= THRESHOLD) {
                for (int row = startRow; row < endRow; row++) {
                    for (int col = 0; col < matrix2[0].length; col++) {
//                        result[row][col] = new Value(0);
                        var values_array = new ArrayList<Value>();
                        for (int i = 0; i < matrix1[row].length; i++) {
//                            result[row][col] = result[row][col].add(matrix1[row][i].multiply(matrix2[i][col]));
                            values_array.add(matrix1[row][i].multiply(matrix2[i][col]));
                        }
                        result[row][col] = Value.add(values_array);
                    }
                }
            } else {
                int mid = (startRow + endRow) / 2;
                invokeAll(new ParallelMatrixMultiplier(result, matrix1, matrix2, startRow, mid),
                        new ParallelMatrixMultiplier(result, matrix1, matrix2, mid, endRow));
            }
        }

    public static Value[][] multiply(Value[][] matrix1, Value[][] matrix2) {
        if (matrix1[0].length != matrix2.length) {
            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
        }

        Value[][] result = new Value[matrix1.length][matrix2[0].length];
        ForkJoinPool pool = ForkJoinPool.commonPool();
        pool.invoke(new ParallelMatrixMultiplier(result, matrix1, matrix2, 0, matrix1.length));
        return result;
    }
}