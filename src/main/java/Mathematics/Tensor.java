package main.java.mathematics;

import jdk.jshell.spi.ExecutionControl;
import main.java.autograd.Value;
import main.java.nn.layers.ILayer;

import java.util.Iterator;
import java.util.Random;

public class Tensor implements IMultiDimObject {

    private Value[][][] values_;
    private int[] size_; // [height, width, depth]
    public Tensor(int height, int width, int depth, IMultiDimObject.InitValues init_values) {
        if (height < 1 || width < 1 || depth < 1)
            throw new RuntimeException("Tensor has non-positive dimensions");

        size_ = new int[] { height, width, depth };
        var random = new Random();
        values_ = new Value[height][width][depth];
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int k = 0; k < depth; ++k) {
                    double current_value = switch (init_values) { case ZEROS -> 0; case RANDOM -> random.nextDouble(-0.25, 0.25); default -> throw new RuntimeException("Unknown value to fill"); };
                    values_[i][j][k] = new Value(current_value);
                }
            }
        }
    }

    public Tensor(Value[][][] tensor) {
        if (tensor == null) throw new RuntimeException("Array to create a tensor is null");
        if (tensor.length == 0 || tensor[0].length == 0 || tensor[0][0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create a tensor from the empty arrays");
        values_ = tensor;
        size_ = new int[] { tensor.length, tensor[0].length, tensor[0][0].length};
//        values_ = new Value[tensor.length][matrix[0].length];
//        for (int i = 0; i < matrix.length; ++i) {
//            for (int j = 0; j < matrix[0].length; ++j)
//                values_[i][j] = matrix[i][j];
//        }
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

    public Tensor(double[][][] tensor) {
        if (tensor.length == 0 || tensor[0].length == 0 || tensor[0][0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create tensor from the empty arrays");
        size_ = new int[] { tensor.length, tensor[0].length, tensor[0][0].length };
        values_ = new Value[tensor.length][tensor[0].length][tensor[0][0].length];
        for (int i = 0; i < tensor.length; ++i) {
            for (int j = 0; j < tensor[0].length; ++j)
                for (int k = 0; k < tensor[0][0].length; ++k) {
                    values_[i][j][k] = new Value(tensor[i][j][k]);
                }
        }
    }

    public int[] get_size() { return size_; }
    private boolean index_is_valid(int index, int comparison) {
        return index >= 0 && index < comparison;
    }
    public Tensor slice(int[] x, int[] y) {
//        System.out.printf("[DEBUG] x = [%d, %d], y = [%d, %d], size_ = [%d, %d, %d]\n", x[0], x[1], y[0], y[1], size_[0], size_[1], size_[2]);
        if (x.length != 2 || y.length != 2) throw new ArrayIndexOutOfBoundsException("Slicing size is invalid");
        if (x[0] > x[1] || y[0] > y[1]) throw new ArrayIndexOutOfBoundsException("Start is larger than end");
        if (!index_is_valid(x[0], size_[0]) || !index_is_valid(x[1] - 1, size_[0]) ||
                !index_is_valid(y[0], size_[1]) || !index_is_valid(y[1] - 1, size_[1]))
            throw new ArrayIndexOutOfBoundsException("Slicing bounds are invalid");

        int new_height = x[1] - x[0];
        int new_width = y[1] - y[0];
        var view_tensor = new Tensor(new_height, new_width, size_[2], InitValues.ZEROS);
        for (int i = x[0]; i < x[1]; ++i) {
            for (int j = y[0]; j < y[1]; ++j) {
                if (size_[2] >= 0)
                    System.arraycopy(values_[i][j], 0, view_tensor.values_[i - x[0]][j - y[0]], 0, size_[2]);
            }
        }
        return view_tensor;
    }

    private boolean has_same_size(Tensor other) {
        if (other == null) throw new NullPointerException("Comparison with the null tensor");
        int[] tenor_size = other.get_size();
        return tenor_size[0] == size_[0] && tenor_size[1] == size_[1] && tenor_size[2] == size_[2];
    }

    public Matrix get_channel_vector(int x, int y) {
        var vector_array = new Value[size_[2]][1];
        for (int k = 0; k < size_[2]; ++k)
            vector_array[k][0] = values_[x][y][k];
        return new Matrix(vector_array);
    }

    public Tensor pw_multiply(Tensor other) {
        if (!has_same_size(other))
            throw new RuntimeException("Tensor has invalid size for the pairwise mul");
        var output_tensor = new Tensor(size_[0], size_[1], size_[2], InitValues.RANDOM);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    output_tensor.values_[i][j][k] = other.values_[i][j][k].multiply(values_[i][j][k]);
            }
        }
        return output_tensor;
    }

    public Tensor add(IMultiDimObject other) {
        if (other == null) throw new NullPointerException("Attempt to add the null matrix");
        if (!has_same_size(other)) throw new RuntimeException("Matrix has invalid size for the addition");

        Tensor other_tensor = (Tensor)other;
        var tensor_array = new Value[size_[0]][size_[1]][size_[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = values_[i][j][k].add(other_tensor.values_[i][j][k]);
                }
            }
        }
        return new Tensor(tensor_array);
    }

    public Tensor add_vector(Matrix vector) {
        if (vector == null) throw new RuntimeException("Attempt to add null to the tensor");
        if (!vector.is_vector()) throw new RuntimeException("Input is not a vector");

        int[] vector_size = vector.get_size();
        if (vector_size[0] != size_[2]) {
            System.out.printf("ERROR: %d != %d\n", vector_size[0], size_[2]);
            throw new RuntimeException("Vector has invalid size to be added");
        }

        var tensor_array = new Value[size_[0]][size_[1]][size_[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = values_[i][j][k].add(vector.get(k, 0));
                }
            }
        }
        return new Tensor(tensor_array);
    }

    public Tensor multiply_vector(Matrix vector) {
        if (vector == null) throw new RuntimeException("Attempt to add null to the tensor");
        if (!vector.is_vector()) throw new RuntimeException("Input is not a vector");

        int[] vector_size = vector.get_size();
        if (vector_size[0] != size_[2]) throw new RuntimeException("Vector has invalid size to be added");

        var tensor_array = new Value[size_[0]][size_[1]][size_[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = values_[i][j][k].multiply(vector.get(k, 0));
                }
            }
        }
        return new Tensor(tensor_array);
    }

    public Tensor multiply(IMultiDimObject other)  {
        throw new RuntimeException("Multiply for tensors is not implemented");
//        return null;
    }

    public Tensor multiply(double constant) {
        var tensor_array = new Value[size_[0]][size_[1]][size_[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = values_[i][j][k].multiply(constant);
                }
            }
        }

        return new Tensor(tensor_array);
    }

    public boolean has_same_size(IMultiDimObject other) {
        if (other == null) throw new NullPointerException("Comparison with a null tensor");
        int[] tensor_size = other.get_size();
        return tensor_size.length == size_.length && tensor_size[0] == size_[0] && tensor_size[1] == size_[1] && tensor_size[2] == size_[2];
    }

    public boolean is_vector() { return size_[1] == 1 && size_[2] == 1; }

    public Value sum() {
        Value value = new Value(0);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    value = value.add(values_[i][j][k]);
            }
        }
        return value;
    }

    public Matrix sum_channels() {
        Matrix output_matrix = new Matrix(size_[0], size_[1], InitValues.ZEROS);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    output_matrix.set(new int[] {i, j}, output_matrix.get(i, j).add(values_[i][j][k]));
            }
        }
        return output_matrix;
    }

    private boolean is_tensor_index(int[] index) {
        return index.length == 3 && index_is_valid(index[0], size_[0]) && index_is_valid(index[1], size_[1]) && index_is_valid(index[2], size_[2]);
    }

    public void set(int[] index, Value value) {
        if (!is_tensor_index(index))
            throw new ArrayIndexOutOfBoundsException("Invalid index to set");
        if (value == null)
            throw new NullPointerException("Attempt to set null value");
        values_[index[0]][index[1]][index[2]] = value;
    }

    public Value get(int[] index) {
        if (!is_tensor_index(index))
            throw new ArrayIndexOutOfBoundsException("Attempt to get matrix value that is out of bounds");
        return values_[index[0]][index[1]][index[2]];
    }

    public void backward() {
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    values_[i][j][k].backward();
            }
        }
    }

    public Matrix get_dim(int dim) {
        if (dim < 0 || dim >= size_[2]) throw new RuntimeException("Attempt to get out of bounds dimension");

        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j)
                matrix_array[i][j] = values_[i][j][dim];
        }

        return new Matrix(matrix_array);
    }

    public Tensor concatenate(Tensor other) {
        if (other == null) throw new RuntimeException("Attempt to concatenate null tensor");
        int[] other_size = other.get_size();
        if (other_size[0] != size_[0] || other_size[1] != size_[1]) throw new RuntimeException("Height and width are not same to concatenate");

        var tensor_array = new Value[size_[0]][size_[1]][size_[2] + other_size[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = values_[i][j][k];
                }
            }
        }
        for (int i = 0; i < other_size[0]; ++i) {
            for (int j = 0; j < other_size[1]; ++j) {
                for (int k = 0; k < other_size[2]; ++k) {
                    tensor_array[i][j][k + size_[2]] = other.values_[i][j][k];
                }
            }
        }

        return new Tensor(tensor_array);
    }

    public Tensor relu() {
        var tensor_array = new Value[size_[0]][size_[1]][size_[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = values_[i][j][k].relu();
                }
            }
        }
        return new Tensor(tensor_array);
    }

    public Iterator<Value> iterator() {
        return new Iterator<Value>() {
            int current_index = 0;
            int matrix_size = size_[0] * size_[1];
            int values_num = size_[0] * size_[1] * size_[2];
            @Override
            public boolean hasNext() {
                return current_index != values_num;
            }

            @Override
            public Value next() {
                var value = values_[(current_index % matrix_size) / size_[1]][(current_index % matrix_size) % size_[1]][current_index / matrix_size];
                current_index++;
                return value;
            }
        };
    }

    public void print() {
        System.out.printf("size_ = [%d, %d, %d]\n", size_[0], size_[1], size_[2]);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    System.out.printf("%f ", values_[i][j][k].get_value());
            }
            System.out.println();
        }
    }
}
