package mathematics;

import autograd.Value;
import mathematics.initializers.ConstantInitializer;
import mathematics.initializers.HeGaussianInitializer;
import mathematics.initializers.IInitializer;
import mathematics.initializers.RandomInitializer;
import nn.layers.Layer;
import nn.models.ModelSettings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.IntStream;

/**
 * Represents a three-dimensional tensor of {@link Value} objects. This class
 * facilitates the manipulation and operation of data in a multi-dimensional array format,
 * which is essential in neural network computations and other mathematical applications.
 */
public class Tensor extends MultiDimObject {
    private Value[][][] values_; // [height, width, depth]

    /**
     * Constructs a tensor with specified dimensions and initialization.
     *
     * @param height The number of rows in the tensor.
     * @param width The number of columns in the tensor.
     * @param depth The depth or third dimension of the tensor.
     * @param init_values The method for initializing the tensor values, using predefined schemes like ZEROS, ONES, etc.
     */
    public Tensor(int height, int width, int depth, MultiDimObject.InitValues init_values) {
        if (height < 1 || width < 1 || depth < 1)
            throw new RuntimeException("Tensor has non-positive dimensions");
        size_ = new int[] { height, width, depth };
        values_ = new Value[height][width][depth];

        IInitializer sampler = switch (init_values) {
            case ZEROS -> new ConstantInitializer(0);
            case ONES -> new ConstantInitializer(1);
            case HE -> new HeGaussianInitializer(height * width * depth);
            case RANDOM -> new RandomInitializer(-0.25, 0.25);
            default -> throw new RuntimeException("Unknown sampler");
        };

        if (mode == ModelSettings.executionMode.PARALLEL) {
            IntStream.range(0, height).parallel().forEach(i -> {
                IntStream.range(0, width).forEach(j -> {
                    IntStream.range(0, depth).forEach(k -> {
                        values_[i][j][k] = new Value(sampler.next());
                    });
                });
            });
        } else {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    for (int k = 0; k < depth; ++k) {
                        values_[i][j][k] = new Value(sampler.next());
                    }
                }
            }
        }
    }

    /**
     * Constructs a tensor from a three-dimensional array of {@link Value} objects.
     *
     * @param tensor A three-dimensional array of {@link Value} objects to initialize the tensor.
     */
    public Tensor(Value[][][] tensor) {
        if (tensor == null) throw new RuntimeException("Array to create a tensor is null");
        if (tensor.length == 0 || tensor[0].length == 0 || tensor[0][0].length == 0)
            throw new ArrayIndexOutOfBoundsException("Attempt to create a tensor from the empty arrays");
        values_ = tensor;
        size_ = new int[] { tensor.length, tensor[0].length, tensor[0][0].length};
    }

    /**
     * Constructs a tensor from an array of {@link Matrix} objects.
     *
     * @param matrices An array of {@link Matrix} objects, each representing a "slice" or layer of the tensor.
     */
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

    /**
     * Constructs a tensor from a three-dimensional array of double values.
     *
     * @param tensor A three-dimensional array of doubles to initialize the tensor.
     */
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

    /**
     * Checks if the specified index is within the valid range for a dimension of the tensor.
     * This utility method ensures that index access errors are prevented by validating
     * that the index is within the acceptable bounds of [0, comparison).
     *
     * @param index The index to be checked for validity.
     * @param comparison The upper boundary (exclusive) for the valid index range,
     *                   typically the size of the dimension being accessed.
     * @return true if the index is within the range [0, comparison); false otherwise.
     */
    private boolean index_is_valid(int index, int comparison) {
        return index >= 0 && index < comparison;
    }

    /**
     * Extracts a sub-tensor from this tensor, specified by range indices in two dimensions.
     * This method creates a view or sub-tensor based on the specified slice indices for rows and columns.
     *
     * @param x An array of two integers specifying the start and end indices for the row dimension.
     *          The end index is exclusive.
     * @param y An array of two integers specifying the start and end indices for the column dimension.
     *          The end index is exclusive.
     * @return A new {@link Tensor} representing the sliced portion of this tensor.
     * @throws ArrayIndexOutOfBoundsException If the length of either 'x' or 'y' is not exactly two,
     *                                        or if the indices are out of the valid range of this tensor's dimensions.
     */
    public Tensor slice(int[] x, int[] y) {
        if (x.length != 2 || y.length != 2) throw new ArrayIndexOutOfBoundsException("Slicing size is invalid");

        var view_tensor = create_subtensor(x, y);
        for (int i = x[0]; i < x[1]; ++i) {
            for (int j = y[0]; j < y[1]; ++j) {
                if (size_[2] >= 0)
                    System.arraycopy(values_[i][j], 0, view_tensor.values_[i - x[0]][j - y[0]], 0, size_[2]);
            }
        }
        return view_tensor;
    }

    /**
     * Creates a sub-tensor based on the specified slice indices for dimensions.
     * This method validates the slicing indices to ensure they are within bounds and correctly ordered. (used by slice() method)
     *
     * @param x An array of two integers specifying the start and end indices for the row dimension.
     * @param y An array of two integers specifying the start and end indices for the column dimension.
     * @return A new {@link Tensor} that represents a slice of the original tensor based on the provided indices.
     * @throws ArrayIndexOutOfBoundsException If the start index is greater than the end index,
     *                                        or if the indices are outside the tensor's dimension bounds.
     */
    private Tensor create_subtensor(int[] x, int[] y) {
        if (x[0] > x[1] || y[0] > y[1]) throw new ArrayIndexOutOfBoundsException("Start is larger than end");
        if (!index_is_valid(x[0], size_[0]) || !index_is_valid(x[1] - 1, size_[0]) ||
                !index_is_valid(y[0], size_[1]) || !index_is_valid(y[1] - 1, size_[1]))
            throw new ArrayIndexOutOfBoundsException("Slicing bounds are invalid");

        int new_height = x[1] - x[0];
        int new_width = y[1] - y[0];
        return new Tensor(new_height, new_width, size_[2], InitValues.ZEROS);
    }

    /**
     * Checks if another tensor has the same dimensions as this tensor.
     *
     * @param other The other tensor to compare against.
     * @return true if the other tensor has the same dimensions; false otherwise.
     * @throws NullPointerException If the other tensor is null.
     */
    private boolean has_same_size(Tensor other) {
        if (other == null) throw new NullPointerException("Comparison with the null tensor");
        int[] tenor_size = other.get_size();
        return tenor_size[0] == size_[0] && tenor_size[1] == size_[1] && tenor_size[2] == size_[2];
    }

    /**
     * Performs an element-wise (pairwise) multiplication of this tensor with another tensor.
     * Both tensors must have the same size. The result is a new tensor where each element is
     * the product of corresponding elements from this tensor and the other tensor.
     *
     * @param other The tensor to multiply with this tensor.
     * @return A new tensor representing the element-wise product.
     * @throws RuntimeException If the other tensor is not of the same size.
     */
    public Tensor pw_multiply(Tensor other) {
        if (!has_same_size(other))
            throw new RuntimeException("Tensor has invalid size for the pairwise mul");
        var output_tensor = new Tensor(size_[0], size_[1], size_[2], InitValues.RANDOM);

        if (mode == ModelSettings.executionMode.PARALLEL) {
            IntStream.range(0, size_[0]).parallel().forEach(i -> {
                for (int j = 0; j < size_[1]; ++j) {
                    for (int k = 0; k < size_[2]; ++k)
                        output_tensor.values_[i][j][k] = other.values_[i][j][k].multiply(values_[i][j][k]);
                }
            });

            return output_tensor;
        }

        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    output_tensor.values_[i][j][k] = other.values_[i][j][k].multiply(values_[i][j][k]);
            }
        }
        return output_tensor;
    }

    /**
     * Adds a tensor to this tensor and returns the result as a new tensor.
     *
     * @param other The tensor to add to this tensor.
     * @return A new tensor representing the sum of this tensor and the other tensor.
     */
    public Tensor add(MultiDimObject other) {
        if (other == null) throw new NullPointerException("Attempt to add the null matrix");
        if (!has_same_size(other)) throw new RuntimeException("Matrix has invalid size for the addition");
        Tensor other_tensor = (Tensor)other;

        if (mode == ModelSettings.executionMode.PARALLEL) {
            IntStream.range(0, size_[0]).parallel().forEach(i -> {
                for (int j = 0; j < size_[1]; ++j) {
                    for (int k = 0; k < size_[2]; ++k)
                        other_tensor.values_[i][j][k] = other_tensor.values_[i][j][k].add(values_[i][j][k]);
                }
            });

            return other_tensor;
        }

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

    public Tensor multiply(MultiDimObject other)  {
        throw new RuntimeException("Multiply for tensors is not implemented");
    }

    /**
     * Multiplies each element of this tensor with a constant and returns the result as a new tensor.
     *
     * @param constant The constant to multiply with each element of the tensor.
     * @return A new tensor where each element is the product of the original element and the constant.
     */
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

    /**
     * Determines if another MultiDimObject (such as another Tensor) has the same size as this tensor.
     *
     * @param other The other MultiDimObject to compare against.
     * @return true if the other object has the same dimensions as this tensor; false otherwise.
     * @throws NullPointerException if the other object is null.
     */
    public boolean has_same_size(MultiDimObject other) {
        if (other == null) throw new NullPointerException("Comparison with a null tensor");
        int[] tensor_size = other.get_size();
        return tensor_size.length == size_.length && tensor_size[0] == size_[0] && tensor_size[1] == size_[1] && tensor_size[2] == size_[2];
    }

    /**
     * Sets the 'requires_grad' property for all elements within the tensor, determining
     * whether automatic differentiation mechanisms should track operations for each element.
     *
     * @param requires_grad Boolean flag indicating whether gradients should be calculated.
     */
    public void set_requires_grad(boolean requires_grad) {
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    values_[i][j][k].requires_grad = requires_grad;
            }
        }
    }

    /**
     * Checks if the tensor qualifies as a vector, defined as having a size of 1 in two dimensions.
     *
     * @return true if the tensor is a vector; false otherwise.
     */
    @Override
    public boolean is_vector() {
        return size_[1] == 1 && size_[2] == 1;
    }

    /**
     * Calculates the sum of all elements in the tensor.
     *
     * @return A Value object representing the sum of all elements.
     */
    public Value sum() {
        var values_array = new ArrayList<Value>();

        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    values_array.add(values_[i][j][k]);
                }
            }
        }
        return Value.add(values_array);
    }

    /**
     * Checks if a provided index array correctly specifies an index within the tensor.
     *
     * @param index Array of integers representing the index to check.
     * @return true if the index is within the tensor's bounds; false otherwise.
     */
    private boolean is_tensor_index(int[] index) {
        return index.length == 3 && index_is_valid(index[0], size_[0]) && index_is_valid(index[1], size_[1]) && index_is_valid(index[2], size_[2]);
    }

    /**
     * Sets a specified value at given indices within the tensor.
     *
     * @param value The value to set within the tensor.
     * @param indices The indices where the value should be set, expected as an array of three integers.
     * @throws RuntimeException If the number of indices is not three.
     * @throws ArrayIndexOutOfBoundsException If the indices are out of the tensor's bounds.
     * @throws NullPointerException If the provided value is null.
     */
    public void set(Value value, int... indices) {
        if (indices.length != 3) throw new RuntimeException("Insufficient number of indices to access the matrix");
        if (!is_tensor_index(indices)) throw new ArrayIndexOutOfBoundsException("Invalid index to set");
        if (value == null) throw new NullPointerException("Attempt to set null value");
        values_[indices[0]][indices[1]][indices[2]] = value;
    }

    /**
     * Retrieves a value from the tensor at specified indices.
     *
     * @param indices An array of three integers representing the indices of the value to retrieve.
     * @return The value at the specified indices.
     * @throws RuntimeException If the number of indices provided is not three.
     * @throws ArrayIndexOutOfBoundsException If the indices are out of the tensor's bounds.
     */
    public Value get(int... indices) {
        if (indices.length != 3) throw new RuntimeException("Insufficient number of indices to access the tensorx");
        if (!is_tensor_index(indices))
            throw new ArrayIndexOutOfBoundsException("Attempt to get matrix value that is out of bounds");
        return values_[indices[0]][indices[1]][indices[2]];
        }

    /**
     * Extracts a specific dimension from the tensor as a Matrix. This is often used in operations that
     * need to process or manipulate one layer or slice of a tensor at a time.
     *
     * @param dim The dimension index to extract, corresponding to the third dimension of the tensor.
     * @return A Matrix containing the specified dimension's data.
     * @throws RuntimeException If the specified dimension index is out of bounds.
     */
    public Matrix get_dim(int dim) {
        if (dim < 0 || dim >= size_[2]) throw new RuntimeException("Attempt to get out of bounds dimension");

        var matrix_array = new Value[size_[0]][size_[1]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j)
                matrix_array[i][j] = values_[i][j][dim];
        }

        return new Matrix(matrix_array);
    }


    /**
     * Applies the Rectified Linear Unit (ReLU) activation function to each element of the tensor.
     * The ReLU function is defined as f(x) = max(0, x), setting all negative elements to zero
     * and keeping positive values unchanged.
     *
     * @return A new {@link Tensor} with the ReLU activation function applied to each element.
     */
    public Tensor relu() {
        return activation(Layer.Activation.ReLU);
    }

    /**
     * Applies the Leaky Rectified Linear Unit (LeakyReLU) activation function to each element of the tensor.
     * The LeakyReLU function allows a small, positive gradient when the unit is not active:
     * f(x) = x for x > 0, and f(x) = 0.01 * x for x less than 0.
     *
     * @return A new {@link Tensor} with the LeakyReLU activation function applied to each element.
     */
    public Tensor leakyRelu() {
        return activation(Layer.Activation.LeakyReLU);
    }

    /**
     * Applies the Sigmoid activation function to each element of the tensor.
     * The Sigmoid function is defined as f(x) = 1 / (1 + exp(-x)), which squashes each element
     * to a range between 0 and 1, making it suitable for probabilities or models where outputs are binary.
     *
     * @return A new {@link Tensor} with the Sigmoid activation function applied to each element.
     */
    public Tensor sigmoid() {
        return activation(Layer.Activation.Sigmoid);
    }

    /**
     * Applies an activation function (ReLU, LeakyReLU, Sigmoid) to each element of the tensor and returns the result as a new tensor.
     *
     * @param activation The activation function to apply.
     * @return A new tensor with the activation function applied to each element.
     */
    protected Tensor activation(Layer.Activation activation) {
        java.util.function.Function<Value, Value> activation_function = switch (activation) {
            case ReLU -> Value::relu;
            case LeakyReLU -> Value::leakyRelu;
            case Sigmoid -> Value::sigmoid;
            default -> throw new RuntimeException("Unknown activation function");
        };

        if (mode == ModelSettings.executionMode.PARALLEL) {
            Value[][][] result = Arrays.stream(values_)
                    .parallel()
                    .map(twoDMatrix -> Arrays.stream(twoDMatrix)
                            .parallel()
                            .map(row -> Arrays.stream(row)
                                    .map(activation_function)
                                    .toArray(Value[]::new))
                            .toArray(Value[][]::new))
                    .toArray(Value[][][]::new);
            return new Tensor(result);
        }

        var tensor_array = new Value[size_[0]][size_[1]][size_[2]];
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k) {
                    tensor_array[i][j][k] = activation_function.apply(values_[i][j][k]);
                }
            }
        }
        return new Tensor(tensor_array);
    }

    /**
     * Provides an iterator over the elements of the tensor, flattening all dimensions.
     *
     * @return An iterator that allows iterating over all elements of the tensor.
     */
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

    /**
     * Prints the tensor to the standard output in a formatted manner.
     */
    public void print() {
        System.out.printf("size_ = [%d, %d, %d]", size_[0], size_[1], size_[2]);
        for (int i = 0; i < size_[0]; ++i) {
            for (int j = 0; j < size_[1]; ++j) {
                for (int k = 0; k < size_[2]; ++k)
                    System.out.printf("%f ", values_[i][j][k].value);
            }
            System.out.println();
        }
        System.out.println();
    }
}
