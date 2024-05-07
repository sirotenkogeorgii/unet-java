package main.java.datasets;

import main.java.mathematics.Tensor;

/**
 * Interface defining the essential methods required for accessing data in datasets.
 * This interface ensures that implementing classes provide mechanisms to retrieve
 * dataset data as Tensors and to obtain the size of the dataset.
 */
public interface IData {

    /**
     * Retrieves the data of the dataset encapsulated in a Tensor object.
     * This method is intended to return the entire set of data contained within the dataset as a Tensor
     *
     * @return A {@link Tensor} representing the dataset's data.
     */
    public Tensor get_data();

    /**
     * Retrieves the size of the dataset as an array of integers.
     * This method typically returns the dimensions of the dataset, where each element
     * in the returned array represents the size of the dataset along a specific dimension.
     * For example, in a 2D dataset, the array might contain [rows, columns].
     *
     * @return An array of integers where each element denotes the size of the dataset
     *         along a particular dimension.
     */
    public int[] get_size();
}
