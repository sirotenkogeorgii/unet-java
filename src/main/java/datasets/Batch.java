package main.java.datasets;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * Represents a batch of dataset samples, providing iterable access to the samples within it.
 * This class encapsulates a collection of {@link IDatasetSample} and facilitates batch operations
 * that are common in machine learning and data processing tasks, such as mini-batch processing
 * in training algorithms.
 */
public class Batch implements Iterable<IDatasetSample> {
    private ArrayList<IDatasetSample> data_;
    private int size_;

    /**
     * Constructs a new Batch with a list of dataset samples.
     *
     * @param data An ArrayList of {@link IDatasetSample} that constitutes the batch.
     * @throws RuntimeException if the provided ArrayList is null.
     */
    public Batch(ArrayList<IDatasetSample> data) {
        if (data == null) throw new RuntimeException("Input data cannot be null.");
        data_ = data;
        size_ = data_.size();
    }

    /**
     * Returns the number of samples in this batch.
     *
     * @return The size of the batch as an integer.
     */
    public int get_size() {
        return size_;
    }

    /**
     * Provides an iterator over the dataset samples in this batch.
     * The iterator supports basic iteration and does not support remove operations.
     *
     * @return An {@link Iterator} of {@link IDatasetSample}, allowing for traversing the dataset samples.
     */
    @Override
    public Iterator<IDatasetSample> iterator() {
        return new Iterator<>() {
            private int current_index_ = 0;

            /**
             * Returns true if the iteration has more samples.
             *
             * @return true if there is at least one more sample to iterate over; false otherwise.
             */
            public boolean hasNext() {
                return current_index_ < size_;
            }

            /**
             * Returns the next dataset sample in the iteration.
             *
             * @return The next {@link IDatasetSample} in this batch.
             */
            public IDatasetSample next() {
                return data_.get(current_index_++);
            }
        };
    }
}
