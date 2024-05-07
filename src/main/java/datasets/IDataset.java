package datasets;

import java.nio.file.Path;

/**
 * Interface defining the operations for managing a dataset.
 * This interface facilitates access to the dataset's underlying data and label storage paths,
 * as well as methods for querying dataset size and retrieving individual samples.
 * It is intended for use in contexts where datasets consist of multiple samples,
 * each possibly associated with a label.
 */
public interface IDataset {

    /**
     * Retrieves the file system path where the dataset's data files are stored.
     * This method returns a {@link Path} that points to the location on the file system
     * containing the data components of the dataset.
     *
     * @return A {@link Path} object representing the location of the data files for the dataset.
     */
    public Path get_data_path();

    /**
     * Retrieves the file system path where the dataset's label files are stored.
     * This method returns a {@link Path} that points to the location on the file system
     * containing the label components of the dataset, if labels are used.
     *
     * @return A {@link Path} object representing the location of the label files for the dataset.
     */
    public Path get_label_path();

    /**
     * Retrieves the total number of samples within the dataset.
     * This method provides the size of the dataset in terms of the number of samples available,
     * which is crucial for iterating over the dataset or indexing into it.
     *
     * @return An integer representing the total number of samples in the dataset.
     */
    public int get_size();

    /**
     * Retrieves a specific sample from the dataset by its index.
     * This method allows for accessing individual samples, typically returning a data structure
     * that includes both the data and label components associated with that sample.
     *
     * @param index The index of the sample to retrieve, where index should be within the range
     *              from 0 to {@code get_size() - 1}.
     * @return An {@link IDatasetSample} representing the dataset sample at the specified index.
     */
    public IDatasetSample get_sample(int index);
}
