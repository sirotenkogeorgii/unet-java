package main.java.datasets;

import java.nio.file.Path;

/**
 * Interface defining the operations for accessing a single sample in a dataset.
 * This interface ensures that any dataset sample can provide access to both its data and labels,
 * as well as the paths to where this information is stored.
 */
public interface IDatasetSample {

    /**
     * Retrieves the data portion of the dataset sample.
     * This method returns an instance of {@link IData} that represents the data component
     * of a sample.
     *
     * @return An {@link IData} instance representing the data of the sample.
     */
    IData get_data();

    /**
     * Retrieves the label portion of the dataset sample.
     * This method returns an instance of {@link IData} that represents the label associated
     * with the data of a sample.
     *
     * @return An {@link IData} instance representing the label of the sample.
     */
    IData get_label();

    /**
     * Retrieves the file system path to the data of the sample.
     * This method returns a {@link Path} object pointing to the location of the sample's data
     * on the file system.
     *
     * @return A {@link Path} object representing the location of the data file for the sample.
     */
    Path get_data_path();

    /**
     * Retrieves the file system path to the label of the sample.
     * This method returns a {@link Path} object pointing to the location of the sample's label
     * on the file system.
     *
     * @return A {@link Path} object representing the location of the label file for the sample.
     */
    Path get_label_path();
}
