package datasets;

import mathematics.Tensor;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Represents image data as a tensor, facilitating operations and manipulations
 * common in image processing and machine learning contexts. This class implements
 * the {@link IData} interface to standardize how image data is accessed and manipulated.
 */
public class ImageData implements IData {
    private Tensor data_;
    private Path path_;

    /**
     * Constructs an instance of ImageData by loading an image from a specified path
     * and converting it into a tensor representation.
     *
     * @param image_path A {@link Path} object pointing to the image file.
     * @throws IOException If an error occurs during reading the image file.
     * @throws RuntimeException If the provided image_path is null.
     */
    public ImageData(Path image_path) throws IOException {
        if (image_path == null) throw new RuntimeException("Image path cannot be null.");
        data_ = new Tensor(DatasetFunctions.convertImageTo3DArray(image_path, 256, 256));
        path_ = image_path;
    }

    /**
     * Retrieves the tensor representation of the image data.
     *
     * @return A {@link Tensor} object containing the image data.
     */
    public Tensor get_data() {
        return data_;
    }

    /**
     * Retrieves the dimensions of the image data as a tensor.
     *
     * @return An array of integers representing the size of the tensor dimensions.
     */
    public int[] get_size() {
        return data_.get_size();
    }

    /**
     * Retrieves the path to the image file from which this data was loaded.
     *
     * @return A {@link Path} object representing the file path of the image.
     */
    public Path get_path() {
        return path_;
    }
}
