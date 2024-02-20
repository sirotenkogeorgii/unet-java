package main.java.datasets;

import main.java.mathematics.Tensor;

import java.io.IOException;
import java.nio.file.Path;

public class ImageData implements IData {
    private Tensor data_;
    private Path path_;
    public ImageData(Path image_path) throws IOException {
        if (image_path == null) throw new RuntimeException();
        data_ = new Tensor(DatasetFunctions.convertImageTo3DArray(image_path, 256, 256));
        path_ = image_path;
    }
    public Tensor get_data() { return data_; }
    public int[] get_size() { return data_.get_size(); }
    public Path get_path() { return path_; }
}
