package main.java.datasets;

import main.java.mathematics.Matrix;
import main.java.datasets.DatasetFunctions;
import main.java.mathematics.Tensor;

import java.io.IOException;
import java.nio.file.Path;

public class SegmentationSample implements IDatasetSample {
    private IData data_;
    private IData mask_;
    private Path data_path_;
    private Path mask_path_;
    public SegmentationSample(Path data_path, Path mask_path) throws IOException {
        if (data_path == null || mask_path == null)
            throw new RuntimeException();
        data_path_ = data_path;
        mask_path_ = mask_path;
        data_ = new ImageData(data_path);
        mask_ = new ImageData(mask_path);
    }
    public IData get_data() { return data_; }
    public IData get_label() { return mask_; }
    public Path get_data_path() { return data_path_; }
    public Path get_label_path() { return mask_path_; }
}
