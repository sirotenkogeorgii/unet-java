package main.java.datasets;

import java.nio.file.Path;

public interface IDatasetSample {
    public IData get_data();
    public IData get_label();
    public Path get_data_path();
    public Path get_label_path();
}
