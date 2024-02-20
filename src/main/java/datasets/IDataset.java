package main.java.datasets;

import java.nio.file.Path;

public interface IDataset {
    public Path get_data_path();
    public Path get_label_path();
    public int get_size();
    public IDatasetSample get_sample(int index);
//    public
//    public void transform()
}
