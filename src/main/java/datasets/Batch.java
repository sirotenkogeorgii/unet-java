package main.java.datasets;

import java.util.ArrayList;
import java.util.Iterator;

public class Batch implements Iterable<IDatasetSample> {
    private ArrayList<IDatasetSample> data_;
    private int size_;
    public Batch(ArrayList<IDatasetSample> data) {
        if (data == null) throw new RuntimeException();
        data_ = data;
        size_ = data_.size();
    }
    public int get_size() { return size_; }
    public Iterator<IDatasetSample> iterator() {
        return new Iterator<>() {
            private int current_index_ = 0;
            public boolean hasNext() { return current_index_ < size_; }
            public IDatasetSample next() { return data_.get(current_index_++); }
        };
    }
}
