package main.java.datasets;

import main.java.mathematics.Tensor;

public interface IData {
    public Tensor get_data();
    public int[] get_size();
}
