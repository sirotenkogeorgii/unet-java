package main.java.nn.models;
import main.java.mathematics.Matrix;

import java.util.ArrayList;

public interface IModel {
    public Matrix forward(Matrix input);
    public ArrayList<Matrix> get_parameters();
}
