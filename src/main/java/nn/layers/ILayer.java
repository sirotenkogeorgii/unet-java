package main.java.nn.layers;

import main.java.mathematics.MultiDimObject;
import main.java.nn.models.ModelSettings;

import java.util.ArrayList;

public interface ILayer {
//    public String name;
    public MultiDimObject forward(MultiDimObject inputs);
    public ArrayList<MultiDimObject> get_parameters();
    public void set_execution_mode(ModelSettings.executionMode mode);
}
