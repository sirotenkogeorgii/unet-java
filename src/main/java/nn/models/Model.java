package main.java.nn.models;

import main.java.mathematics.MultiDimObject;

import java.util.ArrayList;

public abstract class Model {
    protected ModelSettings.executionMode mode_ = ModelSettings.executionMode.SERIAL;
    public ModelSettings.executionMode get_execution_mode() { return mode_; }
//    protected ArrayList<ILayer> layers_;
    public abstract MultiDimObject forward(MultiDimObject input);
    public abstract ArrayList<MultiDimObject> get_parameters();
}
