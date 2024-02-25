package main.java.nn.losses;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.nn.layers.LayerFunctions;

public class BCELoss extends Loss {

    public BCELoss() { loss_value_ = new Value(0); }
    @Override
    public void calculate_loss(IMultiDimObject input, IMultiDimObject target) {
        loss_value_ = LayerFunctions.bce_loss((Matrix)input, (Matrix)target);
    }

    @Override
    public void backward() {
        loss_value_.backward();
    }
}
