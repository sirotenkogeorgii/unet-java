package main.java.nn.losses;

import main.java.autograd.Value;
import main.java.mathematics.IMultiDimObject;
import main.java.mathematics.Matrix;
import main.java.nn.layers.LayerFunctions;

public class BCELoss extends Loss {

    public BCELoss() { loss_value_ = new Value(0); }
    @Override
    public void calculate_loss(IMultiDimObject input, IMultiDimObject target) {
        var input_matrix = (Matrix)input;
        var target_matrix = (Matrix)target;
        loss_value_ = LayerFunctions.bce_loss(input_matrix.get(0, 0), target_matrix.get(0, 0));
        System.out.printf("Loss: %f\n", loss_value_.get_value());
    }

    @Override
    public void backward() {
        loss_value_.backward();
    }
}
