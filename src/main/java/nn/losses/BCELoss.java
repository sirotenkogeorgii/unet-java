package nn.losses;

import autograd.Value;
import mathematics.Matrix;
import mathematics.MultiDimObject;
import nn.layers.LayerFunctions;

/**
 * Implements Binary Cross-Entropy (BCE) Loss, which is typically used for binary classification problems.
 * This loss function measures the performance of a classification model whose output is a probability value between 0 and 1.
 */
public class BCELoss extends CrossEntropyLoss {

    /**
     * Calculates the binary cross-entropy loss between the predicted outputs and actual targets.
     *
     * @param input The predicted outputs from the model as a {@link Matrix}, typically representing probabilities.
     * @param target The actual target outputs as a {@link Matrix}, typically representing binary labels.
     * @return A {@link Value} representing the computed binary cross-entropy loss.
     */
    @Override
    public Value calculate_loss(MultiDimObject input, MultiDimObject target) {
        return LayerFunctions.bce_loss((Matrix)input, (Matrix)target);
    }
}
