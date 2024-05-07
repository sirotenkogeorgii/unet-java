package nn.models;

/**
 * Contains settings that control the execution and behavior of neural network models.
 * This class provides a central place to manage settings that are used across different
 * components of a neural network.
 */
public class ModelSettings {

    /**
     * Enumerates the modes of execution for operations within neural network models.
     * Possible modes: [SERIAL, PARALLEL].
     */
    public static enum executionMode { SERIAL, PARALLEL };
}
