# Deep Learning Framework

This is a framework that provides two main features:
* Multidimensional mathematical calculations using the automatic differentiation engine.
* The previous point allows you to easily build deep neural networks without the need for manual differentiation.
  
---

### Automatic Differentiation Engine

The engine allows you to perform mathematical calculations, which can then be differentiated automatically by calling the ```backward()``` method on the final result. The derivatives for each member of the expression are calculated according to the chain rule. During the calculation of an expression, a DAG is built from beginning to end, that is, an acyclic directed graph. This implementation uses a scalar as a vertex of the graph. Derivatives are then calculated from a given vertex in the opposite direction to the graph orientation. See the ```autograd``` module.

### Example

```java
import autograd.Value;

Value x1 = new Value(2);
Value x2 = new Value(4);

Value a = x1.multiply(x2);
Value y1 = a.log();
Value y2 = x1.exp();

Value w = y1.multiply(y2);
w.backward();
```

The graph and its backpropagation of gradients are shown below, with the only difference being that ```exp``` is used instead of ```sin``` in the code above.

![image](./augmented_computational_graph.png)

Similar operations can be done on matrices and tensors, where all operations are performed according to the usual mathematical rules, but the elements of these high-dimensional objects are Values. Thus, the gradient can also flow through matrix (tensor) operations. Object values ​​can be initialized from different distributions. See module ```mathematics```.

---

### Neural networks

The powerful tools above allow you to build deep neural networks for various pattern recognition tasks. For example, you can build a convolutional neural network. In total, this framework contains the following layers:
* 2D Convolution
* 2D Flatten
* 2D Max Pooling
* Linear Layer
  
See ```nn.layers```.

Among the activation functions:
* Relu
* Leaky Relu
* Sigmoid
* Identity
* Softmax (only for linear layer)
  
To make it convenient to work with layers, they can be wrapped in a ```Model```, which builds a layer interaction graph within itself. See module ```nn.models```.

In order for a model to train, it needs an error function. At the moment, the model can be trained for multi-class classification and binary classification tasks. There are two different losses for this: cross entropy and binary cross entropy. See module ```nn.losses```.

Once the loss function is determined, it needs to be optimized. The framework has three policies on how to do this:
* SGD
* Momentum
* Adam
  
See the ```optimizers``` module.

---

### Operation execution modes

In some places in the framework, several implementations of the same operations are written, the only difference being that one is written for a parallel environment, and the other for a sequential one. This can be resolved with a special parameter. See example below.

---

### Full example

In the program ```/src/Main.java``` you can see a complete example of building and training a model.
