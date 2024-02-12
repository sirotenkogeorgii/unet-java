package main.java.autograd;

interface IDifferentiable {
    public void backward();
    public double get_gradient();
    public void set_gradient(double gradient);
    public double get_value();
    public void set_value(double value);
}