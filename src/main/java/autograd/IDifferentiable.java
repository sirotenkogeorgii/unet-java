package main.java.autograd;

interface IDifferentiable {
    static double gradient_clip_value = 10;
    public void backward();
    public double get_gradient();
    public void set_gradient(double gradient);
    public double get_value();
    public void set_value(double value);
}