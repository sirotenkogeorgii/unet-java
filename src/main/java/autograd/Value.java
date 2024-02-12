package main.java.autograd;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

public class Value implements IDifferentiable {
    private double value_;
    private double gradient_;
    private ArrayList<Value> parents_;
    private Runnable prop_func_;

    public Value(double value) {
        value_ = value;
        gradient_ = 0;
        parents_ = new ArrayList<>();
        prop_func_ = () -> {};
    }

    public Value add(Value other) {
        if (other == null) throw new NullPointerException("Attempt to add null value");
        var new_value = new Value(value_ + other.value_);
        new_value.prop_func_ = () -> {
            gradient_ += new_value.gradient_;
            other.gradient_ += new_value.gradient_;
        };
        new_value.parents_.add(this); new_value.parents_.add(other);
        return new_value;
    }

    public Value add(double constant) { return add(new Value(constant)); }

    public Value sub(Value other) {
        if (other == null) throw new NullPointerException("Attempt to subtract  null value");
        return add(other.multiply(-1));
    }

    public Value sub(double constant) { return sub( new Value(constant)); }

    public Value multiply(Value other) {
        if (other == null) throw new NullPointerException("Attempt to multiply by null value");
        var new_value = new Value(value_ * other.value_);
        new_value.prop_func_ = () -> {
            gradient_ += new_value.gradient_ * other.value_;
            other.gradient_ += new_value.gradient_ * value_;
        };
        new_value.parents_.add(this); new_value.parents_.add(other);
        return new_value;
    }

    public Value multiply(double constant) { return multiply(new Value(constant)); }

    public Value relu() {
        var new_value = new Value(value_ < 0 ? 0 : value_);
        new_value.prop_func_ = () -> {
            gradient_ += new_value.gradient_ * (new_value.value_ > 0 ? 1 : 0);
        };
        new_value.parents_.add(this);
        return new_value;
    }

    public Value sigmoid() {
        var new_value = new Value(1 / (1 + Math.exp(-value_)));
        new_value.prop_func_ = () -> {
            gradient_ += new_value.gradient_ * new_value.value_ * (1 - new_value.value_);
        };
        new_value.parents_.add(this);
        return new_value;
    }

    public double get_gradient() { return gradient_; }
    public double get_value() { return value_; }
    public void set_value(double value) { value_ = value; }
    public void set_gradient(double gradient) { gradient_ = gradient; }

    public void backward() {
        ArrayList<Value> topological_order = get_topological_order();
        gradient_ = 1;
        for (var value: topological_order)
            value.prop_func_.run();
    }

    private ArrayList<Value> get_topological_order() {
        var topological_order = new ArrayList<Value>();
        Queue<Value> queue = new LinkedList<>();
        queue.offer(this);
        
        while (!queue.isEmpty()) {
            var current_value = queue.poll();
            topological_order.add(current_value);
            for (var parent: current_value.parents_) {
                if (!queue.contains(parent))
                    queue.add(parent);
            }
        }

        return topological_order;
    }
}

class Program {
    public static void main(String[] args) {
        var x = new Value(5);
        var y = x.multiply(x);
        y.backward();

        System.out.printf("[DEBUG] result gradient: %f\n", x.get_gradient());
    }
}