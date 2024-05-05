package main.java.autograd;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.HashSet;

public class Value extends Differentiable {
    private ArrayList<Value> parents_;
    public Value(double value_) {
        value = value_;
        gradient = 0;
        parents_ = new ArrayList<>();
        prop_func_ = () -> {};
        requires_grad = true;
    }

    public Value(double value_, boolean requires_grad_) {
        value = value_;
        gradient = 0;
        parents_ = new ArrayList<>();
        prop_func_ = () -> {};
        requires_grad = requires_grad_;
    }

    public static Value add(ArrayList<Value> values) {
        if (values == null) throw new NullPointerException("Attempt to sum null array");

        Value new_value = new Value(0);
        for (Value current_value: values) {
            new_value.value += current_value.value;
            if (current_value.requires_grad) new_value.parents_.add(current_value);
        }
        new_value.prop_func_ = () -> {
            for (var current_parent: new_value.parents_) {
                current_parent.gradient += new_value.gradient;
                //current_parent.gradient = Math.max(-gradient_clip_value, Math.min(current_parent.gradient, gradient_clip_value));
            }
        };
        if (new_value.parents_.isEmpty()) new_value.requires_grad = false;
        return new_value;
    }

    public Value add(Value other) {
        if (other == null) throw new NullPointerException("Attempt to add null value");
        var new_value = new Value(value + other.value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient;
            other.gradient += new_value.gradient;

            //gradient = Math.max(-gradient_clip_value, Math.min(gradient, gradient_clip_value));
            //other.gradient = Math.max(-gradient_clip_value, Math.min(other.gradient, gradient_clip_value));
        };
        if (requires_grad) new_value.parents_.add(this);
        if (other.requires_grad) new_value.parents_.add(other);
        if (new_value.parents_.isEmpty()) new_value.requires_grad = false;
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
        var new_value = new Value(value * other.value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * other.value;
            other.gradient += new_value.gradient * value;
            //gradient = Math.max(-gradient_clip_value, Math.min(gradient, gradient_clip_value));
            //other.gradient = Math.max(-gradient_clip_value, Math.min(other.gradient, gradient_clip_value));
        };
        if (requires_grad) new_value.parents_.add(this);
        if (other.requires_grad) new_value.parents_.add(other);
        if (new_value.parents_.isEmpty()) new_value.requires_grad = false;
        return new_value;
    }

    public Value multiply(double constant) { return multiply(new Value(constant)); }

    public Value relu() {
        var new_value = new Value(value < 0 ? 0 : value);
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * (new_value.value > 0 ? 1 : 0); //(value_ > 0 ? 1 : 0);
            //gradient = Math.max(-gradient_clip_value, Math.min(gradient, gradient_clip_value));
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    public Value log() {
        var new_value = value == 0 ? new Value(Math.log(value + 1e-15)) : new Value(Math.log(value));
        new_value.prop_func_ = () -> {
            gradient += value == 0 ? new_value.gradient * (1 / (value + 1e-15)) : new_value.gradient * (1 / value);
            //gradient = Math.max(-gradient_clip_value, Math.min(gradient, gradient_clip_value));
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    public Value sigmoid() {
        var new_value = new Value(1 / (1 + Math.exp(-value)));
        new_value.prop_func_ = () -> {
            gradient += new_value.gradient * new_value.value * (1 - new_value.value);
            //gradient = Math.max(-gradient_clip_value, Math.min(gradient, gradient_clip_value));
        };
        if (requires_grad) new_value.parents_.add(this);
        new_value.requires_grad = requires_grad;
        return new_value;
    }

    public void backward() {
        ArrayList<Value> topological_order = Value.get_topological_order(this);

        gradient = 1;
        for (var value: topological_order)
            value.prop_func_.run();
    }

    protected static ArrayList<Value> get_topological_order(Value value) {
        var topological_order = new ArrayList<Value>();
        Queue<Value> queue = new LinkedList<>();
        Set<Value> inQueue = new HashSet<>();

        queue.offer(value);
        inQueue.add(value);

        while (!queue.isEmpty()) {
            var current_value = queue.poll();
            topological_order.add(current_value);
            inQueue.remove(current_value);

            for (var parent: current_value.parents_) {
                if (!inQueue.contains(parent)) {
                    queue.add(parent);
                    inQueue.add(parent);
                }
            }
        }

        return topological_order;
    }
}
