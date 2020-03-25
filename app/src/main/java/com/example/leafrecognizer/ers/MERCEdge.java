package com.example.leafrecognizer.ers;

/**
 * Created by Cesare on 18/06/2016.
 */
public class MERCEdge {
    public int a_, b_;
    public double w_, gain_;

    public MERCEdge(int a_, int b_, double w_, double gain_) {
        this.a_ = a_;
        this.b_ = b_;
        this.w_ = w_;
        this.gain_ = gain_;
    }

    public boolean equal(MERCEdge other) {
        return this.gain_ == other.gain_;
    }

    public boolean notEqual(MERCEdge other) {
        return this.gain_ != other.gain_;
    }

    public boolean greaterEq(MERCEdge other) {
        return this.gain_ >= other.gain_;
    }

    public boolean lesserEq(MERCEdge other) {
        return this.gain_ <= other.gain_;
    }

    public boolean greater(MERCEdge other) {
        return this.gain_ > other.gain_;
    }

    public boolean lesser(MERCEdge other) {
        return this.gain_ < other.gain_;
    }

}
