package com.example.leafrecognizer.ers;

/**
 * Created by Cesare on 07/06/2016.
 */
public class Edge {
    private int a_, b_;
    private double w_, gain_;

    public Edge(int a_, int b_, double w_, double gain_) {
        this.a_ = a_;
        this.b_ = b_;
        this.w_ = w_;
        this.gain_ = gain_;
    }

    public void setA_(int a_) {
        this.a_ = a_;
    }

    public void setB_(int b_) {
        this.b_ = b_;
    }

    public void setW_(double w_) {
        this.w_ = w_;
    }

    public void setGain_(double gain_) {
        this.gain_ = gain_;
    }

    public int getA_() {
        return this.a_;
    }

    public int getB_() {
        return this.b_;
    }

    public double getW_() {
        return this.w_;
    }

    public double getGain_() {
        return this.gain_;
    }

    public boolean equal(Edge other) {
        return this.w_ == other.w_;
    }

    public boolean notEqual(Edge other) {
        return this.w_ != other.w_;
    }

    public boolean greaterEq(Edge other) {
        return this.w_ >= other.w_;
    }

    public boolean lesserEq(Edge other) {
        return this.w_ <= other.w_;
    }

    public boolean greater(Edge other) {
        return this.w_ > other.w_;
    }

    public boolean lesser(Edge other) {
        return this.w_ < other.w_;
    }
}
