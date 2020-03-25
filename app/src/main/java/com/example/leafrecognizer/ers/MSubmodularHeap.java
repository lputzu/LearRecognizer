package com.example.leafrecognizer.ers;

import android.util.Log;

/**
 * Created by Cesare on 08/06/2016.
 */
public class MSubmodularHeap extends MHeap{

    private MERCDisjointSet u_;
    private double[] loop_;
    private double balancingTerm_;

    public MSubmodularHeap(int length) {
        super(length);
    }
    public MSubmodularHeap(Edge[] inarr, int length) {
        super(inarr, length);
    }

    public void EasyPartialUpdateTree(MERCDisjointSet u, double balancingTerm, double[] loop) {
        // access to the disjoint set structure
        u_ = u;
        // keep track the loop value
        loop_ = loop;
        // copy the balancing parameter value.
        balancingTerm_ = balancingTerm;

        // A special heap update structure that utilize the submodular property.
        EasySubmodularMaxHeapifyTree();
    }

    private void EasySubmodularMaxHeapifyTree() {
        //If the root node value is not updated, then update it
        //If the root node value is updated, then it is the maximum value in the current heap.
        //We don't need to update the other nodes because the dimnishing return property guarantees that the value can only be smaller.
        while(EasyUpdateValueTree(1) == 0) {
            // If the edge form a loop, remove it from the loop and update the heap.
            if(this.array_[1].gain_ == 0){
                this.HeapExtractMax();
            }
            else { // Let insert the value into some correct place in the heap.
                this.MaxHeapify(1); // find the maximum one through maxheapify
            }
        }
    }

    private int EasyUpdateValueTree(int i) {
        double erGain, bGain;
        // store the old gain
        double oldGain = this.array_[i].gain_;

        int a, b;
        a = u_.Find(this.array_[i].a_);
        b = u_.Find(this.array_[i].b_);


        // If the edge forms a cycle, makes the gain zero.
        // Later, we will remove the zero edges from the heap.
        if(a == b) {
            this.array_[i].gain_ = 0.0;
        }
        else {
            // recompute the entropy rate gain
            erGain = MERCFunctions.ComputeERGain( this.array_[i].w_, loop_[this.array_[i].a_] -
                    this.array_[i].w_, loop_[this.array_[i].b_] - this.array_[i].w_);

            // recomptue the balancing gain
            bGain = MERCFunctions.ComputeBGain(u_.rNumVertices(), u_.rSize(a), u_.rSize(b));


            // compute the overall gain
            this.array_[i].gain_ = (erGain+balancingTerm_*bGain);
            //array_[i].erGain_ = erGain;
            //array_[i].bGain_ = bGain;
        }

        // If the value is uptodate, we return one. (It will exit the while loop.)
        if(oldGain == this.array_[i].gain_) {
            return 1;
        }

        // If it is not, then we return zero. (It will trigger another MaxHeapify.)
        return 0;
    }

}
