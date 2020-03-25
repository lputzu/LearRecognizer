package com.example.leafrecognizer.ers;

import android.util.Log;

/**
 * Created by Cesare on 08/06/2016.
 */
public class MHeap {

    protected int maxHeap_;
    protected int length_;		// size of the array
    protected int nElements_;   // # of elements in the heap
    MERCEdge[] array_;			// the container

    public MHeap(){

    }

    public MHeap(int length) {
        maxHeap_ = 1; // max heap by default
        nElements_ = 0;
        length_ = length;
        array_ = new MERCEdge[length_+1];
    }

    public MHeap(Edge[] inarr, int length) {
        maxHeap_ = 1; // max heap by default
        nElements_ = 0;
        length_ = length;
        array_ = new MERCEdge[length_+1];
        MERCEdge app;
        for(int i=0; i<length_; i++) {
            app = new MERCEdge(inarr[i].getA_(), inarr[i].getB_(), inarr[i].getW_(), inarr[i].getGain_());
            array_[i+1] = app;
        }
    }

    public MERCEdge HeapExtractMax() {
        if( HeapSize() < 1) {
            Log.d("HEAP", "Heap underflow error");
        }

        MERCEdge maxElem = array_[1];
        array_[1] = array_[HeapSize()];
        nElements_--;
        MaxHeapify(1);

        return maxElem;
    }

    public boolean IsEmpty() {
        return (nElements_ == 0);
    }

    public int Left(int i) {
        return (i<<1);
    }

    public int Right(int i) {
        return (i<<1)+1;
    }

    public int Parent(int i) {
        return (i>>1);
    }

    public int HeapSize() {
        return nElements_;
    }

    public void BuildMaxHeap() {
        maxHeap_ = 1;
        nElements_ = length_;
        int hLength = length_/2;
        for(int i = hLength; i>=1 ; i--)
            MaxHeapify(i);
    }

    public void MaxHeapify(int i) {
        int left, right, largest;

        left = Left(i);
        right = Right(i);

        if((left <= HeapSize()) && array_[left].greater(array_[i]))
            largest = left;
        else
            largest = i;

        if((right <= HeapSize()) && array_[right].greater(array_[largest]))
            largest = right;

        if(largest != i) {
            MERCEdge tmp;
            tmp = array_[i];
            array_[i] = array_[largest];
            array_[largest] = tmp;
            MaxHeapify(largest);
        }
    }
}
