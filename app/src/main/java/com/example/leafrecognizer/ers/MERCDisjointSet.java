package com.example.leafrecognizer.ers;

import android.util.Log;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Cesare on 07/06/2016.
 */
public class MERCDisjointSet {
    public int[] p_;
    public int[] size_;
    private List<List<Integer>> lists_;
    int nElements_;
    int nVertices_;

    public MERCDisjointSet(int nElements) {
        nElements_ = nElements;
        nVertices_ = nElements_;
        p_ = new int[nElements_];
        size_ = new int[nElements_];
        lists_ = new ArrayList<List<Integer>>();

        List<Integer> app;

        // Initialization with the cluster size and id
        for (int i = 0; i < nElements; i++) {
            p_[i] = i;
            size_[i] = 1;
            app = new ArrayList<Integer>();
            app.add(i);
            lists_.add(app);
        }
    }

    public void Set(int x, int l) {
        p_[x] = l;
    }

    public int Find(int x) {
        // return the cluster ID
        return p_[x];
    }

    public int Join(int a, int b) {
        int aID = Find(a);
        int bID = Find(b);

        // The size is only maintained for cluster ID.
        int aSize = size_[aID];
        int bSize = size_[bID];

        int newID, delID;

        if( bSize < aSize ) {
            newID = aID;
            delID = bID;
        }
        else {
            newID = bID;
            delID = aID;
        }

        size_[newID] = aSize + bSize;
        size_[delID] = 0;

        for(int i = 0; i < lists_.get(delID).size(); i++) {
            p_[lists_.get(delID).get(i)] = newID;
        }

        // traduzione della riga lists_[newID].Append(lists_[delID]);
        lists_.get(newID).addAll(lists_.get(delID));
        lists_.get(delID).clear();

        nElements_--;
        return newID;
    }

    // return the cluster size containing the vertex x
    public int rSize(int x) {
        return size_[this.Find(x)];
    }

    // return the number of connected components in the set
    public int rNumSets() {
        return nElements_;
    }

    // return the total number of vertices in the set
    public int rNumVertices() {
        return nVertices_;
    }

}
