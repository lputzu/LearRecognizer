package com.example.leafrecognizer.ers;

import android.util.Log;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Cesare on 08/06/2016.
 */
public class MERCOutputImage {

    public static int[] DisjointSetToLabel(MERCDisjointSet u) {
        int nSegments = 0;
        int segIndex = 0;
        int nVertices = u.rNumVertices();
        List<List<Integer>> sarray = new ArrayList<List<Integer>>();
        int[] labeling = new int[nVertices];
        List<Integer> app;

        // inizializzazione sarray
        for(int k=0; k<nVertices; k++) {
            app = new ArrayList<Integer>();
            sarray.add(app);
        }

        int comp;

        for (int k = 0; k<nVertices; k++) {
            comp = u.Find(k);
            sarray.get(comp).add(k);
        }

        for(int k = 0; k<nVertices; k++) {
            if(sarray.get(k).size() > 0) {
                nSegments++;
            }
        }

        for(int k = 0; k<nVertices; k++) {
            if(sarray.get(k).size() > 0) {
                for(int j = 0; j<sarray.get(k).size(); j++) {
                    labeling[sarray.get(k).get(j)] = segIndex;
                }
                segIndex++;
            }
        }

        return labeling;
    }
}