package com.example.leafrecognizer.ers;

import android.util.Log;

/**
 * Created by Cesare on 07/06/2016.
 */
public class MERCLazyGreedy {

    public MERCDisjointSet ClusteringTree(int nVertices, MERCInputImage edges, int kernel,
                                          double sigma, double lambda, int nC) {

        int nEdges = edges.get_nEdges();
        MERCDisjointSet u = new MERCDisjointSet(nVertices); // GIUSTO
        MERCFunctions.ComputeSimilarity(edges, sigma, kernel); // GIUSTO
        double[] loop = MERCFunctions.ComputeLoopWeight(nVertices, edges); // GIUSTO
        double wT = MERCFunctions.ComputeTotalWeight(loop, nVertices); // GIUSTO
        MERCFunctions.NormalizeEdgeWeight(edges, loop, wT);  // GIUSTO


        double[] erGainArr = new double [nEdges];   // gain in entropy rate term
        double[] bGainArr = new double [nEdges];	// gain in balancing term
        double maxERGain = 0, maxBGain = 1e-20;

        for(int i=0; i<nEdges; i++) {  // GIUSTO
            erGainArr[i] = MERCFunctions.ComputeERGain(
                edges.edges_[i].getW_(),
                loop[edges.edges_[i].getA_()] - edges.edges_[i].getW_(),
                loop[edges.edges_[i].getB_()] - edges.edges_[i].getW_());

            int a = u.Find(edges.edges_[i].getA_());
            int b = u.Find(edges.edges_[i].getB_());

            if(a!=b) {
                bGainArr[i] = MERCFunctions.ComputeBGain(nVertices, u.rSize(a), u.rSize(b));
            }
            else {
                bGainArr[i] = 0;
            }

            if(erGainArr[i] > maxERGain) {
                maxERGain = erGainArr[i];
            }

            if(bGainArr[i] > maxBGain) {
                maxBGain = bGainArr[i];
            }
        }

        double balancing = lambda*maxERGain/Math.abs(maxBGain); // GIUSTO

        for(int i=0; i<nEdges; i++) { // GIUSTO
            edges.edges_[i].setGain_(erGainArr[i]+balancing*bGainArr[i]);
        }

        // Heap
        MSubmodularHeap heap = new MSubmodularHeap(edges.edges_, nEdges); // GIUSTO
        heap.BuildMaxHeap(); // GIUSTO

        MERCEdge bestEdge;
        int cc = nVertices;
        int a, b;

        while( cc > nC ) {
            if(heap.IsEmpty()) {
                Log.d("HEAP", "Empty");
                return u;
            }
            // find the best edge to add
            bestEdge = heap.HeapExtractMax(); // GIUSTO

            // insert the edge into the graph  // GIUSTO
            a = u.Find(bestEdge.a_);
            b = u.Find(bestEdge.b_);

            if(a!=b) {
                u.Join(a,b); // GIUSTO
                cc--;
            }

            heap.EasyPartialUpdateTree(u, balancing, loop);
        }

        return u;
    }
}
