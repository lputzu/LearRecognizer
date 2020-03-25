package com.example.leafrecognizer.ers;

/**
 * Created by Cesare on 07/06/2016.
 */
public class MERCFunctions {

    public static double ComputeERGain(double wij, double ci, double cj) {
        double er = ((wij+ci)*Math.log(wij+ci) + (wij+cj)*Math.log(wij+cj) - ci*Math.log(ci)  -
                    cj*Math.log(cj) - 2*wij*Math.log(wij)) / Math.log(2.0);
        if( er!=er )
            return 0;
        else
            return er;
    }

    public static double ComputeBGain(int nVertices, int si, int sj) {
        double Si = si*1.0/nVertices;
        double Sj = sj*1.0/nVertices;
        double b = (-(Si+Sj)*Math.log(Si+Sj) + Si*Math.log(Si) + Sj*Math.log(Sj)) / Math.log(2.0) + 1.0;
        return b;
    }

    public static void NormalizeEdgeWeight(MERCInputImage edges, double[] loop, double wT) {
        int nEdges = edges.get_nEdges();
        int nVertices = edges.get_nNodes();


        for(int i = 0; i<nEdges; i++) {
             edges.edges_[i].setW_(edges.edges_[i].getW_() / wT);
        }

        for(int i = 0; i<nVertices; i++) {
            loop[i] = loop[i] / wT;
        }
    }

    public static double ComputeTotalWeight(double[] loop, int nVertices) {
        double wT = 0;
        for(int i=0; i<nVertices; i++) {
            wT += loop[i];
        }

        return wT;
    }

    public static double[] ComputeLoopWeight(int nVertices, MERCInputImage edges) {
        int nEdges = edges.get_nEdges();
        double[] loop = new double [nVertices];

        for(int i=0; i<nVertices; i++) {
            loop[i] = 0;
        }

        for(int i=0; i<nEdges; i++) {
            loop[edges.edges_[i].getA_()] += edges.edges_[i].getW_();
            loop[edges.edges_[i].getB_()] += edges.edges_[i].getW_();
        }

        return loop;
    }

    public static void ComputeSimilarity(MERCInputImage edges, double sigma, int kernel) {
        switch(kernel) {
            case 0:
                ComputeSimilarityGaussian(edges, sigma);
                break;
            //case 1:
            //	ComputeSimilarityLaplacian(edges,sigma);
            //	break;
            //case 2:
            //	ComputeSimilarityCauchy(edges,sigma);
            //	break;
            //case 3:
            //	ComputeSimilarityRationalQuadratic(edges,sigma);
            //	break;
            //case 4:
            //	ComputeSimilarityInverseMultiquadric(edges,sigma);
            //	break;
        }
    }

    public static void ComputeSimilarityGaussian(MERCInputImage edges, double sigma) {
        int nEdges = edges.get_nEdges();

        double twoSigmaSquare = 2*sigma*sigma;

        for(int i=0; i<nEdges; i++) {
            edges.edges_[i].setW_(Math.exp(-(edges.edges_[i].getW_()*edges.edges_[i].getW_())/twoSigmaSquare));
        }
    }
}
