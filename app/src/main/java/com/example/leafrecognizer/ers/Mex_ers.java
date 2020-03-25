package com.example.leafrecognizer.ers;

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Created by Cesare on 07/06/2016.
 */
public class Mex_ers {

    public Mat execute(Mat image, int nC) {
        Mat output = new Mat( image.rows(), image.cols(), CvType.CV_8U );
        double lambda = 0.5, sigma = 5.0;
        int conn8 = 1, kernel = 0;
        MERCLazyGreedy merc = new MERCLazyGreedy();
        MERCDisjointSet result;

        int width = image.cols();
        int height = image.rows();

        MERCInputImage input = new MERCInputImage();


        Mat prova = new Mat(image.cols(), image.rows(), CvType.CV_8U);

        for(int i=0; i<image.rows(); i++) {
            for(int j=0; j<image.cols(); j++) {
                prova.put(j,i,image.get(i,j)[0]);
            }
        }

        // Read the image for segmentation
        input.readImage(prova, conn8);  //  GIUSTO
        //Log.d("Mex_ers", "immagine letta");


        // Entropy rate superpixel segmentation
        result = merc.ClusteringTree(input.get_nNodes(), input, kernel, sigma, lambda*1.0*nC, nC); // GIUSTO
        //Log.d("Mex_ers", "clustering tree creato");


        int[] label = MERCOutputImage.DisjointSetToLabel(result);

        //Log.d("Mex_ers", "label creato     lenght: " + label.length);


        for (int row = 0; row < height; row++) {
            for(int col = 0; col < width; col++) {
                output.put(row, col, label[col + row*width]);
            }
        }

        //Log.d("Mex_ers", "fine processo");

        return output;
    }

}
