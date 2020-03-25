package com.example.leafrecognizer.classification;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Toast;

import com.example.leafrecognizer.Test;


public class LeafProcessor {

    String sdpath = Environment.getExternalStorageDirectory().getPath();
    String app_path = sdpath + File.separator + "LeafRecog";

    private static final int numOfFeatures = 51;
    private static final double EPSILON = 0.000000000001;

	public Mat segmentLeafThresh(Mat inputImage, int thresh) {
        Mat grayImage = new Mat(inputImage.rows(), inputImage.cols(), CvType.CV_8UC1, new Scalar(0));
        Mat v = new Mat();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat maskImage = new Mat(inputImage.rows(), inputImage.cols(), CvType.CV_8UC1, new Scalar(0));

        //from RGB to gray
        Imgproc.cvtColor(inputImage , grayImage, Imgproc.COLOR_RGB2GRAY);

	    //binarization (threshold)
	    Imgproc.threshold(grayImage, maskImage, thresh, 255, Imgproc.THRESH_BINARY_INV);

        //smoothing (median filter)
	    Imgproc.medianBlur(maskImage, maskImage, 5);

	    // leave only the bigger connected component with holes filled
	    Imgproc.findContours(maskImage, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

	    int maxAreaIdx = -1;
	    double maxArea = 0;
	    for(int idx = 0; idx < contours.size(); idx++) {
	        Mat contour = contours.get(idx);
	        double contourArea = Imgproc.contourArea(contour);
	        if (contourArea > maxArea) {
	        	maxAreaIdx = idx;
	            maxArea = contourArea;
	        }
	    }

        //copy the bigger connected component, directly filled, on a new black image
        maskImage.setTo(new Scalar(0,0,0));
        Imgproc.drawContours(maskImage, contours, maxAreaIdx, new Scalar(255, 255, 255), -1);

	    // obtain the masked image
        Mat maskedGray = new Mat();
	    grayImage.copyTo(maskedGray, maskImage);

/*
        Bitmap bmp = Bitmap.createBitmap(maskedGray.cols(), maskedGray.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(maskedGray, bmp);

        Test.testSalvaImmagine(app_path, "testSegmentazione.jpg", bmp);
*/


        // segmented gray image
        return maskedGray;
	}

	public Mat extractFeatures(Mat inputImage) {
		Mat leafDataMat = new Mat(numOfFeatures, 1, CvType.CV_32FC1, new Scalar(0, 0, 0)); // Mat of features
	    Point centroid = new Point();
        Mat maskImage = new Mat();
		double[] excInc = new double[2];

	    // compute the center of mass
	    Moments moments = Imgproc.moments(inputImage, true);
	    centroid.x = moments.get_m10() / moments.get_m00();
	    centroid.y = moments.get_m01() / moments.get_m00();

        Mat tmp = new Mat();
        inputImage.copyTo(tmp);

        // compute the contours
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat v = new Mat();
        Imgproc.findContours(tmp, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        int maxAreaIdx = -1;
        double maxArea = 0;
        for(int idx = 0; idx < contours.size(); idx++) {
            Mat contour = contours.get(idx);
            double contourArea = Imgproc.contourArea(contour);
            if (contourArea > maxArea) {
                maxAreaIdx = idx;
                maxArea = contourArea;
            }
        }

        MatOfPoint leafContour = contours.get(maxAreaIdx);

        double[] radii = centroidRadii(leafContour, centroid, excInc);

        double[] shapeFeatures = shapeDescriptor(excInc[0], excInc[1], leafContour, inputImage);

        Mat hu = new Mat();
	    Imgproc.HuMoments(moments, hu);
	    double[] huMoments = new double[hu.rows()];
	    for(int i = 0; i < hu.rows(); i++){
	    	huMoments[i] = hu.get(i, 0)[0];
	    }

	    //double[] textureFeatures = getTextureFeatures(true);

	    //double[] textureFeaturesRidotto = getTextureFeaturesRidotto(true, inputImage);

        //double[] lbpFeatures = LBPInvariante(inputImage);

	    //metto tutto nel Mat

		int j = 0;
		for(int k = 0; k < 36 && j < numOfFeatures; k++, j++){    // A - AJ
			double[] temp = {radii[k]};
			leafDataMat.put(j, 0, temp);
		}



		for(int k = 0; k < 8 && j < numOfFeatures; k++, j++){    // AK - AR
			double[] temp = {shapeFeatures[k]};
			leafDataMat.put(j, 0, temp);
		}


		for(int k = 0; k < 7 && j < numOfFeatures; k++, j++){    // AS - AY
			double[] temp = {huMoments[k]};
			leafDataMat.put(j, 0, temp);
		}


/*		for(int k = 0; k < 20 && j < numOfFeatures; k++, j++){
			//double[] temp = {textureFeatures[k]};
			double[] temp = {textureFeaturesRidotto[k]};
			leafDataMat.put(j, 0, temp);
		}*/

/*        for(int k = 0; k < 36 && j < numOfFeatures; k++, j++) {
            double[] temp = {lbpFeatures[k]};
            leafDataMat.put(j,0,temp);
        }*/
        return leafDataMat;
	}

	public double[] LBPRotationInvariance(Mat inputImage){
        double[] feature = new double[10];
        int label; // number of 1
        int transitionPixel;
        int numOfTransition;
        double currentPixel;

        // draw contorn
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat app = new Mat(inputImage.rows(), inputImage.cols(), CvType.CV_8U);
        Imgproc.findContours(inputImage, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        int maxAreaIdx = -1;
        double maxArea = 0;
        for(int idx = 0; idx < contours.size(); idx++) {
            Mat contour = contours.get(idx);
            double contourArea = Imgproc.contourArea(contour);
            if (contourArea > maxArea) {
                maxAreaIdx = idx;
                maxArea = contourArea;
            }
        }

        Imgproc.drawContours(app, contours, maxAreaIdx, new Scalar(255, 255, 255), -1);

        Mat maskedGray = new Mat();
        inputImage.copyTo(maskedGray, app);

        for(int i=1; i<inputImage.rows() -1; i++) {
            for(int j=1; j<inputImage.cols() -1; j++){

                if(app.get(i,j)[0] == 255) {

                    currentPixel = inputImage.get(i, j)[0];
                    label = 0;
                    numOfTransition = 0;

                    if (currentPixel >= inputImage.get(i - 1, j - 1)[0]) {
                        label++;
                        transitionPixel = 1;
                    } else {
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                        transitionPixel = 1;
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j + 1)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                        transitionPixel = 1;
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i, j + 1)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                        transitionPixel = 1;
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j + 1)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                        transitionPixel = 1;
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                        transitionPixel = 1;
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j - 1)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                        transitionPixel = 1;
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                        transitionPixel = 0;
                    }

                    if (currentPixel >= inputImage.get(i, j - 1)[0]) {
                        label++;
                        if (transitionPixel == 0) {
                            numOfTransition++;
                        }
                    } else {
                        if (transitionPixel == 1) {
                            numOfTransition++;
                        }
                    }

                    if (numOfTransition > 2) {
                        feature[9]++;
                    } else {
                        feature[label]++;
                    }
                }
            }
        }

        return feature;
    }

	public double[] LBPprova(Mat inputImage){
        double[] feature = new double[10];
        int[] check = new int[8];
        int label; // number of 1
        int transitionPixel;
        int numOfTransition;
        double currentPixel;

        // draw contorn
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat app = new Mat(inputImage.rows(), inputImage.cols(), CvType.CV_8U);
        Imgproc.findContours(inputImage, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        int maxAreaIdx = -1;
        double maxArea = 0;
        for(int idx = 0; idx < contours.size(); idx++) {
            Mat contour = contours.get(idx);
            double contourArea = Imgproc.contourArea(contour);
            if (contourArea > maxArea) {
                maxAreaIdx = idx;
                maxArea = contourArea;
            }
        }

        Imgproc.drawContours(app, contours, maxAreaIdx, new Scalar(255, 255, 255), -1);

        Mat maskedGray = new Mat();
        inputImage.copyTo(maskedGray, app);


        for(int i=1; i<inputImage.rows() -1; i++) {
            for(int j=1; j<inputImage.cols() -1; j++){

                if(app.get(i,j)[0] == 255) {

                    currentPixel = inputImage.get(i, j)[0];
                    label = 0;
                    numOfTransition = 0;

                    for(int h=0; h<check.length; h++) {
                        check[h] = 0;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j - 1)[0]) {
                        check[0] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j)[0]) {
                        check[1] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j + 1)[0]) {
                        check[2] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i, j + 1)[0]) {
                        check[3] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j + 1)[0]) {
                        check[4] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j)[0]) {
                        check[5] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j - 1)[0]) {
                        check[6] = 1;
                        label++;
                    }

                    if (currentPixel >= inputImage.get(i, j - 1)[0]) {
                        check[7] = 1;
                        label++;
                    }

                    int a = check[0], b;

                    for(int h=1; h<check.length; h++){
                        b=check[h];
                        if(a!=b){
                            numOfTransition++;
                            a=b;
                        }
                    }

                    if (numOfTransition > 2) {
                        feature[9]++;
                    } else {
                        feature[label]++;
                    }
                }
            }
        }
/*
        double[] featureClear = new double[5];

        for(int i=0; i<featureClear.length; i++){
            featureClear[i] = feature[i+5];
        }

        return featureClear;
        */

        return feature;
    }

    public double[] LBPInvariante(Mat inputImage){
        double[] feature = new double[36];
        int[] check = new int[8];
        int temp;
        int numOfOne; // number of 1
        double currentPixel;

        Mat resultMat2, resultMat3, resultMat4, resultMat5, resultMat6;
        int[] label2 = {2,9,10,11};
        int[] label3 = {3,12,13,14,15,16,17};
        int[] label4 = {4,18,19,20,21,22,23,24,25,26};
        int[] label5 = {5,27,28,29,30,31,32};
        int[] label6 = {6,33,34,35};

        List<int[]> vectorlist2 = new ArrayList<int[]>();
        List<int[]> vectorlist3 = new ArrayList<int[]>();
        List<int[]> vectorlist4 = new ArrayList<int[]>();
        List<int[]> vectorlist5 = new ArrayList<int[]>();
        List<int[]> vectorlist6 = new ArrayList<int[]>();
        vectorlist2.add(new int[] {1,1,0,0,0,0,0,0});   // 11000000
        vectorlist2.add(new int[] {1,0,1,0,0,0,0,0});   // 10100000
        vectorlist2.add(new int[] {1,0,0,1,0,0,0,0});   // 10010000
        vectorlist2.add(new int[] {1,0,0,0,1,0,0,0});   // 10001000

        vectorlist3.add(new int[] {1,1,1,0,0,0,0,0});   // 11100000
        vectorlist3.add(new int[] {1,1,0,1,0,0,0,0});   // 11010000
        vectorlist3.add(new int[] {1,0,1,1,0,0,0,0});   // 10110000
        vectorlist3.add(new int[] {1,1,0,0,1,0,0,0});   // 11001000
        vectorlist3.add(new int[] {1,0,1,0,1,0,0,0});   // 10101000
        vectorlist3.add(new int[] {1,0,0,1,1,0,0,0});   // 10011000
        vectorlist3.add(new int[] {1,0,1,0,0,1,0,0});   // 10100100

        vectorlist4.add(new int[] {1,1,1,1,0,0,0,0});   // 11110000
        vectorlist4.add(new int[] {1,1,1,0,1,0,0,0});   // 11101000
        vectorlist4.add(new int[] {1,1,0,1,1,0,0,0});   // 11011000
        vectorlist4.add(new int[] {1,0,1,1,1,0,0,0});   // 10111000
        vectorlist4.add(new int[] {1,1,1,0,0,1,0,0});   // 11100100
        vectorlist4.add(new int[] {1,1,0,1,0,1,0,0});   // 11010100
        vectorlist4.add(new int[] {1,0,1,1,0,1,0,0});   // 10110100
        vectorlist4.add(new int[] {1,1,0,0,1,1,0,0});   // 11001100
        vectorlist4.add(new int[] {1,0,1,0,1,1,0,0});   // 10101100
        vectorlist4.add(new int[] {1,0,1,0,1,0,1,0});   // 10101010

        vectorlist5.add(new int[] {1,1,1,1,1,0,0,0});   // 11111000
        vectorlist5.add(new int[] {1,1,1,1,0,1,0,0});   // 11110100
        vectorlist5.add(new int[] {1,1,1,0,1,1,0,0});   // 11101100
        vectorlist5.add(new int[] {1,1,0,1,1,1,0,0});   // 11011100
        vectorlist5.add(new int[] {1,0,1,1,1,1,0,0});   // 10111100
        vectorlist5.add(new int[] {1,1,1,0,1,0,1,0});   // 11101010
        vectorlist5.add(new int[] {1,1,0,1,1,0,1,0});   // 11011010

        vectorlist6.add(new int[] {1,1,1,1,1,1,0,0});   // 11111100
        vectorlist6.add(new int[] {1,1,1,1,1,0,1,0});   // 11111010
        vectorlist6.add(new int[] {1,1,1,1,0,1,1,0});   // 11110110
        vectorlist6.add(new int[] {1,1,1,0,1,1,1,0});   // 11101110

        resultMat2 = createList(vectorlist2, label2);
        resultMat3 = createList(vectorlist3, label3);
        resultMat4 = createList(vectorlist4, label4);
        resultMat5 = createList(vectorlist5, label5);
        resultMat6 = createList(vectorlist6, label6);

        for(int i=1; i<inputImage.rows() -1; i++) {
            for(int j=1; j<inputImage.cols() -1; j++){
                if(inputImage.get(i,j)[0] != 0) {

                    currentPixel = inputImage.get(i, j)[0];
                    numOfOne = 0;

                    for(int h=0; h<check.length; h++) {
                        check[h] = 0;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j - 1)[0]) {
                        check[0] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j)[0]) {
                        check[1] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i - 1, j + 1)[0]) {
                        check[2] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i, j + 1)[0]) {
                        check[3] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j + 1)[0]) {
                        check[4] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j)[0]) {
                        check[5] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i + 1, j - 1)[0]) {
                        check[6] = 1;
                        numOfOne++;
                    }

                    if (currentPixel >= inputImage.get(i, j - 1)[0]) {
                        check[7] = 1;
                        numOfOne++;
                    }

                    switch (numOfOne){
                        case 2:
                            feature[Find(resultMat2, convertIntoNumber(check))]++;
                            break;
                        case 3:
                            feature[Find(resultMat3, convertIntoNumber(check))]++;
                            break;
                        case 4:
                            feature[Find(resultMat4, convertIntoNumber(check))]++;
                            break;
                        case 5:
                            feature[Find(resultMat5, convertIntoNumber(check))]++;
                            break;
                        case 6:
                            feature[Find(resultMat6, convertIntoNumber(check))]++;
                            break;
                        default:
                            feature[numOfOne]++;
                    }
                }
            }
        }

        return feature;
    }

    public static int Find(Mat inputMat, int value) {
        for(int i=0; i<inputMat.rows(); i++){
            if((double) value == inputMat.get(i,0)[0])
                return (int) inputMat.get(i,1)[0];
        }

        return -1;
    }

    public static int convertIntoNumber(int[] vector){
        int conversion = 0;

        for(int i=0, j=vector.length-1; i<vector.length; i++, j--){
            conversion += vector[i] * Math.pow(2,j);
        }

        return conversion;
    }

    public static int[] rightShift(int[] vector){
        int last = vector[vector.length-1];          // save off first element

        for( int index =vector.length-2; index >= 0 ; index-- )
            vector[index+1] = vector [index];

        vector[0] = last;

        return vector;
    }

    public static Mat createList(List<int[]> vector, int[] label){
        Mat returnMat = new Mat(vector.size()*8,2,CvType.CV_32F);

        for (int i=0; i<vector.size(); i++) {
            int[] currentVector = vector.get(i);
            for(int j=0; j<8; j++){
                returnMat.put(j+(i*8), 0, (double) convertIntoNumber(currentVector));
                returnMat.put(j+(i*8), 1, (double) label[i]);
                currentVector = rightShift(currentVector);
            }
        }

        return returnMat;
    }

	private double[] centroidRadii(MatOfPoint leafContour, Point centroid, double[] excInc){
		Point[] contourPoints = leafContour.toArray();

		int dim = contourPoints.length;
		double[] fullRadii = new double[dim];
		double[] radii = new double[36];
		float range = dim/36;
		double min = 100000;
		double max = 0;
		double sum;
		int pos = 0; // posizione punto con la massima distanza dal centro
        int start, stop;

		for (int i = 0; i < dim; i++) 	{
			fullRadii[i] = euclDist(contourPoints[i], centroid);
			if (fullRadii[i] > max){
				max = fullRadii[i];
				pos = i;
			}
			if (fullRadii[i] < min)
				min = fullRadii[i];
		}

        excInc[0] = max;
        excInc[1] = min;

		pos = pos - Math.round(range/2);
		if (pos < 0)
			pos = dim - pos;
		start = pos;

		for (int i = 0; i < 36; i++) {
			stop = pos + Math.round(range*(i+1));
			sum = 0;
			for (int j = start; j < stop; j++) {
				sum = sum + fullRadii[j%dim];
			}
			radii[i]= sum/(stop-start);
			if (i==0)
				max = radii[i];
			// i raggi vengono normalizzati
			radii[i] /= max;

			start = stop;
		}

		return radii;
	}

	private double euclDist(Point a, Point b){

		return Math.sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));

	}

	private double[] shapeDescriptor(double excircle, double incircle, MatOfPoint leafContour, Mat maskedGray){
		double[] shapeFeatures = new double[8];
		MatOfInt hullIds = new MatOfInt();
		MatOfPoint convexHull = new MatOfPoint();
		MatOfPoint2f leafContour2f = new MatOfPoint2f(leafContour.toArray() );
		MatOfPoint minBoundRectPoints = new MatOfPoint();
        Mat maskedRGB = new Mat();

		//convex hull
		Imgproc.convexHull(leafContour, hullIds);

		convexHull.create((int)hullIds.size().height, 1, CvType.CV_32SC2);

		for(int i = 0; i < hullIds.size().height ; i++)
		{
		    int index = (int)hullIds.get(i, 0)[0];
		    double[] point = new double[] {
		        leafContour.get(index, 0)[0], leafContour.get(index, 0)[1]
		    };
		    convexHull.put(i, 0, point);
		}

		// draw the convex hull
		List<MatOfPoint> hulls = new ArrayList<MatOfPoint>();
		hulls.add(convexHull);
    	// from gray to RGB
    	Imgproc.cvtColor(maskedGray, maskedRGB, Imgproc.COLOR_GRAY2RGBA, 4);
    	// disegno il convex hull, per controllare
    	Imgproc.drawContours(maskedRGB, hulls, 0, new Scalar(0, 255, 0), 3);

		//leaf area
		double area = Imgproc.contourArea(leafContour);
		//leaf perimeter
		double perimeter = Imgproc.arcLength(leafContour2f, true);

		//convex hull area
		double areaCH = Imgproc.contourArea(convexHull);
		//convex hull perimeter
		MatOfPoint2f convexHull2f = new MatOfPoint2f( convexHull.toArray() );
		double perimeterCH = Imgproc.arcLength(convexHull2f, true);

		RotatedRect minBoundRect = Imgproc.minAreaRect(leafContour2f);
		Point[] pts = new Point[4];
		minBoundRect.points(pts);

		minBoundRectPoints.create(4, 1, CvType.CV_32SC2);

		for(int i = 0; i < 4 ; i++)
		{
		    double[] point = new double[] {
		        pts[i].x, pts[i].y
		    };
		    minBoundRectPoints.put(i, 0, point);
		}

		//draw the minimum bounding rectangle
		List<MatOfPoint> rects = new ArrayList<MatOfPoint>();
		rects.add(minBoundRectPoints);
		Imgproc.drawContours(maskedRGB, rects, 0, new Scalar(255, 0, 0), 3);
		// extract the sizes of the bounding rectangle
		Size boundingRectSize = minBoundRect.size;

		double minAxisLength = Math.min(boundingRectSize.height, boundingRectSize.width);
		double maxAxisLength = Math.max(boundingRectSize.height, boundingRectSize.width);

		// fill features vector
		//Elongation
		shapeFeatures[0] = 1 - minAxisLength/maxAxisLength;
		//Rattangularity
		shapeFeatures[1] = area / (minAxisLength*maxAxisLength);
		//Solidity
		shapeFeatures[2] = area / areaCH;
		//Eccentricity
		shapeFeatures[3] = Math.sqrt(Math.pow(maxAxisLength, 2) - Math.pow(minAxisLength, 2)) / maxAxisLength;
		//Compactness
		shapeFeatures[4] = (4*Math.PI*area)/Math.pow(perimeter, 2);
		//Circularity
		shapeFeatures[5] = (4*Math.PI*area)/Math.pow(perimeterCH, 2);
		//Convexity
		shapeFeatures[6] = perimeterCH / perimeter;
		//Sfericity
		shapeFeatures[7] = incircle / excircle;

		return shapeFeatures;
	}

	private double[] getTextureFeatures(boolean symmetric, Mat maskedGray){
		int [][] offset = new int[][]{{1, 0},{1, -1},{0, -1},{-1, -1}};
		int dim = offset.length;
		double[] feature = new double[20*dim];
		double[] featurei;
		int [][] glcm;
        int occurrences;

		for (int i = 0; i < dim; i++){
            glcm = new int [256][256];
			occurrences = GLCM(glcm, new int[]{offset[i][0],offset[i][1]}, symmetric, maskedGray);
			featurei = textureDescriptors(glcm, occurrences);

			for (int j = 0; j < 20; j++){
                feature[(i*20)+j] = featurei[j];
            }
		}

		return feature;
	}

	private double[] getTextureFeaturesRidotto(boolean symmetric, Mat maskedGray){
		int [][] offset = new int[][]{{1, 0},{1, -1},{0, -1},{-1, -1}};
		int dim = offset.length;
		double[] feature = new double[5*dim];
		double[] featurei;
		int [][] glcm;
        int occurrences;


		for (int i = 0; i < dim; i++){
            glcm = new int [256][256];
			occurrences = GLCM(glcm, new int[]{offset[i][0],offset[i][1]}, symmetric, maskedGray);
			featurei = textureDescriptorsRidotto(glcm, occurrences);
			for (int j = 0; j < 5; j++)
				feature[(i*5)+j] = featurei[j];
		}
		return feature;
	}

	private int GLCM(int[][] glcm, int[] offset, boolean symmetric, Mat maskedGray){
		int i2, j2, width = maskedGray.cols(), height = maskedGray.rows();
		int occurrences = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				i2 = i + offset[0];
				j2 = j + offset[1];
				if(i2>=0 && j2 >= 0 && i2 < height && j2 < width){
					int pixel1 = (int) maskedGray.get(i, j)[0];
					int pixel2 = (int) maskedGray.get(i2, j2)[0];
					glcm[pixel1][pixel2]++;
					occurrences++;
					if (symmetric){
						glcm[pixel2][pixel1]++;
						occurrences++;
					}
				}
			}
		}

		occurrences -= glcm[0][0];
		glcm[0][0] = 0;

		return occurrences;
	}

	private double[] textureDescriptors(int [][] GLCM, int occurrences){
		 Log.i("Check", "Dentro quest'altra funzione considero occurrences = " + occurrences);

		double[] feature = new double[20];
		double u_x = 0, u_y = 0, s_x = 0, s_y = 0, hx = 0, hy = 0, hxy1 = 0, hxy2 = 0;
	    double []p_x = new double[256];
	    double []p_y = new double[256];
	    double []p_xminusy = new double[256];
	    double []p_xplusy = new double[511];
   	    double sumOcc = 0;

		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 256; j++){
				double G1 = (double)GLCM[i][j]/occurrences;
				sumOcc += G1;
				int i1 = i+1;
				int j1 = j+1;
				int ipluj = i+j;
	            int iminj = i-j;
	            int aiminj = Math.abs(iminj);

				p_x[i] = p_x[i] + G1;
				p_y[i] = p_y[i] + G1;
				p_xplusy[ipluj] = p_xplusy[ipluj] + G1;
				p_xminusy[aiminj] = p_xminusy[aiminj] + G1;
	            u_x  = u_x + i1 * G1;
	            u_y  = u_y + j1 * G1;

	    		feature[0] = feature[0] + (i1*j1*G1);//Autocorrelation
	            feature[1] = feature[1] + (Math.pow(aiminj, 2)*G1);//Contrast
	            feature[5] = feature[5] + (aiminj*G1);//Dissimilarity
	            feature[6] = feature[6] + Math.pow(G1,2);//Energy
	            feature[7] = feature[7] - (G1*Math.log(G1 + EPSILON));//Entropy
	            feature[8] = feature[8] + (G1/( 1 + Math.pow(iminj, 2)));//Homogeneity
	            feature[18] = feature[18] + (G1/( 1 + ( aiminj/256) ));//Inverse difference normalized
	            feature[19] = feature[19] + (G1/( 1 + Math.pow(iminj/256,2)));//Inverse difference moment normalized

				if (feature[9]< G1)
					feature[9]= G1;//Maximum probability

			}
		}

		//Log.i("Controllo", "Somma GLCM (deve essere 1): " + sumOcc);

		double u = (u_x + u_y)/2;

		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 256; j++){
				int i1 = i+1;
				int j1 = j+1;
				double G1 = (double)GLCM[i][j]/occurrences;

	            feature[10] = feature[10] + G1 * Math.pow(i1 - u,2);//Sum of squares: Variance
	            s_x  = s_x  + (Math.pow(i1 - u_x,2) * G1);
	            s_y  = s_y  + (Math.pow(j1 - u_y,2) * G1);
	            feature[3] = feature[3] + (Math.pow(i1 + j1 - u_x - u_y,4) * G1);//Cluster Prominence
	            feature[4] = feature[4] + (Math.pow(i1 + j1 - u_x - u_y,3) * G1);//Cluster Shade
	            hxy1 = hxy1 - (G1 * Math.log(p_x[i] * p_y[j] + EPSILON));
	            hxy2 = hxy2 - ((p_x[i] * p_y[j]) * Math.log(p_x[i] * p_y[j] + EPSILON));
			}
	        hx = hx - (p_x[i] * Math.log(p_x[i] + EPSILON));
	        hy = hy - (p_y[i] * Math.log(p_y[i] + EPSILON));
	        feature[14] = feature[14] + (Math.pow(i,2)*p_xminusy[i]);//Difference variance
	        feature[15] = feature[15] - (p_xminusy[i] * Math.log(p_xminusy[i] + EPSILON));//Difference entropy
		}

	    s_x = Math.sqrt(s_x);
	    s_y = Math.sqrt(s_y);
	    feature[2] = (feature[0] - u_x*u_y)/(s_x*s_y);//Correlation
	    feature[16]= ( feature[7] - hxy1 ) / ( Math.max(hx,hy) );//Information measure of correlation1
	    feature[17] = Math.sqrt( 1 - Math.pow(Math.E, -2*( hxy2 - feature[7])) );//Information measure of correlation2

		for (int i = 0; i < 511; i++) {
		    feature[11] = feature[11] + ((i+2)*p_xplusy[i]);//Sum average
		    feature[13] = feature[13] - (p_xplusy[i]*Math.log(p_xplusy[i] + EPSILON));//Sum entropy
		}

		for (int i = 0; i < 511; i++) {
			feature[12] = feature[12] + (Math.pow(((i+1) - feature[11]),2)*p_xplusy[i]);//Sum variance
		}

		return feature;
	}

	private double[] textureDescriptorsRidotto(int [][] GLCM, int occurrences){
		 Log.i("Check", "Dentro quest'altra funzione considero occurrences = " + occurrences);

		double[] feature = new double[5];
		double u_x = 0, u_y = 0, s_x = 0, s_y = 0;

		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 256; j++){
				double G1 = (double)GLCM[i][j]/occurrences;

				int i1 = i+1;
				int j1 = j+1;

	            int iminj = i-j;
	            int aiminj = Math.abs(iminj);


	            u_x  = u_x + i1 * G1;
	            u_y  = u_y + j1 * G1;

	            feature[0] = feature[0] + (Math.pow(aiminj, 2)*G1);//Contrast
	            feature[2] = feature[2] + Math.pow(G1,2);//Energy
	            feature[3] = feature[3] - (G1*Math.log(G1 + EPSILON));//Entropy
	            feature[4] = feature[4] + (G1/( 1 + Math.pow(iminj, 2)));//Homogeneity

			}
		}

		//Log.i("Controllo", "Somma GLCM (deve essere 1): " + sumOcc);
		feature[0] = feature[0]*(1/Math.pow(255, 2));
		feature[3] = feature[3]*(1/(2*Math.log(256)));


		for (int i = 0; i < 256; i++) {
			for (int j = 0; j < 256; j++){
				int i1 = i+1;
				int j1 = j+1;
				double G1 = (double)GLCM[i][j]/occurrences;

			    feature[1] = feature[1] + (i - u_x)*(j - u_y)*G1;


	            s_x  = s_x  + (Math.pow(i1 - u_x,2) * G1);
	            s_y  = s_y  + (Math.pow(j1 - u_y,2) * G1);

			}
	    }

	    s_x = Math.sqrt(s_x);
	    s_y = Math.sqrt(s_y);

	    feature[1] = feature[1]/(s_x*s_y*2) + 0.5;//Correlation

		return feature;
	}
}