package com.example.leafrecognizer;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Jason on 12/10/2014.
 */
public class SLIC {
//    RNG rng(12345);

    //    Mat sobel = (Mat_<float>(3,3) << -1/16., -2/16., -1/16., 0, 0, 0, 1/16., 2/16., 1/16.);
    Mat sobel;
    Mat labell;
    private int nx, ny;
    private int m; //compactness

    public SLIC(int nx, int ny, int m){
        this.nx = nx;
        this.ny = ny;
        this.m = m;
        sobel = new Mat(3, 3, CvType.CV_32FC1);
        sobel.put(0, 0, -1/16.);
        sobel.put(0, 1, -2/16.);
        sobel.put(0, 2, -1/16.);
        sobel.put(1, 0, 0);
        sobel.put(1, 1, 0);
        sobel.put(1, 2, 0);
        sobel.put(2, 0, 1/16.);
        sobel.put(2, 1, 2/16.);
        sobel.put(2, 2, 1/16.);
    }

    public SuperpixelImage createSuperpixel(Bitmap input){
        // Read in image
        Mat im = new Mat(input.getWidth(),input.getHeight(),CvType.CV_8U);
        Utils.bitmapToMat(input,im);
        // Log.d("check "," input "+ input.getWidth()+" "+input.getHeight());
        // Log.d("check "," input "+ im);

        if (input == null) {
            Log.d("SLIC ERROR: ","no image data at "+input);
            return null;
        }


        // CALCOLO GRADIENTE
        Mat grad =  new Test().testLaplacian(im);


        // Scale to [0,1] and l*a*b colorspace
        im.convertTo(im, CvType.CV_32F, 1/255.);
        Mat imlab = new Mat();
        Imgproc.cvtColor(im, imlab, Imgproc.COLOR_BGR2Lab);

        int h = im.rows();
        int w = im.cols();
        int n = nx*ny; //spcount 225 -> n=K -> N=nx=ny

        float dx = w / (float)nx;
        float dy = h / (float)ny;
        int S = (int) ((dx + dy + 1)/2); // window width

        // Initialize centers
        MatOfPoint2f centers = new MatOfPoint2f(); //centroids
        List<Point> centersList = new ArrayList<Point>();
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                centersList.add(new Point(i * dx + dx / 2, j * dy + dy / 2));
            }
        }
        centers.fromList(centersList); //la matrice viene creata dalla lista

        // Initialize labels and distance maps
        MatOfInt label_vec = new MatOfInt();
        List<Integer> label_vec_list = new ArrayList<Integer>();
        for (int i = 0; i < n; i++)
            label_vec_list.add(i * 255 * 255 / n); //255
        label_vec.fromList(label_vec_list);

        Scalar negativeOne = new Scalar(-1);
        Mat labels = new Mat();
        Mat labels2 = new Mat();
        Mat dists = new Mat();
        Core.multiply(Mat.ones(imlab.size(), CvType.CV_32S),negativeOne,labels);
        Core.multiply(Mat.ones(imlab.size(), CvType.CV_32S),negativeOne,labels2);
        Core.multiply(Mat.ones(imlab.size(), CvType.CV_32S),negativeOne,dists);

        Mat window;
        Point p1, p2;
        double[] p1_lab;
        double[] p2_lab;
        double[] p1_grad;
        double[] p2_grad;

        // Iterate "numberOfIteration" times. In practice more than enough to converge
        int numberOfIteration = 1; //max_iteration 5
        for (int i = 0; i < numberOfIteration; i++) {
            // For each center...
            for (int c = 0; c < n; c++)
            {
                int label = label_vec.toList().get(c);

                //Log.d("DEBUG", Integer.toString(label));

                p1 = centers.toList().get(c);
                p1_lab = imlab.get((int)p1.y,(int)p1.x);//.at<Vec3f>(p1);
                p1_grad = grad.get((int)p1.y,(int)p1.x);

                //Log.d("DEBUG", Double.toString(gradiente[0]));

                int xmin = (int) Math.max(p1.x-S, 0); //funzione find_neighbors_of
                int ymin = (int) Math.max(p1.y-S, 0);
                int xmax = (int) Math.min(p1.x+S, w-1);
                int ymax = (int) Math.min(p1.y+S, h-1);

                // Search in a window around the center
                window = new Mat(im, new Range(ymin, ymax), new Range(xmin, xmax));

                // Reassign pixels to nearest center
                for (int row = 0; row < window.rows(); row++) {
                    for (int col = 0; col < window.cols(); col++) {
                        p2 = new Point(xmin + col, ymin + row);
                        p2_lab = imlab.get((int) p2.y, (int) p2.x);// at<Vec3f>(p2);
                        p2_grad = grad.get((int) p2.y, (int) p2.x);
                        float d = dist(p1, p2, p1_lab, p2_lab, m, S, p1_grad[0], p2_grad[0]);
                        float last_d = (float) dists.get((int) p2.y, (int) p2.x)[0];
                        if (d < last_d || last_d == -1) {
                            dists.put((int) p2.y, (int) p2.x, d);
                            labels.put((int) p2.y, (int) p2.x, label);
                            labels2.put((int) p2.y, (int) p2.x, c);
                        }
                    }
                }
            }
        }

        this.labell = labels2;
        // Return the superpixel data object




        return new SuperpixelImage(centers,label_vec,labels,sobel);
    }

    public Bitmap createBoundedBitmap(Bitmap input){
        // Read in image
//        Mat im = imread(input);
        Mat im = new Mat(input.getWidth(),input.getHeight(),CvType.CV_8U);
        Utils.bitmapToMat(input,im);

        if (input == null) {
            Log.d("SLIC ERROR: ","no image data at "+input);
            return null;
        }


        // CALCOLO GRADIENTE
        Mat grad =  new Test().testLaplacian(im);


        // Scale to [0,1] and l*a*b colorspace
        im.convertTo(im, CvType.CV_32F, 1/255.);
        Mat imlab = new Mat();
        Imgproc.cvtColor(im, imlab, Imgproc.COLOR_BGR2Lab);

        int h = im.rows();
        int w = im.cols();
        int n = nx*ny;

        float dx = w / (float)nx;
        float dy = h / (float)ny;
        int S = (int) ((dx + dy + 1)/2); // window width

        // Initialize centers
        MatOfPoint2f centers = new MatOfPoint2f();
        List<Point> centersList = new ArrayList<Point>();
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                centersList.add(new Point(j * dx + dx / 2, i * dy + dy / 2));
            }
        }
        centers.fromList(centersList);

        // Initialize labels and distance maps
        MatOfInt label_vec = new MatOfInt();

        List<Integer> label_vec_list = new ArrayList<Integer>();
        for (int i = 0; i < n; i++)
            label_vec_list.add(i * 255 * 255 / n); //255
        label_vec.fromList(label_vec_list);

        Scalar negativeOne = new Scalar(-1);
        Mat labels = new Mat();
        Mat dists = new Mat();
        Core.multiply(Mat.ones(imlab.size(), CvType.CV_32S),negativeOne,labels);
        Core.multiply(Mat.ones(imlab.size(), CvType.CV_32S),negativeOne,dists);

        Mat window;
        Point p1, p2;
        double[] p1_lab;
        double[] p2_lab;
        double[] p1_grad;
        double[] p2_grad;

        // Iterate 1 times. In practice more than enough to converge
        int numberOfIteration = 1;
        for (int i = 0; i < numberOfIteration; i++) {
            // For each center...
            for (int c = 0; c < n; c++)
            {
                int label = label_vec.toList().get(c);
                p1 = centers.toList().get(c);
                p1_lab = imlab.get((int)p1.y,(int)p1.x);//.at<Vec3f>(p1);
                p1_grad = grad.get((int)p1.y,(int)p1.x);

                int xmin = (int) Math.max(p1.x-S, 0);
                int ymin = (int) Math.max(p1.y-S, 0);
                int xmax = (int) Math.min(p1.x+S, w-1);
                int ymax = (int) Math.min(p1.y+S, h-1);

                // Search in a window around the center
                window = new Mat(im, new Range(ymin, ymax), new Range(xmin, xmax));

                // Reassign pixels to nearest center
                for (int row = 0; row < window.rows(); row++) {
                    for (int col = 0; col < window.cols(); col++) {
                        p2 = new Point(xmin + col, ymin + row);
                        p2_lab = imlab.get((int) p2.y, (int) p2.x);// at<Vec3f>(p2);
                        p2_grad = grad.get((int) p2.y, (int) p2.x);

                        float d = dist(p1, p2, p1_lab, p2_lab, m, S, p1_grad[0], p2_grad[0]);
                        float last_d = (float) dists.get((int) p2.y, (int) p2.x)[0];

                        //Log.d("DISTANZE", "new:  " + d + "   old:   " + last_d);

                        if (d < last_d || last_d == -1) {
                            dists.put((int) p2.y, (int) p2.x, d);
                            labels.put((int) p2.y, (int) p2.x, label);
                        }
                    }
                }
            }
        }


        SuperpixelImage superpixelImage = new SuperpixelImage(centers,label_vec,labels,sobel);
        return superpixelImage.createBitmapWithBoundary(input);
    }

    private float dist(Point p1, Point p2, double[] p1_lab, double[] p2_lab, float compactness, float S,
                       double p1_grad, double p2_grad){
        float dl = (float) (p1_lab[0] - p2_lab[0]);
        float da = (float) (p1_lab[1] - p2_lab[1]);
        float db = (float) (p1_lab[2] - p2_lab[2]);


        float dx = (float) (p1.x - p2.x);
        float dy = (float) (p1.y - p2.y);


        double dg = (p1_grad - p2_grad);

        if(p1_grad > p2_grad && ((p1_grad - p2_grad) < 40))
            dg = 0;
        if(p2_grad > p1_grad && ((p2_grad - p1_grad) < 40))
            dg = 0;


        dg = (dg <= 0.0F) ? 0.0F - dg : dg;
        float d_xy = (float) Math.sqrt(dx * dx + dy * dy);
        float d_lab = (float) Math.sqrt(dl*dl + da*da + db*db);


        //dg = (dg < 15) ? 0 : dg;
        //return (float) dg;
        return d_lab + compactness/S * d_xy;
    }

    public Mat GetLabels() {
        Mat int_labels = new Mat();
        //MinMaxLocResult s = Core.minMaxLoc(this.labels);
        //this.labels.convertTo(int_labels, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal));
        this.labell.convertTo(int_labels, CvType.CV_16U);//,65535.0f/(s.maxVal-s.minVal),-s.minVal*65535.0f/(s.maxVal-s.minVal));

        return int_labels;
    }
}
