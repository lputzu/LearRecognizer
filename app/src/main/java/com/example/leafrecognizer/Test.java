package com.example.leafrecognizer;

import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import com.example.leafrecognizer.ers.*;

/**
 * Created by Cesare on 26/05/2016.
 */
public class Test {

    private static String appPath = Environment.getExternalStorageDirectory().getPath() + File.separator + "LeafRecog";
    private static String datasetPath = appPath + File.separator;
    static String sdpath = Environment.getExternalStorageDirectory().getPath();
    static String app_path = sdpath + File.separator + "LeafRecog";

    public static Mat normalizzaFeatureHu(Mat inputMat) {
        File csvFile = new File(app_path + File.separator + "leafDatasetHu_normalizzatoMaxMin.csv");

        try{
            FileReader fr = new FileReader(csvFile);
            BufferedReader br = new BufferedReader(fr);

            String newLine;
            float value;
            int i = 0;

            //Log.i("Comunicazione", "Creato BufferedReader");

            while((newLine = br.readLine()) != null)
            {
                String[] maxMin = newLine.split(",");

                inputMat.put(0, i, (inputMat.get(0, i)[0] - Double.valueOf(maxMin[1])) /
                        (Double.valueOf(maxMin[0]) - Double.valueOf(maxMin[1])));

                i++;
            }

            br.close();
            fr.close();

        }
        catch (Exception e)
        {
        }

        return inputMat;
    }

    public static Mat testNormalizzaHu(){
        Mat trainData = Test.getDatasetMat("leafDatasetHu-001.csv", 7);
        File csvFile = new File(app_path + File.separator + "leafDatasetHu_normalizzatoMaxMin.csv");

        try {
            FileWriter fw = new FileWriter(csvFile);

            double max, min;
            for (int i = 0; i < trainData.cols(); i++) {
                max = trainData.get(0, i)[0];
                min = trainData.get(0, i)[0];
                for (int j = 0; j < trainData.rows(); j++) {
                    if (trainData.get(j, i)[0] > max)
                        max = trainData.get(j, i)[0];
                    if (trainData.get(j, i)[0] < min)
                        min = trainData.get(j, i)[0];
                }


                fw.append("" + max);
                fw.append(",");
                fw.append("" + min);
                if(i != trainData.cols()-1) {
                    fw.append("\n");
                }

                Log.d("norm", "max: " + max + "   min: " + min);

                for (int j = 0; j < trainData.rows(); j++) {
                    if((max - min) != 0)
                        trainData.put(j, i, (trainData.get(j, i)[0] - min) / (max - min));
                }
            }

            fw.close();
        }
        catch (Exception e) {
        }

        return trainData;
    }

    public static Mat testNormalizzaLBP(){
        Mat trainData = Test.getDatasetMat("leafDatasetLBP2.csv", 36);
        File csvFile = new File(app_path + File.separator + "leafDatasetLBP2_normalizzatoDati.csv");

        try {
            FileWriter fw = new FileWriter(csvFile);

            double max, min;
            for (int i = 0; i < trainData.cols(); i++) {
                max = trainData.get(0, i)[0];
                min = trainData.get(0, i)[0];
                for (int j = 0; j < trainData.rows(); j++) {
                    if (trainData.get(j, i)[0] > max)
                        max = trainData.get(j, i)[0];
                    if (trainData.get(j, i)[0] < min)
                        min = trainData.get(j, i)[0];
                }


                fw.append("" + max);
                fw.append(",");
                fw.append("" + min);
                if(i != trainData.cols()-1) {
                    fw.append("\n");
                }

                Log.d("norm", "max: " + max + "   min: " + min);

                for (int j = 0; j < trainData.rows(); j++) {
                    if((max - min) != 0)
                        trainData.put(j, i, (trainData.get(j, i)[0] - min) / (max - min));
                }
            }

            fw.close();
        }
        catch (Exception e) {
        }

        return trainData;
    }

    public static Mat normalizzaFeatureLBP(Mat inputMat) {
        File csvFile = new File(app_path + File.separator + "leafDatasetHu_normalizzatoDati.csv");

        try{
            FileReader fr = new FileReader(csvFile);
            BufferedReader br = new BufferedReader(fr);

            String newLine;
            float value;
            int i = 0;

            //Log.i("Comunicazione", "Creato BufferedReader");

            while((newLine = br.readLine()) != null) {
                String[] maxMin = newLine.split(",");

                inputMat.put(0, i, (inputMat.get(0, i)[0] - Double.valueOf(maxMin[1])) /
                        (Double.valueOf(maxMin[0]) - Double.valueOf(maxMin[1])));

                i++;
            }

            br.close();
            fr.close();

        }
        catch (Exception e)
        {
        }

        return inputMat;
    }

    public static void testKNN(){

        Mat test = new Mat(1,9,CvType.CV_32F);
        for(int i=0; i<test.cols(); i++){
            test.put(0,i,4);
        }

        Mat trainData = new Mat(6,9,CvType.CV_32F);
        for(int i=0; i<trainData.rows(); i++){
            for(int j=0; j<trainData.cols(); j++){
                trainData.put(i,j,i+1);
            }
        }

        List<Integer> trainLabs = new ArrayList<Integer>();
        // 10 digits a 5 rows:
        for (int r=0; r<6; r++) {
            trainLabs.add(r);
        }

        // make a Mat of the train labels, and train knn:
        Mat res = new Mat(6,1,CvType.CV_32F), neigh = new Mat(), dist = new Mat();
        CvKNearest knn = new CvKNearest();
        knn.train(trainData, Converters.vector_int_to_Mat(trainLabs));
        // now test predictions:

        Mat one_feature = trainData.row(1);

        float p = knn.find_nearest(test, 3, res, neigh, dist);
        Log.d("PROVA", " " + p + " " + res.dump() +  "  " + neigh.dump()+ "  " + dist.dump());







/*



        // TRAIN DATA
        Mat trainData = new Mat(6,9,CvType.CV_32FC1);
        for(int i=0; i<trainData.rows(); i++){
            for(int j=0; j<trainData.cols(); j++){
                trainData.put(i,j,i+1);
            }
        }
        Test.testFromMatToCvs(trainData, "trainData");


        // TRAIN LABELS
        Mat trainLabels = new Mat(1,6,CvType.CV_32SC1);
        for(int i=0; i<6; i++){
            trainLabels.put(0,i,i);
        }
        Test.testFromMatToCvs(trainLabels, "trainLabels");


        // CREATING KNN
        Mat respondes = new Mat(6,1,CvType.CV_32FC1);
        CvKNearest knn = new CvKNearest(trainData, respondes, trainLabels, false, 5);

        // TESTING VALUE
        Mat test = new Mat(1,9,CvType.CV_32FC1);
        for(int i=0; i<test.cols(); i++){
            test.put(0,i,1);
        }
        Test.testFromMatToCvs(test, "test");


        // TESTING KNN WITH TESTING VALUE
        Mat neigh = new Mat(), dist = new Mat();
        float r = knn.find_nearest(test, 1, respondes, neigh, dist);

        Test.testFromMatToCvs(respondes, "RESULT");
        Test.testFromMatToCvs(neigh, "neigh");
        Test.testFromMatToCvs(dist, "dist");

        Log.d("PROVA", "" + r + "  " + respondes.dump());
*/

    }

    public static int[] testCalculateHistogram(Mat input) {
        int histogram[] = new int[256];

        for(int i=0; i<input.rows(); i++) {
            for(int j=0; j<input.cols(); j++) {
                histogram[(int) input.get(i,j)[0]]++;
            }
        }

        return histogram;
    }

    public static boolean testCheckLeaf(Mat input){
        int histogram[] = Test.testCalculateHistogram(input);
        int pixels = input.rows() * input.cols();
        int salencyPixels = 0;

        for(int i=240; i< 255; i++) {
            salencyPixels += histogram[i];
        }

/*        for(int i=0; i<10; i++) {
            salencyPixels += histogram[i];
        }*/

        Log.d("TOT", "   " + pixels);
        Log.d("SAL", "   " + salencyPixels);
        Log.d("PER aasd", "   " + (salencyPixels*100/pixels) + "%");

        return true;
    }

    public static void testSalvaImmagine(String photoDir, String photoPath, Bitmap realImage){

        String photoFile;
        String filePath;
        File pictureFile;
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyymmddhhmmss");
        String date = dateFormat.format(new Date());



        File pictureFileDir = new File(photoDir);
        if (!pictureFileDir.exists() &&  !pictureFileDir.mkdirs()) {
            Log.d("SAVING", "Can't create   directory to save image");
        }

        int lastPoint = photoPath.lastIndexOf(".");
        int lastSeparator = photoPath.lastIndexOf(File.separator);
        photoFile = "SEGMENTAZIONE_SLIC_" + photoPath.substring(lastSeparator + 1,lastPoint) + ".jpg";

        filePath = photoDir + File.separator + photoFile;

        pictureFile = new File( filePath);

        try {
            FileOutputStream fos = new FileOutputStream(pictureFile);
            realImage.compress(Bitmap.CompressFormat.JPEG, 90, fos);
            fos.close();
            //Highgui.imwrite(filePath, mMask3);
            //  Highgui.imwrite(filePath2, mMask2);
            Log.d("SAVING", "New Image saved");
        } catch (Exception error) {
            Log.d("SAVING", "Image could not be saved");
        }
    }

    public Mat testSobel(Mat image) {

        Mat src_gray = new Mat();
        Mat grad = new Mat();
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;

        Imgproc.cvtColor( image, src_gray, Imgproc.COLOR_RGBA2GRAY );


        Mat grad_x = new Mat(), grad_y = new Mat();
        Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

        /// Gradient X
        Imgproc.Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, Imgproc.BORDER_DEFAULT );
        /// Gradient Y
        Imgproc.Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, Imgproc.BORDER_DEFAULT );

        grad_x.convertTo(abs_grad_x, CvType.CV_8U);
        grad_y.convertTo(abs_grad_y, CvType.CV_8U);

        Core.add(abs_grad_x,abs_grad_y, grad);

        return grad;
    }

    public static Mat testWatershed(Mat image) {

        Mat threeChannel = new Mat();
        Imgproc.cvtColor(image, threeChannel, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(threeChannel, threeChannel, 100, 255, Imgproc.THRESH_BINARY);

        Mat fg = new Mat(image.size(),CvType.CV_8U);
        Imgproc.erode(threeChannel,fg,new Mat(),new Point(-1,-1),2);

        Mat bg = new Mat(image.size(),CvType.CV_8U);
        Imgproc.dilate(threeChannel,bg,new Mat(),new Point(-1,-1),3);
        Imgproc.threshold(bg,bg,1, 128,Imgproc.THRESH_BINARY_INV);

        Mat markers = new Mat(image.size(),CvType.CV_8U, new Scalar(0));
        Core.add(fg, bg, markers);

        WatershedSegmenter segmenter = new WatershedSegmenter();
        segmenter.setMarkers(markers);

        Imgproc.cvtColor(image, threeChannel, Imgproc.COLOR_RGBA2BGR);

        return segmenter.process(threeChannel);
    }

    public Mat testLaplacian(Mat image) {

        Mat grad = new Mat(), src_gray = new Mat();

        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;

        Imgproc.cvtColor(image, src_gray, Imgproc.COLOR_RGBA2GRAY );

        Imgproc.equalizeHist(src_gray,src_gray);

        Imgproc.Laplacian( src_gray, grad, ddepth, kernel_size, scale, delta, Imgproc.BORDER_DEFAULT );
        Core.convertScaleAbs( grad, grad );

        /*
        for(int i=0; i < grad.rows(); i++) {
            for(int j=0; j<grad.cols(); j++) {
                if(grad.get(i,j)[0] < 15)
                    grad.put(i,j,0);
            }
        }

*/
/*
        Imgproc.equalizeHist(grad,grad);

        for(int i=0; i < grad.rows(); i++) {
            for(int j=0; j<grad.cols(); j++) {
                if(grad.get(i,j)[0] < 60)
                    grad.put(i,j,0);
            }
        }

        Imgproc.erode(grad,grad,new Mat(),new Point(-1,-1),1);
        Imgproc.dilate(grad,grad,new Mat(),new Point(-1,-1),1);

        */

        return grad;
    }

    public static void testFromMatToCvs(Mat input, String fileName) {
        File csvFile = new File(Environment.getExternalStorageDirectory().getPath() + File.separator + "LeafRecog" +
                File.separator + fileName + ".csv");

        try {

            FileWriter fw = new FileWriter(csvFile);
            double[] value;

            for(int i=0; i<input.rows(); i++) {
                for(int j=0; j<input.cols(); j++) {
                    value = input.get(i, j);
                    fw.append("" + value[0]);
                    fw.append(",");
                }
                fw.append("\n");
            }

            fw.flush();
            fw.close();
        } catch (Exception e) {
            Log.e("TEST_CLASS", "Qua c'è stato un errore: " + e.getMessage());
        }
    }

    public static Mat testERS(Mat input) {
        Mex_ers process = new Mex_ers();
        Mat output = process.execute(input, 225);

        Log.d("check","out:   " + output.rows() + "  " + output.cols());

        return output;
    }

    public static Bitmap testFromLabelToSuperpixel(Mat label, Mat mImgRGBA) {

        Mat output = label.clone();

        //Log.d("check",output.rows() + "  " + output.cols());

        output.convertTo(output, CvType.CV_32F);
        Mat gx = new Mat();
        Mat gy = new Mat();
        Mat grad = new Mat();


        Mat sobel = new Mat(3, 3, CvType.CV_32FC1);
        sobel.put(0, 0, -1/16.);
        sobel.put(0, 1, -2/16.);
        sobel.put(0, 2, -1/16.);
        sobel.put(1, 0, 0);
        sobel.put(1, 1, 0);
        sobel.put(1, 2, 0);
        sobel.put(2, 0, 1/16.);
        sobel.put(2, 1, 2/16.);
        sobel.put(2, 2, 1/16.);

        Imgproc.filter2D(output, gx,  -1, sobel);
        Imgproc.filter2D(output, gy, -1, sobel.t());
        Core.magnitude(gx, gy, grad);

        Core.compare(grad,new Scalar(0.0001),grad,Core.CMP_GT);
        Core.divide(grad,new Scalar(255),grad);
        Mat show = new Mat();
        Core.subtract(Mat.ones(grad.size(),grad.type()),grad,show);
        show.convertTo(show, CvType.CV_8U);

        //Log.d("check",show.rows() + "  " + show.cols());

        Vector<Mat> rgb = new Vector<Mat>(3);

        Mat im = new Mat(mImgRGBA.cols(), mImgRGBA.rows(), CvType.CV_8U);
        im = mImgRGBA.clone();

        im.convertTo(im,CvType.CV_8UC3);

        Core.split(im, rgb);

        //Log.d("check", "" + rgb.get(0).rows() + "  " + rgb.get(0).cols() + "   " + show.rows() + "  " + show.cols());

        for (int i = 0; i < 3; i++)
            rgb.set(i,rgb.get(i).mul(show));

        Mat outputMat = new Mat();
        Core.merge(rgb, outputMat);

        Bitmap outputB = Bitmap.createBitmap(outputMat.width() , outputMat.height(), Bitmap.Config.RGB_565);

        return outputB;
    }

    public void BackUpVersioneFunzionante(){

        /*
        // VERSIONE FUNZIONANTE

        //sesta versione
        Mat prova = Highgui.imread(photoPath);
        GMRsaliency salmap = new GMRsaliency(this,realImage,prova);
        Mat mMask = salmap.GetSal(1,null,0);



        //sesta versione
        Core.MinMaxLocResult s = Core.minMaxLoc(mMask);
        mMask.convertTo(mMask, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal)); //CV_32FC1

        //Test.testCheckLeaf( mMask);


        //binarizzazione con soglia prima del salvataggio
        Mat blur = new Mat();
        Mat mMask2 = new Mat();
        Imgproc.bilateralFilter(mMask, blur, 12,24,6);
        Imgproc.threshold(blur, mMask2,0,255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);


        //versione vecchia
        LeafProcessor leafProc = new LeafProcessor(this,  mImgRGBA, mMask, mMask2);

        Mat mMask3 = leafProc.segmentLeafRG_LAB();

        */
    }

    public static Mat getDatasetMat(String datasetName, int numOfFeatures){

        Mat leafDataset;

        try
        {
            File csvFile = new File(app_path + File.separator + "Features" + File.separator + datasetName);
            FileReader fr = new FileReader(csvFile);
            BufferedReader br = new BufferedReader(fr);

            //leafDataset = new Mat(1907,71,CvType.CV_32FC1, new Scalar(0));

            leafDataset = new Mat(1907, numOfFeatures, CvType.CV_32FC1, new Scalar(0));


            String newLine;
            float value;
            int i = 0;

            //Log.i("Comunicazione", "Creato BufferedReader");

            while((newLine = br.readLine()) != null)
            {
                String[] leafValues = newLine.split(",");
                //Log.i("Scorrendo", "Siamo alla riga numero " + i + " e il primo valore è " + Float.parseFloat(leafValues[0]));

                for(int j=0; j<leafValues.length; j++)
                {
                    value = Float.parseFloat(leafValues[j]);
                    leafDataset.put(i, j, value);
                }

                i++;
            }

            br.close();
            fr.close();

        }
        catch (Exception e)
        {
            leafDataset = null;
        }

        return leafDataset;
    }

}
