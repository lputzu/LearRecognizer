package com.example.leafrecognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FilenameFilter;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Vector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;
import org.opencv.ml.CvSVM;
import org.opencv.utils.Converters;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.leafrecognizer.R;
import com.example.leafrecognizer.saliency.ImageSignature;
import com.example.leafrecognizer.saliency.ImageSignatureSaliencyDetector;
import com.example.leafrecognizer.watershed.*;

public class ImageProcessActivity extends Activity {

    private Button cameraButton;
    private Button homeButton;
    private Bitmap salImage;
    private Bitmap realImage;
    private Bitmap realImage2;

    // the saliency detector
    private ImageSignatureSaliencyDetector mImageSignature;

    private ImageView image;
    final String TAG = "Hello World";
    String sdpath = Environment.getExternalStorageDirectory().getPath();
    String app_path = sdpath + File.separator + "LeafRecog";
    String immagini_test = app_path + File.separator + "Testing";
    String immagini_test_elaborate = app_path + File.separator + "TestingElab";
    String photoDir = app_path + File.separator + "Segmentazione";
    String FlaviaImages = sdpath + File.separator + "Full";
    final String modelPath = app_path + File.separator + "model.xml";
    String segmentazioneFlavia = app_path + File.separator + "SegmentazioneFlavia";
    String FlaviaLeavesFull = sdpath + File.separator + "FlaviaLeavesFull";
    private String photoPath2;

    //private String photoPath =  FlaviaImages + File.separator + "23.jpg";
    //private String photoPath = FlaviaImages + File.separator + "01.jpg";
    private String photoPath = immagini_test + File.separator + "1.jpg";

    //private String photoPathTestSVM = app_path + File.separator + "2.jpg";


    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Create and set View
                    try {
                        processImage();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mOpenCVCallBack))
        {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        }

        setContentView(R.layout.image_process_layout);
        Intent i = getIntent();

        // TODO  riga commentata per eseguire i test (da decommentare in seguito)
        //photoPath = i.getStringExtra("photoPath");


         //photoPath = photoDir + File.separator + "0012.jpg";
        //String photoPath2 = photoDir + File.separator + "0012qcut.png";
        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future
        realImage = BitmapFactory.decodeFile(photoPath, options);
        //realImage2 = BitmapFactory.decodeFile(photoPath2, options);
        // salImage = BitmapFactory.decodeFile(photoPath2, options);
        image = (ImageView) findViewById(R.id.imageView_photo);


        cameraButton = (Button) findViewById(R.id.back_2_camera);

        cameraButton.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                try {
                    startCameraActivity();

                } catch (Exception e) {

                }
            }
        });

        homeButton = (Button) findViewById(R.id.go_home2);

        homeButton.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                try {
                    gotoMainActivity();

                } catch (Exception e) {

                }
            }
        });


    }

    @Override
    protected void onPause() {
        super.onPause();

    }

    @Override
    public void onResume() {
        super.onResume();
    }

    public void startCameraActivity() {
        Intent intent = new Intent(this, MainActivity.class);
        //Intent intent = new Intent(this, CameraActivity.class);
        startActivity(intent);
    }

    public void gotoMainActivity() {
        /*Intent intent = new Intent(this, MainActivity.class);
        startActivity(intent);*/
        this.finish();
        System.exit(0);
    }

    public void processImage() throws InterruptedException{
        //visualizzaSuperpixelSlic();
        //visualizzxaSuperpixelErs();
        //visualizzaWatershed();

        TestClassificazione();

        //segmentaSLIC();

        //segmentaERS();

        //prova();

        //Test.testSalvaImmagine(immagini_test, "Test.jpg", realImage);
    }

    public void segmentaFlavia() {
        Mat mImgRGBA = new Mat();
        Mat prova = new Mat();

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 4;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File pictureFileDir = new File(FlaviaLeavesFull);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        int nFiles = files.length;

        try {
            for(int h=1440; h<=nFiles; h++) {
                photoPath = files[h-1].getPath();

                Log.d("immagine","n:  " + h + "     " + photoPath + realImage.getHeight() + "   " + realImage.getWidth());

                realImage = BitmapFactory.decodeFile(photoPath, options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                Utils.bitmapToMat(BitmapFactory.decodeFile(photoPath, options), prova);
                GMRsaliency salmap = new GMRsaliency(this,realImage,prova);
                Mat mMask = salmap.GetSal(1, null, 0);

                //sesta versione
                MinMaxLocResult s = Core.minMaxLoc(mMask);
                mMask.convertTo(mMask, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal)); //CV_32FC1
                Utils.matToBitmap(mMask, realImage);
                //Test.testSalvaImmagine(photoDir, photoPath, realImage);

                //binarizzazione con soglia prima del salvataggio
                Mat blur = new Mat();
                Mat mMask2 = new Mat();
                Imgproc.bilateralFilter(mMask, blur, 12,24,6);
                Imgproc.threshold(blur, mMask2,0,255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);

                //versione vecchia
                LeafProcessor leafProc = new LeafProcessor(this, mImgRGBA,mMask,mMask2);
                Mat mMask3 = leafProc.segmentLeafRG_LAB_NotCircle();
                Imgproc.cvtColor(mMask3, mMask3, Imgproc.COLOR_RGB2GRAY);

                Utils.matToBitmap(mMask3, realImage);

                //image.setImageBitmap(realImage);

                Test.testSalvaImmagine(segmentazioneFlavia, photoPath, realImage);
            }
        } catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    public Mat getLabelMat(){
        Mat labels = new Mat(1907,1, CvType.CV_32FC1, new Scalar(0));

        int j=0;
        int x=0;
        for(int i=0; i<59; i++){ // 0
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<63; i++){ // 1
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<72; i++){ // 2
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<73; i++){ // 3
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<56; i++){ // 4
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<62; i++){ // 5
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<52; i++){ // 6
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<59; i++){ // 7
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<55; i++){ // 8
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<65; i++){ // 9
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<50; i++){ // 10
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<63; i++){ // 11
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<52; i++){ // 12
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<65; i++){ // 13
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<60; i++){ // 14
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<56; i++){ // 15
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<77; i++){ // 16
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<62; i++){ // 17
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<61; i++){ // 18
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<66; i++){ // 19
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<60; i++){ // 20
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<55; i++){ // 21
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<55; i++){ // 22
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<65; i++){ // 23
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<54; i++){ // 24
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<52; i++){ // 25
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<53; i++){ // 26
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<55; i++){ // 27
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<57; i++){ // 28
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<64; i++){ // 29
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<53; i++){ // 30
            labels.put(j,0,x);
            j++;
        }
        x++;

        for(int i=0; i<56; i++){ // 31
            labels.put(j,0,x);
            j++;
        }

        return labels;
    }

    public void TestClassificazione() {

        Mat leafDataset = Test.getDatasetMat("RadiiShapeHu.csv", 51);

        Mat leafLabels = getLabelMat();

        int[] totals = new int[32];

        //conteggio del numero delle foglie per ogni classe
        for(int i=0; i<leafLabels.rows(); i++){
            int lab = (int)leafLabels.get(i, 0)[0];
            totals[lab]++;
        }

        // controprova
        int sum = 0;
        for(int i=0; i<totals.length; i++){
            sum += totals[i];
        }

        Log.d("Comunicazione", "Somma: " + sum);

        int[] toTest = new int[32];
        int[] toTrain = new int[32];
        int total;
        int toTestSum = 0;
        int toTrainSum = 0;

        //conteggio delle foglie per ogni classe da destinare a training e testing (1/10 testing, 9/10 training)
        for(int i=0; i<totals.length; i++){
            total = totals[i];
            toTest[i] = (int) (total/10);
            toTrain[i] = total - toTest[i];
            toTestSum += toTest[i];
            toTrainSum += toTrain[i];
            sum += totals[i];
        }

        Log.d("Comunicazione", "Dimensione training set: " + toTrainSum);
        Log.d("Comunicazione", "Dimensione test set: " + toTestSum);

        Mat trainingSet = new Mat();
        ArrayList<Integer> testSetLeafNumber = new ArrayList<Integer>();
        ArrayList<Integer> trainingLabels = new ArrayList<Integer>();
        Mat testLabels = new Mat();

        int totalIter = 0;
        int classIter;

        //divisione del dataset in training e test set
        for(classIter = 0; classIter<32; classIter++)
        {
            for(int j=0; j<toTrain[classIter] && totalIter<sum; j++, totalIter++)
            {
                trainingSet.push_back(leafDataset.row(totalIter));
                trainingLabels.add((int) leafLabels.get(totalIter,0)[0]);
            }
            for(int j=0; j<toTest[classIter] && totalIter<sum; j++, totalIter++)
            {
                testSetLeafNumber.add(totalIter);
                testLabels.push_back(leafLabels.row(totalIter));
            }
        }


        Mat mImgRGBA = new Mat();
        Mat prova = new Mat();

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 4;
        options.inPurgeable = true;
        options.inInputShareable = true;

        File pictureFileDir = new File(FlaviaLeavesFull);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        CvKNearest knn = new CvKNearest();
        knn.train(trainingSet, Converters.vector_int_to_Mat(trainingLabels));

        try {
            File myFile = new File(app_path + File.separator + "TestClass_Knn6.txt");
            myFile.createNewFile();
            FileOutputStream fOut = new FileOutputStream(myFile);
            OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);
            myOutWriter.write("Test classificazione" + '\n' + '\n' + '\n');

            for(int h=0; h<testSetLeafNumber.size(); h++) {

                photoPath = files[testSetLeafNumber.get(h)].getPath();
                realImage = BitmapFactory.decodeFile(photoPath, options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                Log.d("Check", (h+1) + "/178" + "   class: " + (int) testLabels.get(h,0)[0] + "   "   + photoPath);

                Utils.bitmapToMat(BitmapFactory.decodeFile(photoPath, options), prova);
                GMRsaliency salmap = new GMRsaliency(this,realImage,prova);
                Mat mMask = salmap.GetSal(1, null, 0);

                //sesta versione
                MinMaxLocResult s = Core.minMaxLoc(mMask);
                mMask.convertTo(mMask, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal)); //CV_32FC1
                Utils.matToBitmap(mMask, realImage);
                //Test.testSalvaImmagine(photoDir, photoPath, realImage);

                //binarizzazione con soglia prima del salvataggio
                Mat blur = new Mat();
                Mat mMask2 = new Mat();
                Imgproc.bilateralFilter(mMask, blur, 12,24,6);
                Imgproc.threshold(blur, mMask2,0,255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);

                //versione vecchia
                LeafProcessor leafProc = new LeafProcessor(this, mImgRGBA,mMask,mMask2);
                Mat mMask3 = leafProc.segmentLeafRG_LAB_NotCircle();
                Imgproc.cvtColor(mMask3, mMask3, Imgproc.COLOR_RGB2GRAY);

                myOutWriter.write((int) testLabels.get(h,0)[0] + "    " + photoPath + '\n');

                // CLASSIFICAZIONE

                com.example.leafrecognizer.classification.LeafProcessor leafPr =
                        new com.example.leafrecognizer.classification.LeafProcessor();

                Mat testTemp = leafPr.extractFeatures(mMask3);

                Mat test = new Mat(1, testTemp.rows(), CvType.CV_32F);

                for (int i = 0; i < testTemp.rows(); i++) {
                    test.put(0, i, testTemp.get(i, 0));
                }

                //Test.normalizzaFeatureHu(test);

                Mat res = new Mat(), neigh = new Mat(), dist = new Mat();

                float p = knn.find_nearest(test, 6, res, neigh, dist);
                Log.d("CLASSIFICAZIONE", neigh.dump());

                myOutWriter.write(neigh.dump() + '\n' + '\n');
            }

            myOutWriter.close();
            fOut.close();

        } catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(),
                    Toast.LENGTH_SHORT).show();
        }

        Toast.makeText(this, "END", Toast.LENGTH_LONG).show();
    }

    public void visualizzaSuperpixelSlic() {
        SLIC slic = new SlicBuilder().buildSLIC();

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File pictureFileDir = new File(immagini_test);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        int nFiles = files.length;

        try{
            for(int h=1; h<=nFiles; h++) {
                photoPath = files[h - 1].getPath();

                Log.d("immagine", "" + h + "/" + nFiles + "    " + photoPath);

                realImage = BitmapFactory.decodeFile(photoPath, options);

                Bitmap imageCell = slic.createBoundedBitmap(realImage);

                realImage = imageCell;

                Test.testSalvaImmagine(immagini_test_elaborate, photoPath, realImage);
            }
        }catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        Log.d("FINE", "processo ultimato");
        Toast.makeText(this, "END", Toast.LENGTH_LONG).show();
    }

    public void visualizzxaSuperpixelErs() {
        Mat mImgRGBA = new Mat();
        Mat src_gray = new Mat();

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File pictureFileDir = new File(immagini_test);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        int nFiles = files.length;

        try{
            for(int h=1; h<=nFiles; h++) {
                photoPath = files[h - 1].getPath();

                Log.d("immagine", "" + h + "/" + nFiles + "    " + photoPath);

                realImage = BitmapFactory.decodeFile(photoPath, options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                src_gray = new Mat();
                Imgproc.cvtColor(mImgRGBA, src_gray,  Imgproc.COLOR_RGBA2GRAY );
                Mat output = Test.testERS(src_gray);

                realImage = Test.testFromLabelToSuperpixel(output, mImgRGBA);

                Test.testSalvaImmagine(immagini_test_elaborate, photoPath, realImage);
            }
        }catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        Log.d("FINE", "processo ultimato");
        Toast.makeText(this, "END", Toast.LENGTH_LONG).show();
    }

    public void visualizzaWatershed() {
        Mat mImgRGBA = new Mat();

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File pictureFileDir = new File(immagini_test);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        int nFiles = files.length;

        try{
            for(int h=1; h<=nFiles; h++) {
                photoPath = files[h - 1].getPath();

                Log.d("immagine", "" + h + "/" + nFiles + "    " + photoPath);

                realImage = BitmapFactory.decodeFile(photoPath, options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                Mat im = new Mat(realImage.getWidth(),realImage.getHeight(),CvType.CV_8U);

                Mat src_gray = new Mat();
                Imgproc.cvtColor(mImgRGBA, src_gray, Imgproc.COLOR_RGBA2GRAY );

                Mat prova = new Test().testSobel(mImgRGBA);
                /*src_gray = new Watershed_Algorithm().run(prova);

                Vector<Mat> rgb = new Vector<Mat>(3);

                Utils.bitmapToMat(realImage, im);

                im.convertTo(im,CvType.CV_8UC3);

                Core.split(im, rgb);
                for (int i = 0; i < 3; i++){
                    for(int j=0; j<im.rows(); j++){
                        for(int k=0; k<im.cols(); k++) {
                            if(src_gray.get(j,k)[0] == 255)
                                rgb.get(i).put(j,k,0);
                        }
                    }
                }

                Mat outputMat = new Mat();
                Core.merge(rgb, outputMat);*/
                Utils.matToBitmap(prova, realImage);

                Test.testSalvaImmagine(immagini_test_elaborate, photoPath, realImage);
            }
        }catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
        }

        Log.d("FINE", "processo ultimato");
        Toast.makeText(this, "END", Toast.LENGTH_LONG).show();
    }

    public void segmentaSLIC() {
        Mat mImgRGBA = new Mat();
        Mat prova = new Mat();

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File pictureFileDir = new File(immagini_test);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        int nFiles = files.length;

        try {
            for(int h=1; h<=nFiles; h++) {
                photoPath = files[h-1].getPath();

                Log.d("immagine", "" + h + "/" + nFiles + "    " + photoPath);

                realImage = BitmapFactory.decodeFile(photoPath, options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                Utils.bitmapToMat(BitmapFactory.decodeFile(photoPath, options), prova);
                GMRsaliency salmap = new GMRsaliency(this,realImage,prova);
                Mat mMask = salmap.GetSal(1, null, 0);

                //sesta versione
                MinMaxLocResult s = Core.minMaxLoc(mMask);
                mMask.convertTo(mMask, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal)); //CV_32FC1
                Utils.matToBitmap(mMask, realImage);
                //Test.testSalvaImmagine(photoDir, photoPath, realImage);

                //binarizzazione con soglia prima del salvataggio
                Mat blur = new Mat();
                Mat mMask2 = new Mat();
                Imgproc.bilateralFilter(mMask, blur, 12,24,6);
                Imgproc.threshold(blur, mMask2,0,255, Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);

                //versione vecchia
                LeafProcessor leafProc = new LeafProcessor(this, mImgRGBA,mMask,mMask2);
                Mat mMask3 = leafProc.segmentLeafRG_LAB_NotCircle();

                Utils.matToBitmap(mMask3, realImage);

                Test.testSalvaImmagine(immagini_test_elaborate, photoPath, realImage);
            }
        } catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    public void segmentaERS() {
        Mat mImgRGBA = new Mat();
        Mat prova = new Mat();

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File pictureFileDir = new File(immagini_test);

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });

        int nFiles = files.length;

        try {
            for(int h=1; h<=nFiles; h++) {
                photoPath = files[h-1].getPath();

                Log.d("immagine", "" + h + "/" + nFiles + "    " + photoPath);

                realImage = BitmapFactory.decodeFile(photoPath, options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                Mat src_gray = new Mat();
                Imgproc.cvtColor(mImgRGBA, src_gray,  Imgproc.COLOR_RGBA2GRAY );
                Mat output = Test.testERS(src_gray);

                prova = Highgui.imread(photoPath);
                GMRsaliency salmap = new GMRsaliency(this,realImage,prova);
                Mat mMask = salmap.GetSal(2, output,  225);

                //sesta versione
                MinMaxLocResult s = Core.minMaxLoc(mMask);
                mMask.convertTo(mMask, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal)); //CV_32FC1

                //binarizzazione con soglia prima del salvataggio
                Mat blur = new Mat();
                Mat mMask2 = new Mat();
                Imgproc.bilateralFilter(mMask, blur, 12,24,6);
                Imgproc.threshold(blur, mMask2,0,255,Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);

                //versione vecchia
                LeafProcessor leafProc = new LeafProcessor(this, mImgRGBA,mMask,mMask2);
                Mat mMask3 = leafProc.segmentLeafRG_LAB_NotCircle();

                Utils.matToBitmap(mMask3, realImage);

                Test.testSalvaImmagine(immagini_test_elaborate, photoPath, realImage);
            }
        } catch (Exception e) {
            Toast.makeText(getBaseContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
}

