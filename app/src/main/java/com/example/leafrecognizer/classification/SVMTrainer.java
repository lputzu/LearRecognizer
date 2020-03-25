package com.example.leafrecognizer.classification;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;
import android.widget.ImageView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

/**
 * Created by Cesare on 06/07/2016.
 */
public class SVMTrainer {

    static{ System.loadLibrary("opencv_java"); }

    private static final int numOfFeatures = 71;
    private static Bitmap realImage;
    private ImageView image;
    static String photoPathTest = Environment.getExternalStorageDirectory().getPath() + File.separator +
            "Full" + File.separator + "03.jpg";
    String photoPath2;
    final String TAG = "Hello World";
    final static String appPath = Environment.getExternalStorageDirectory().getPath() + File.separator + "LeafRecog";
    final String datasetPath = appPath + File.separator + "leafDataset.csv";
    final String labelPath = appPath + File.separator + "label.txt";
    final static String modelPath = appPath + File.separator + "model.xml";

    final String datasetPathJava = appPath + File.separator + "leafDatasetJava.txt";

    File pictureFileDir;
    File csvFile;


    public static void Test() {


        CvSVM svm = new CvSVM();
        svm.load(modelPath, "Modellino");


        Mat mImgRGBA = new Mat();


        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future
        realImage = BitmapFactory.decodeFile(photoPathTest, options);



        Utils.bitmapToMat(realImage, mImgRGBA);

        LeafProcessor leafProc = new LeafProcessor();
        Mat segmentedImage = leafProc.segmentLeafThresh(mImgRGBA, 230);
        //questa è la chiamata definitiva, quella per mettere in leafData le features
        Mat leafData = leafProc.extractFeatures(segmentedImage);

        Mat output = new Mat();
        Log.d("CLASSIFICAZIONE", "GGH" + svm.predict(leafData, true));



    }




    public void trainSVM()
    {
        Mat leafDataset = new Mat();
        Mat leafLabels = new Mat();


        leafDataset = getDatasetMat(datasetPath);

        //leafLabels = getLabelMat(labelPath);

        // ########################################################################################

        // inserimento label delle foglie
        leafLabels = new Mat(32,1,CvType.CV_32FC1, new Scalar(0));

        for(int i=0; i<leafLabels.rows(); i++) {
            leafLabels.put(i,0, i+1);
        }


        // ########################################################################################


        Log.i("Comunicazione", "Tutto ok, la Mat è stata creata e ha dimensioni: (" + leafDataset.rows() + ", " + leafDataset.cols());
        Log.i("Comunicazione", "Tutto ok, la Mat2 è stata creata e ha dimensioni: (" + leafLabels.rows() + ", " + leafLabels.cols());


        //creazione SVM, impostazione e training
        CvSVM svm = new CvSVM();

        CvSVMParams params = new CvSVMParams();
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.POLY);
        params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, 200, 0.00001));
        //params.set_nu(0.03);	//solo per svm_type=NU_SVC
        params.set_degree(2);   //solo per kernel_type=POLY
        //params.set_C(1);	//solo per svm_type=C_SVC


        svm.train(leafDataset, leafLabels, new Mat(), new Mat(), params);

        Log.i("Comunicazione", "Modello creato! Sto per salvarlo");

        svm.save(modelPath, "Modellino");

        Log.i("Comunicazione", "File del modello salvato! Si chiama " + (new File(modelPath).getName()));

    }

    private void tryNormalization()
    {
        //caricamento da file di dataset e labels
        Mat leafDataset = new Mat();

        leafDataset = getDatasetMat(datasetPath);

        Mat leafDatasetNorm = normalizeDatasetZScore(leafDataset);

    }

    private void testSVM(){

        //caricamento da file di dataset e labels
        Mat leafDataset = new Mat();
        Mat leafLabels = new Mat();

        leafDataset = getDatasetMat(datasetPath);
        //leafDataset = getDatasetMat(datasetPathJava);

        //leafDataset = normalizeDatasetMaxMin(leafDataset);
        //leafDataset = normalizeDatasetZScore(leafDataset);
        //leafDataset = normalizeDatasetNorma(leafDataset);

        for(int j=0; j<leafDataset.cols(); j++)
        {
            Log.i("CheckMat", "Colonna " + j + ": " + leafDataset.get(0, j)[0]);
        }

        leafLabels = getLabelMat(labelPath);

        int[] totals = new int[32];


        //conteggio del numero delle foglie per ogni classe

        for(int i=0; i<leafLabels.rows(); i++){
            int lab = (int)leafLabels.get(i, 0)[0];
            totals[lab-1]++;
        }

        // controprova (controllo somma)
        int sum = 0;
        for(int i=0; i<totals.length; i++){
            Log.i("Comunicazione", "Foglia numero " + (i+1) + ": " + totals[i] + " esemplari");
            sum += totals[i];
        }

        Log.i("Comunicazione", "Somma: " + sum);



        int[] toTest = new int[32];
        int[] toTrain = new int[32];
        int total;
        int toTestSum = 0;
        int toTrainSum = 0;

        //conteggio delle foglie per ogni classe da destinare a training e testing (1/10 testing, 9/10 training)

        for(int i=0; i<totals.length; i++){
            total = totals[i];
            toTest[i] = (int) (total/5);
            toTrain[i] = total - toTest[i];
            toTestSum += toTest[i];
            toTrainSum += toTrain[i];
            sum += totals[i];
        }

        Log.i("Comunicazione", "Dimensione training set: " + toTrainSum);
        Log.i("Comunicazione", "Dimensione test set: " + toTestSum);


        Mat trainingSet = new Mat();
        Mat testSet = new Mat();
        Mat trainingLabels = new Mat();
        Mat testLabels = new Mat();



        //COSTRUIRE TRAINING SET,  TEST SET e LABELS
        int totalIter = 0;
        int classIter;

        //divisione del dataset in training e test set

        for(classIter = 0; classIter<32; classIter++)
        {
            for(int j=0; j<toTrain[classIter] && totalIter<sum; j++, totalIter++)
            {
                trainingSet.push_back(leafDataset.row(totalIter));
                trainingLabels.push_back(leafLabels.row(totalIter));
            }
            for(int j=0; j<toTest[classIter] && totalIter<sum; j++, totalIter++)
            {
                testSet.push_back(leafDataset.row(totalIter));
                testLabels.push_back(leafLabels.row(totalIter));
            }
        }

        Log.i("Comunicazione", "Dimensioni Mat training set: (" + trainingSet.rows() + ", " + trainingSet.cols() + ")");
        Log.i("Comunicazione", "Dimensioni Mat training labels: (" + trainingLabels.rows() + ", " + trainingLabels.cols() + ")");
        Log.i("Comunicazione", "Dimensioni Mat test set: (" + testSet.rows() + ", " + testSet.cols() + ")");
        Log.i("Comunicazione", "Dimensioni Mat test labels: (" + testLabels.rows() + ", " + testLabels.cols() + ")");


        //creazione SVM, impostazione e training
        CvSVM svm = new CvSVM();

        CvSVMParams params = new CvSVMParams();
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.LINEAR);
        params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, 100, 0.00001));
        //params.set_nu(0.03);	//solo per svm_type=NU_SVC
        //params.set_degree(2);   //solo per kernel_type=POLY
        //params.set_C(1);	//solo per svm_type=C_SVC


        svm.train(trainingSet, trainingLabels, new Mat(), new Mat(), params);

        Log.i("Comunicazione", "Training effettuato!");

        int giuste = 0;
        int[] indovinate = new int[32];
        //Fase di testing
        for(int i=0; i<testSet.rows(); i++)
        {
            int prediction = (int) svm.predict(testSet.row(i));
            int real = (int)testLabels.get(i, 0)[0];

            Log.i("Comunicazione", "TEST Foglia " + (i+1) + ": classe prevista: " + prediction + " (reale: "  + real + ")");
            if (prediction == real){
                giuste++;
                indovinate[real-1]++;
            }
        }

        for(int i=0; i<32; i++)
        {
            int temp = indovinate[i]*10000/toTest[i];
            double percent = temp/100.;
            Log.i("Comunicazione", "Classe " + (i+1) + ": azzeccate " + indovinate[i] + " su " + toTest[i] + "(" + percent  + "%)");
        }
        int temp = giuste*10000/toTestSum;
        double percent = temp/100.;
        Log.i("Comunicazione", "Sono state azzeccate " + giuste + " foglie su " + toTestSum + "(" + percent  + "%)");

    }

    private void testSVM_train1(){

        //caricamento da file di dataset e labels
        Mat leafDataset = new Mat();
        Mat leafLabels = new Mat();

        leafDataset = getDatasetMat(datasetPath);
        //leafDataset = getDatasetMat(datasetPathJava);

        //leafDataset = normalizeDatasetMaxMin(leafDataset);
        //leafDataset = normalizeDatasetZScore(leafDataset);
        //leafDataset = normalizeDatasetNorma(leafDataset);

        for(int j=0; j<leafDataset.cols(); j++)
        {
            Log.i("CheckMat", "Colonna " + j + ": " + leafDataset.get(0, j)[0]);
        }

        leafLabels = getLabelMat(labelPath);

        int[] totals = new int[32];


        //conteggio del numero delle foglie per ogni classe

        for(int i=0; i<leafLabels.rows(); i++){
            int lab = (int)leafLabels.get(i, 0)[0];
            totals[lab-1]++;
        }

        // controprova (controllo somma)
        int sum = 0;
        for(int i=0; i<totals.length; i++){
            Log.i("Comunicazione", "Foglia numero " + (i+1) + ": " + totals[i] + " esemplari");
            sum += totals[i];
        }

        Log.i("Comunicazione", "Somma: " + sum);



        int[] toTest = new int[32];
        int[] toTrain = new int[32];
        int total;
        int toTestSum = 0;
        int toTrainSum = 0;

        //conteggio delle foglie per ogni classe da destinare a training e testing (1/10 testing, 9/10 training)

        for(int i=0; i<totals.length; i++){
            total = totals[i];
            toTrain[i] = 30;
            toTest[i] = total - toTrain[i];
            toTestSum += toTest[i];
            toTrainSum += toTrain[i];
            sum += totals[i];
        }

        Log.i("Comunicazione", "Dimensione training set: " + toTrainSum);
        Log.i("Comunicazione", "Dimensione test set: " + toTestSum);


        Mat trainingSet = new Mat();
        Mat testSet = new Mat();
        Mat trainingLabels = new Mat();
        Mat testLabels = new Mat();



        //COSTRUIRE TRAINING SET,  TEST SET e LABELS
        int totalIter = 0;
        int classIter;

        //divisione del dataset in training e test set

        for(classIter = 0; classIter<32; classIter++)
        {
            for(int j=0; j<toTrain[classIter] && totalIter<sum; j++, totalIter++)
            {
                trainingSet.push_back(leafDataset.row(totalIter));
                trainingLabels.push_back(leafLabels.row(totalIter));
            }
            for(int j=0; j<toTest[classIter] && totalIter<sum; j++, totalIter++)
            {
                testSet.push_back(leafDataset.row(totalIter));
                testLabels.push_back(leafLabels.row(totalIter));
            }
        }

        Log.i("Comunicazione", "Dimensioni Mat training set: (" + trainingSet.rows() + ", " + trainingSet.cols() + ")");
        Log.i("Comunicazione", "Dimensioni Mat training labels: (" + trainingLabels.rows() + ", " + trainingLabels.cols() + ")");
        Log.i("Comunicazione", "Dimensioni Mat test set: (" + testSet.rows() + ", " + testSet.cols() + ")");
        Log.i("Comunicazione", "Dimensioni Mat test labels: (" + testLabels.rows() + ", " + testLabels.cols() + ")");


        //creazione SVM, impostazione e training
        CvSVM svm = new CvSVM();

        CvSVMParams params = new CvSVMParams();
        params.set_svm_type(CvSVM.C_SVC);
        params.set_kernel_type(CvSVM.POLY);
        params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, 200, 0.00001));
        //params.set_nu(0.03);	//solo per svm_type=NU_SVC
        params.set_degree(2);   //solo per kernel_type=POLY
        //params.set_C(1);	//solo per svm_type=C_SVC


        svm.train(trainingSet, trainingLabels, new Mat(), new Mat(), params);

        Log.i("Comunicazione", "Training effettuato!");

        int giuste = 0;

        int[] indovinate = new int[32];
        //Fase di testing
        for(int i=0; i<testSet.rows(); i++)
        {
            int prediction = (int) svm.predict(testSet.row(i));
            int real = (int)testLabels.get(i, 0)[0];

            Log.i("Comunicazione", "TEST Foglia " + (i+1) + ": classe prevista: " + prediction + " (reale: "  + real + ")");
            if (prediction == real){
                giuste++;
                indovinate[real-1]++;
            }
        }

        for(int i=0; i<32; i++)
        {
            int temp = indovinate[i]*10000/toTest[i];
            double percent = temp/100.;
            Log.i("Comunicazione", "Classe " + (i+1) + ": azzeccate " + indovinate[i] + " su " + toTest[i] + "(" + percent  + "%)");
        }


        int temp = giuste*10000/toTestSum;
        double percent = temp/100.;
        Log.i("Comunicazione", "Sono state azzeccate " + giuste + " foglie su " + toTestSum + "(" + percent  + "%)");

    }

    private Mat normalizeDatasetMaxMin(Mat leafDataset)
    {
        Mat normDataset = new Mat(leafDataset.rows(), leafDataset.cols(), CvType.CV_32FC1, new Scalar(0));

        for(int j=0; j<leafDataset.cols(); j++)
        {
            double max = leafDataset.get(0, j)[0];
            double min = leafDataset.get(0, j)[0];

            for(int i=1; i<leafDataset.rows(); i++)
            {
                double val = leafDataset.get(i, j)[0];
                if(val>max)
                {
                    max = val;
                }
                if(val<min)
                {
                    min = val;
                }
            }

            double denom = max - min;

            if(denom!=0)
            {
                for(int i=0; i<leafDataset.rows(); i++)
                {
                    double val = leafDataset.get(i, j)[0];
                    double normVal = (val - min)/denom;
                    normDataset.put(i, j, normVal);
                }
            }



        }

        return normDataset;
    }

    private Mat normalizeDatasetZScore(Mat leafDataset)
    {
        Mat normDataset = new Mat(leafDataset.rows(), leafDataset.cols(), CvType.CV_32FC1, new Scalar(0));

        Log.i("Controllo", "Colonne: " + leafDataset.cols());


        for(int j=0; j<leafDataset.cols(); j++)
        {
            MatOfDouble mean = new MatOfDouble();
            MatOfDouble stddev = new MatOfDouble();

            Core.meanStdDev(leafDataset.col(j), mean, stddev);

//			 Log.i("Controllo", "(" + j + ") Media: (" + mean.rows() + ", " + mean.cols() + "), Dim. el: " + mean.get(0, 0).length);
//			 Log.i("Controllo", "(" + j + ") Dev. standard: (" + stddev.rows() + ", " + stddev.cols() + "), Dim. el: " + stddev.get(0, 0).length);
//			 Log.i("Controllo", "(" + j + ") Colonne: " + leafDataset.cols());
            //Log.i("Controllo", "(" + j + ") Media: " + mean.get(0, 0)[0] + ", Dev. standard: " + stddev.get(0, 0)[0]);

            double colMean = mean.get(0, 0)[0];
            double colStdDev = stddev.get(0, 0)[0];

            if(colStdDev!=0)
            {
                for(int i=0; i<leafDataset.rows(); i++)
                {
                    double numeratore = leafDataset.get(i, j)[0] - colMean;
                    double normVal = numeratore/colStdDev;
                    normDataset.put(i, j, normVal);
                }
            }
            else
            {
                for(int i=0; i<leafDataset.rows(); i++)
                {
                    normDataset.put(i, j, 0);
                }
            }

        }

        return normDataset;
    }

    private Mat normalizeDatasetNorma(Mat leafDataset)
    {
        Mat normDataset = new Mat(leafDataset.rows(), leafDataset.cols(), CvType.CV_32FC1, new Scalar(0));

        Log.i("Controllo", "Colonne: " + leafDataset.cols());


        for(int j=0; j<leafDataset.cols(); j++)
        {
            double norma = Core.norm(leafDataset.col(j), Core.NORM_L1);

            if(norma!=0)
            {
                for(int i=0; i<leafDataset.rows(); i++)
                {
                    double normVal = leafDataset.get(i, j)[0]/norma;
                    normDataset.put(i, j, normVal);
                }
            }

        }

        return normDataset;
    }

    Mat getDatasetMat(String path){

        Mat leafDataset;

        try
        {
            csvFile = new File(path);
            Log.i("Comunicazione", "Ok, trovato il file: " + csvFile.getName());
            FileReader fr = new FileReader(csvFile);
            BufferedReader br = new BufferedReader(fr);

            //leafDataset = new Mat(1907,71,CvType.CV_32FC1, new Scalar(0));

            leafDataset = new Mat(32, numOfFeatures, CvType.CV_32FC1, new Scalar(0));


            String newLine;
            float value;
            int i = 0;

            Log.i("Comunicazione", "Creato BufferedReader");

            while((newLine = br.readLine()) != null)
            {
                String[] leafValues = newLine.split(",");
                Log.i("Scorrendo", "Siamo alla riga numero " + i + " e il primo valore è " + Float.parseFloat(leafValues[0]));

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

    Mat getLabelMat(String path){

        Mat labelMat;

        try
        {
            csvFile = new File(path);
            Log.i("Comunicazione", "Ok, trovato il file: " + csvFile.getName());
            FileReader fr = new FileReader(csvFile);
            BufferedReader br = new BufferedReader(fr);

            labelMat = new Mat(1907,1,CvType.CV_32FC1, new Scalar(0));
            String newLine;
            float value;
            int i = 0;

            Log.i("Comunicazione", "Creato BufferedReader");

            while((newLine = br.readLine()) != null)
            {
                value = Float.parseFloat(newLine);
                labelMat.put(i, 0, value);

                i++;
            }

            br.close();
            fr.close();

        }
        catch (Exception e)
        {
            labelMat = null;
        }

        return labelMat;
    }

}
