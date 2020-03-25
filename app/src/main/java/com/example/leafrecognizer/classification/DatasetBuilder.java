package com.example.leafrecognizer.classification;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import com.example.leafrecognizer.R;
import com.example.leafrecognizer.Test;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.util.Locale;

/**
 * Created by Cesare on 16/05/2016.
 */
public class DatasetBuilder extends Activity {

    private Bitmap realImage;
    String photoPath;
    String photoPath2;
    final String TAG = "Hello World";

    String sdpath = Environment.getExternalStorageDirectory().getPath();
    String app_path = sdpath + File.separator + "LeafRecog";
    String imagespath = sdpath + File.separator + "FlaviaLeavesFull";
    String imagespathSegmentate = app_path + File.separator + "SegmentazioneFlavia";
    File pictureFileDir;
    File csvFile;
    File provaFile;
    TextView textView;
    Context context;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    createDataset();
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

        setContentView(R.layout.dataset_builder);

        pictureFileDir = new File(imagespath);

        csvFile = new File(app_path + File.separator + "Features" + File.separator +"RadiiShapeHu.csv");
        context = this;
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
    }

    void appendMatToCSV(Mat leafData, FileWriter fw){
        int size = leafData.rows();
        try
        {
            for(int i=0; i<size; i++)
            {
                double[] value = leafData.get(i, 0);
                fw.append("" + value[0]);
                if(i<size-1)
                    fw.append(",");
                else
                    fw.append("\n");
            }

            fw.flush();

        }
        catch (Exception e)
        {
        }
    }


    public void createDataset() {
        //ELABORAZIONI
        Mat mImgRGBA = new Mat();

        final BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 4;
        options.inPurgeable = true;                   //Tell to gc that whether it needs free memory, the Bitmap can be cleared
        options.inInputShareable = true;              //Which kind of reference will be used to recover the Bitmap data after being clear, when it will be used in the future

        File[] files = pictureFileDir.listFiles(new FilenameFilter() {
            public boolean accept(File dir, String name) {
                return name.toLowerCase(Locale.ENGLISH).endsWith(".jpg");
            }
        });


        int nFiles = files.length;

        try
        {
            //FileWriter fw = new FileWriter(csvFile);
            for(int i=1; i<=1; i++)
            {
                Log.d("DATASET", "foglia numero: " + i + "/" + nFiles + "   " + files[i-1].getPath());
                realImage = BitmapFactory.decodeFile(files[i-1].getPath(), options);
                Utils.bitmapToMat(realImage, mImgRGBA);

                LeafProcessor leafProc = new LeafProcessor();

                Mat segmentedImage = leafProc.segmentLeafThresh(mImgRGBA, 230);

                //questa è la chiamata definitiva, quella per mettere in leafData le features
                Mat leafData = leafProc.extractFeatures(segmentedImage);

                //appendMatToCSV(leafData, fw);
            }
            //fw.close();
        }
        catch (Exception e)
        {
        }

        // normalizzazione dati Hu
/*            try
        {
            Mat normalizzato = Test.testNormalizzaHu();

            FileWriter fw = new FileWriter(csvFile);

            for(int i=0; i<normalizzato.rows(); i++)
            {
                for(int j=0; j<normalizzato.cols(); j++){
                    fw.append("" + normalizzato.get(i, j)[0]);
                    fw.append(",");
                }
                if(i!=normalizzato.rows()-1)
                    fw.append("\n");
            }

            fw.close();
        }
        catch (Exception e) {

        }*/

        Log.d("DATASET", "immagini processate");
    }
}
