package com.example.leafrecognizer;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.ImageView;
import com.example.leafrecognizer.R;
import com.example.leafrecognizer.classification.*;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends Activity {

    private Button cameraButton;
    private Button datasetButton;
    private Button test;
    private Button svmTest;
    //private Button camera2Button;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        //this.requestWindowFeature(Window.FEATURE_NO_TITLE);

        setContentView(R.layout.main_activity);

        cameraButton = (Button) findViewById(R.id.start_camera);

        datasetButton = (Button) findViewById(R.id.datasetButton);

        test = (Button) findViewById(R.id.test);

        svmTest = (Button) findViewById(R.id.svmTest);

        cameraButton.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                try {
                    startCameraActivity();

                } catch (Exception e) {

                }
            }
        });

        datasetButton.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                try {
                    startDatasetBuilder();

                } catch (Exception e) {

                }
            }
        });

        test.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                try {
                    startTest();

                } catch (Exception e) {

                }
            }
        });

        svmTest.setOnClickListener(new OnClickListener() {

            @Override
            public void onClick(View v) {

                try {
                    SVMTrainer.Test();

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
        Intent intent = new Intent(this, CameraActivity.class);
        //questo solo se carichi una immagine da memoria
        //Intent intent = new Intent(this, ImageProcessActivity.class);
        startActivity(intent);
    }

    public void startDatasetBuilder() {
        Intent intent = new Intent(this, DatasetBuilder.class);
        startActivity(intent);
    }

    public void startTest() {
        Intent intent = new Intent(this, ImageProcessActivity.class);
        startActivity(intent);
    }
}
