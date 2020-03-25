package com.example.leafrecognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
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
import org.opencv.ml.CvSVM;
import org.opencv.video.BackgroundSubtractorMOG2;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Color;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

public class Saliency extends ContextWrapper{
	
	private static final int  numScales =6;
	private Mat saliencyMap;
	private Mat mImgRGBA;
	//private Mat mIntensity;
	private Mat gray;
	private Mat integralImage;
	private Mat intensity;
	private Mat intensityOn;
	private Mat intensityOff;
	private Mat                  mMat0;
	private MatOfInt             mChannels[];
	private MatOfInt             mHistSize;
	private int                  mHistSizeNum = 25;
	private MatOfFloat           mRanges;
	private Mat hist;
	private Mat logP;
	Point point, P1, P2;
	Mat [] intensityScaledOn = new Mat[numScales];
	Mat [] intensityScaledOff = new Mat[numScales];
	private int width;
	private int height;
	
	
	public Saliency(Context base, Mat colorImg){
		super(base);
		
		mImgRGBA = colorImg.clone();
		
		width = mImgRGBA.rows();
		height = mImgRGBA.cols();
		
		//mIntensity = new Mat(width,height,CvType.CV_8UC1, new Scalar(0));
		saliencyMap = new Mat( width,height,CvType.CV_8UC1, new Scalar(0));
		
		for (int i=0; i<numScales;i++) {
			intensityScaledOn[i] = new Mat( width,height, CvType.CV_8UC1, new Scalar(0)); 
			intensityScaledOff[i] = new Mat(width,height,  CvType.CV_8UC1, new Scalar(0));
			}
		
		}
	
	public Mat computeSaliencyImpl() throws InterruptedException {
		if (mImgRGBA.empty())
			return null;
		if (mImgRGBA.channels() > 1)
			Imgproc.cvtColor(mImgRGBA , mImgRGBA, Imgproc.COLOR_BGR2GRAY); 

		/*Thread ciao = new Thread()
	    {
	        @Override
	        public void run() 
	        {
	        	
	        	
	        }
	    };
	    ciao.start();
	    ciao.join();*/
		
		saliencyMap = calcIntensityChannel(mImgRGBA);
		
		/*for(int r=200; r<210; r++)
		  	  for(int s=200; s<210; s++)
		  	  {
		  Log.d("check","sal"+saliencyMap.get(r,s)[0]);	
		  	  }*/
		
		return saliencyMap;
		
	}

	private Mat calcIntensityChannel(Mat src) {
		
		  if(saliencyMap.channels() > 1)
		    {
		        //("Error: saliencyMap image must have only one channel.\n");
		        return saliencyMap;
		    }
		
		gray = new Mat(src.rows(),src.cols(), CvType.CV_8UC1, new Scalar(0));
		integralImage = new Mat(src.rows()+1,src.cols()+1, CvType.CV_32FC1, new Scalar(0));
		intensity = new Mat(src.rows(),src.cols(), CvType.CV_8UC1, new Scalar(0));
		intensityOn = new Mat(src.rows(),src.cols(), CvType.CV_8UC1, new Scalar(0));
		intensityOff = new Mat(src.rows(),src.cols(), CvType.CV_8UC1, new Scalar(0));
		int i = 0;
		int neighborhood;
		int neighborhoods[] = {3*4, 3*4*2, 3*4*2*2, 7*4, 7*4*2, 7*4*2*2};
		
		
		
		// Prepare the input image: put it into a grayscale image.
	    if(src.channels()==3)
	    {
	    	Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY); //RGBA??
	    }
	    else
	    {
	    	gray=src.clone();
	    }
	    
	 // smooth pixels at least twice, as done by Frintrop and Itti
	    Imgproc.GaussianBlur(gray,gray,new Size(3,3),0,0);
	    Imgproc.GaussianBlur(gray,gray,new Size(3,3),0,0);


	    // Calculate integral image, only once.
	    Imgproc.integral(gray, integralImage, CvType.CV_32F);
	    
	    
	    
	    for (i=0; i<numScales; i++)
	    {
	    	neighborhood = neighborhoods[i];
	    	getIntensityScaled(integralImage,gray,intensityScaledOn[i],intensityScaledOff[i],neighborhood);
	    	
	    }
	    
	    
	    mixScales(intensityScaledOn, intensityOn, intensityScaledOff, intensityOff);
	    
	   mixOnOff(intensityOn, intensityOff, intensity);
	   
	   
	   /*for(int r=200; r<210; r++)
		  	  for(int s=200; s<210; s++)
		  	  {
		  Log.d("check","dst"+intensity.get(r,s)[0]);	
		  	  }*/
	    
	    return this.intensity;

	}



	private void getIntensityScaled(Mat integralImage, Mat gray, Mat intensityScaledOn,	Mat intensityScaledOff, int neighborhood) {
		
		float value, meanOff;
		float meanOn;
		point = new Point();
		int x,y;
		intensityScaledOn.setTo(new Scalar(0));
		intensityScaledOff.setTo(new Scalar(0));
		
		for (y=0; y<gray.rows();y++)
		{
			for (x=0; x<gray.cols();x++)
			{
				
				point.x=x;
				point.y=y;
				
				value=(float) getmean(integralImage, point, neighborhood, (int) gray.get(y,x)[0]);
				
				meanOn = (float) (gray.get(y, x)[0] - value);
				meanOff = (float) (value - gray.get(y, x)[0]);
				
				if(meanOn > 0)
	                intensityScaledOn.put(y,x,meanOn);
	            else
	            	intensityScaledOn.put(y,x,0);

	            if(meanOff > 0)
	            	intensityScaledOff.put(y,x,meanOff);
	            else
	            	intensityScaledOff.put(y,x,0);
	            
			}
		}
		
	}

	private double getmean(Mat src, Point pix, int neighbourhood,int centerVal) {
		
		P1 = new Point();
		P2 = new Point();
		float value = 0;
		
		P1.x = pix.x - neighbourhood + 1;
	    P1.y = pix.y - neighbourhood + 1;
	    P2.x = pix.x + neighbourhood + 1;
	    P2.y = pix.y + neighbourhood + 1;

	    if(P1.x < 0) //invertiti per prova
	        P1.x = 0;
	    else if(P1.x > src.cols() - 1)
	        P1.x = src.cols() - 1;
	    if(P2.x < 0)
	        P2.x = 0;
	    else if(P2.x > src.cols() - 1)
	        P2.x = src.cols() - 1;
	    if(P1.y < 0)
	        P1.y = 0;
	    else if(P1.y > src.rows() - 1)
	        P1.y = src.rows() - 1;
	    if(P2.y < 0)
	        P2.y = 0;
	    else if(P2.y > src.rows() - 1)
	        P2.y = src.rows() - 1;

	    // we use the integral image to compute fast features
	    value = (float) ((src.get((int) P2.y, (int)P2.x)[0]) +
	            (src.get((int)P1.y, (int)P1.x)[0]) -
	            (src.get((int)P2.y, (int)P1.x)[0]) -
	            (src.get((int)P1.y, (int)P2.x)[0]));
	    value = (float) ((value - centerVal)/  (( (P2.x - P1.x) * (P2.y - P1.y))-1))  ;
	    return value;
		
	}
	
	private void mixScales(Mat[] intensityScaledOn, Mat intensityOn, Mat[] intensityScaledOff, Mat intensityOff) {
		
		int i=0, x, y;
	    int width = intensityScaledOn[0].rows();
	    int height = intensityScaledOn[0].cols();
	    int maxValOn = 0, currValOn=0;
	    int maxValOff = 0, currValOff=0;
	    int maxValSumOff = 0, maxValSumOn=0;
	    Mat mixedValuesOn = new Mat(width,height, CvType.CV_16UC1, new Scalar(0));
	    Mat mixedValuesOff = new Mat(width,height,  CvType.CV_16UC1, new Scalar(0));
	    
	    for(i=0;i<numScales;i++)
	    {
	        for(y=0;y<width;y++)
	            for(x=0;x<height;x++)
	            {
	                      currValOn = (int) intensityScaledOn[i].get(y,x)[0];
	                      if(currValOn > maxValOn)
	                          maxValOn = currValOn;

	                      currValOff = (int) intensityScaledOn[i].get(y,x)[0];
	                      if(currValOff > maxValOff)
	                          maxValOff = currValOff;

	                      mixedValuesOn.put(y,x,currValOn+(int)mixedValuesOn.get(y,x)[0]);
	                      mixedValuesOff.put(y,x,currValOff+(int)mixedValuesOff.get(y,x)[0]);
	            }
	    }
	    
	    for(y=0;y<width;y++)
	        for(x=0;x<height;x++)
	        {
	            currValOn = (int) mixedValuesOn.get(y,x)[0];
	            currValOff = (int) mixedValuesOff.get(y,x)[0];
	                  if(currValOff > maxValSumOff)
	                      maxValSumOff = currValOff;
	                  if(currValOn > maxValSumOn)
	                      maxValSumOn = currValOn;
	        }


	    for(y=0;y<width;y++)
	        for(x=0;x<height;x++)
	        {
	            intensityOn.put(y,x,255.*((float)(mixedValuesOn.get(y, x)[0] / (float)maxValSumOn)));
	            intensityOff.put(y,x,255.*((float)(mixedValuesOff.get(y, x)[0] / (float)maxValSumOff)));
	         }
		
	}
	
	private void mixOnOff(Mat intensityOn, Mat intensityOff, Mat intensityArg) {
		int x,y;
	    int width = intensityOn.rows();
	    int height= intensityOn.cols();
	    int maxVal=0;

	    int currValOn, currValOff, maxValSumOff, maxValSumOn;

	    Mat intensity = new Mat( width,height, CvType.CV_8UC1, new Scalar(0));


	    maxValSumOff = 0;
	    maxValSumOn = 0;
	    
	    for(y=0;y<width;y++)
	        for(x=0;x<height;x++)
	        {
	            currValOn = (int) intensityOn.get(y, x)[0];
	            //Log.d("check","curvalOn "+currValOn+"y "+y+" x "+x);	
	            currValOff = (int) intensityOff.get(y, x)[0];
	            //Log.d("check","curvalOn "+currValOff+"y "+y+" x "+x);
	                  if(currValOff > maxValSumOff)
	                      maxValSumOff = currValOff;
	                  if(currValOn > maxValSumOn)
	                      maxValSumOn = currValOn;
	        }

	        if(maxValSumOn > maxValSumOff)
	            maxVal = maxValSumOn;
	        else
	            maxVal = maxValSumOff;
	        
	        for(y=0;y<width;y++)
	            for(x=0;x<height;x++)
	            {
	            	intensity.put(y,x,(255.*(((float)intensityOn.get(y, x)[0] + (float)intensityOff.get(y, x)[0]) / maxVal)));
	            	//Log.d("check","intensityOn "+intensityOn.get(y,x)[0]+ " intensityOff"+ intensityOff.get(y, x)[0]+" Maxval"+maxVal);	
	                }		
	        
	        /*for(int i=200; i<230; i++)
		    	  for(int j=200; j<230; j++)
		    	  {
		    Log.d("check","On"+intensity2.get(i,j)[0]);	
		    	  }*/

	        this.intensity = intensity.clone();
	        
	        /*for(int r=200; r<210; r++)
			  	  for(int s=200; s<210; s++)
			  	  {
			  Log.d("check","intArg"+this.intensity.get(r,s)[0]);	
			  	  }*/
		
	}
	
}
