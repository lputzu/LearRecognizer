package com.example.leafrecognizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.content.ContextWrapper;
import android.util.Log;

public class pySaliencyMap extends ContextWrapper{
	
	// parameters for computing optical flows using the Gunner Farneback's algorithm
	private static final double farne_pyr_scale = 0.5;
	private static final int farne_levels = 3;
	private static final int farne_winsize = 15;
	private static final int farne_iterations = 3;
	private static final int farne_poly_n = 5;
	private static final double farne_poly_sigma = 1.2;
	private static final int farne_flags = 0;

	// parameters for detecting local maxima
	private static final int  step_local = 8;

	// feature weights
	private static final double weight_intensity   = 0.30;
	private static final double weight_color       = 0.30;
	private static final double weight_orientation = 0.20;
	private static final double weight_motion      = 0.20;
	
	
	private static final int  numScales =6;
	
	// coefficients of Gabor filters
	private static double GaborKernel_0[][] = {

		{1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06},
		{2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05},
		{0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076},
		{0.00062494, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.00062494},
		{0.000921261, 0.006375831, -0.174308068, -0.067914552, 1, -0.067914552, -0.174308068, 0.006375831, 0.000921261},
		{0.00062494, 0.004325061, -0.118242318, -0.046070008, 0.678352526, -0.046070008, -0.118242318, 0.004325061, 0.00062494},
		{0.000195076, 0.001350077, -0.036909595, -0.014380852, 0.211749204, -0.014380852, -0.036909595, 0.001350077, 0.000195076},
		{2.80209E-05, 0.000193926, -0.005301717, -0.002065674, 0.030415784, -0.002065674, -0.005301717, 0.000193926, 2.80209E-05},
		{1.85212E-06, 1.28181E-05, -0.000350433, -0.000136537, 0.002010422, -0.000136537, -0.000350433, 1.28181E-05, 1.85212E-06}

		};
	private static double GaborKernel_45[][] = {

		{4.0418E-06, 2.2532E-05, -0.000279806, -0.001028923, 3.79931E-05, 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06},
		{2.2532E-05, 0.00092512, 0.002373205, -0.013561362, -0.0229477, 0.000389916, 0.003516954 , 0.000288732, -9.04408E-06},
		{-0.000279806, 0.002373205, 0.044837725, 0.052928748, -0.139178011, -0.108372072, 0.000847346 , 0.003516954, 0.000132863},
		{-0.001028923, -0.013561362, 0.052928748, 0.46016215, 0.249959607, -0.302454279, -0.108372072, 0.000389916, 0.000744712},
		{3.79931E-05, -0.0229477, -0.139178011, 0.249959607, 1, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05},
		{0.000744712, 0.000389916, -0.108372072, -0.302454279, 0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923},
		{0.000132863, 0.003516954, 0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806},
		{-9.04408E-06, 0.000288732, 0.003516954, 0.000389916, -0.0229477, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05},
		{-1.01551E-06, -9.04408E-06, 0.000132863, 0.000744712, 3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06}

		};
	private static double GaborKernel_90[][] = {

		{1.85212E-06, 2.80209E-05, 0.000195076, 0.00062494, 0.000921261, 0.00062494, 0.000195076, 2.80209E-05, 1.85212E-06},
		{1.28181E-05, 0.000193926, 0.001350077, 0.004325061, 0.006375831, 0.004325061, 0.001350077, 0.000193926, 1.28181E-05},
		{-0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433},
		{-0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537},
		{0.002010422, 0.030415784, 0.211749204, 0.678352526, 1, 0.678352526, 0.211749204, 0.030415784, 0.002010422},
		{-0.000136537, -0.002065674, -0.014380852, -0.046070008, -0.067914552, -0.046070008, -0.014380852, -0.002065674, -0.000136537},
		{-0.000350433, -0.005301717, -0.036909595, -0.118242318, -0.174308068, -0.118242318, -0.036909595, -0.005301717, -0.000350433},
		{1.28181E-05, 0.000193926, 0.001350077, 0.004325061, 0.006375831, 0.004325061, 0.001350077, 0.000193926, 1.28181E-05},
		{1.85212E-06, 2.80209E-05, 0.000195076, 0.00062494, 0.000921261, 0.00062494, 0.000195076, 2.80209E-05, 1.85212E-06}

		};
	private static double GaborKernel_135[][] = {

		{-1.01551E-06, -9.04408E-06, 0.000132863, 0.000744712, 3.79931E-05, -0.001028923, -0.000279806, 2.2532E-05, 4.0418E-06},
		{-9.04408E-06, 0.000288732, 0.003516954, 0.000389916, -0.0229477, -0.013561362, 0.002373205, 0.00092512, 2.2532E-05},
		{0.000132863, 0.003516954, 0.000847346, -0.108372072, -0.139178011, 0.052928748, 0.044837725, 0.002373205, -0.000279806},
		{0.000744712, 0.000389916, -0.108372072, -0.302454279, 0.249959607, 0.46016215, 0.052928748, -0.013561362, -0.001028923},
		{3.79931E-05, -0.0229477, -0.139178011, 0.249959607, 1, 0.249959607, -0.139178011, -0.0229477, 3.79931E-05},
		{-0.001028923, -0.013561362, 0.052928748, 0.46016215, 0.249959607 , -0.302454279, -0.108372072, 0.000389916, 0.000744712},
		{-0.000279806, 0.002373205, 0.044837725, 0.052928748, -0.139178011, -0.108372072, 0.000847346, 0.003516954, 0.000132863},
		{2.2532E-05, 0.00092512, 0.002373205, -0.013561362, -0.0229477, 0.000389916, 0.003516954, 0.000288732, -9.04408E-06},
		{4.0418E-06, 2.2532E-05, -0.000279806, -0.001028923, 3.79931E-05 , 0.000744712, 0.000132863, -9.04408E-06, -1.01551E-06}

		};
	
	private int width;
	private int height;
	
	private Mat saliencyMap;
	private Mat input;
	private Mat GaborKernel0;
	private Mat GaborKernel45;
	private Mat GaborKernel90;
	private Mat GaborKernel135;
	private Mat I,R,G,B;
	private Size sSize;
	List<Mat> rgb;
	Mat[] IFM;
	Mat[] GaussianMap;
	Mat[] CFM_RG;
	Mat[] CFM_BY;

	public pySaliencyMap(Context base, Mat image) {
		super(base);
		
		input = image.clone();
		width = input.cols();
		height = input.rows();
		sSize = new Size(width,height);
		
		saliencyMap = new Mat(height, width, CvType.CV_32FC1, new Scalar(0));
		GaborKernel0 = new Mat(9,9,CvType.CV_32FC1);
		GaborKernel45 = new Mat(9,9,CvType.CV_32FC1);
		GaborKernel90 = new Mat(9,9,CvType.CV_32FC1);
		GaborKernel135 = new Mat(9,9,CvType.CV_32FC1);
		I = new Mat(height, width, CvType.CV_32FC1);
		R = new Mat(height, width, CvType.CV_32FC1);
		G = new Mat(height, width, CvType.CV_32FC1);
		B = new Mat(height, width, CvType.CV_32FC1);
		
	
		for(int i=0; i<9; i++) 
			for(int j=0; j<9; j++){
				GaborKernel0.put(i, j, GaborKernel_0[i][j]); // 0 degree orientation
				GaborKernel45.put(i, j, GaborKernel_45[i][j]); // 45 degree orientation
				GaborKernel90.put(i, j, GaborKernel_90[i][j]); // 90 degree orientation
				GaborKernel135.put(i, j, GaborKernel_135[i][j]); // 135 degree orientation
				}
		}
	
	public Mat computeSalMap() {
		
		// intensity and RGB extraction
		Vector<Mat> rgb = new Vector<Mat>(3);
		Core.split(input, rgb);
		B=rgb.get(0);
		G=rgb.get(1);
		R=rgb.get(2);
		Imgproc.cvtColor(input,I,Imgproc.COLOR_BGR2GRAY);
		
		B.convertTo(B, CvType.CV_32FC1, 1.0/255.0);
		G.convertTo(G, CvType.CV_32FC1, 1.0/255.0);
		R.convertTo(R, CvType.CV_32FC1, 1.0/255.0);
		I.convertTo(I, CvType.CV_32FC1, 1.0/255.0);
		
		/*for(int i=200; i<230; i++)
	    	  for(int j=100; j<110; j++)
	    	  {
	    Log.d("colors 32f","B"+B.get(i, j)[0]+"i"+i+"j"+j);		  
	    Log.d("colors ","G"+G.get(i, j)[0]+"i"+i+"j"+j);
	    Log.d("colors ","R"+R.get(i, j)[0]+"i"+i+"j"+j);
	    Log.d("colors ","I"+I.get(i, j)[0]+"i"+i+"j"+j);	      
	    	  }*/
		
		//intensity feature maps
		Mat[] IFM = new Mat[6];
		IFMgetFM(I, IFM);
		//Log.d("check ","IFMgetFM");
		
		// color feature maps
		Mat[] CFM_RG = new Mat[6];
		Mat[] CFM_BY = new Mat[6];
		CFMGetFM(R, G, B, CFM_RG, CFM_BY);
		//Log.d("check ","CFMgetFM");
		
		// orientation feature maps
		Mat[] OFM = new Mat[24];
		OFMGetFM(I, OFM);
		//Log.d("check ","OFMgetFM");
		
		// motion feature maps
		Mat[] MFM_X = new Mat[6];
		Mat[] MFM_Y = new Mat[6];
		MFMGetFM(I, MFM_X, MFM_Y);
		//Log.d("check ","MFMgetFM");
		
		
		//=========================
		// Conspicuity Map Generation
		//=========================

		Mat ICM = ICMGetCM(IFM,sSize);
		//Log.d("check ","ICMgetCM");
		Mat CCM = CCMGetCM(CFM_RG, CFM_BY,sSize);
		//Log.d("check ","CCMgetCM");
		Mat OCM = OCMGetCM(OFM,sSize);
		//Log.d("check ","OCMgetCM");
		Mat MCM = MCMGetCM(MFM_X, MFM_Y,sSize);
		//Log.d("check ","MCMgetCM");

		//=========================
		// Saliency Map Generation
		//=========================

		// Normalize conspicuity maps
		Mat ICM_norm, CCM_norm, OCM_norm, MCM_norm;
		ICM_norm = SMNormalization(ICM);
		CCM_norm = SMNormalization(CCM);
		OCM_norm = SMNormalization(OCM);
		MCM_norm = SMNormalization(MCM);

		// Adding Intensity, Color, Orientation CM to form Saliency Map
		Core.addWeighted(ICM_norm, weight_intensity, OCM_norm, weight_orientation, 0.0, saliencyMap);
		Core.addWeighted(CCM_norm, weight_color, saliencyMap, 1.00, 0.0, saliencyMap);
		Core.addWeighted(MCM_norm, weight_motion, saliencyMap, 1.00, 0.0, saliencyMap);
	
		// Output Result Map
		saliencyMap = SMRangeNormalize(saliencyMap);
		Imgproc.blur(saliencyMap, saliencyMap,new Size(7,7)); // smoothing (if necessary)
		
		return saliencyMap;
		
	}

	private void IFMgetFM(Mat src, Mat[] dst) {
		FMGaussianPyrCSD(src, dst);
		}
	
	private void CFMGetFM(Mat R, Mat G, Mat B, Mat[] RGFM, Mat[] BYFM) {
		
		// allocate
		int height = R.rows();
		int width = R.cols();
		Mat tmp1 = new Mat(height, width, CvType.CV_32FC1);
		Mat tmp2 = new Mat(height, width, CvType.CV_32FC1);
		Mat RGBMax = new Mat(height, width, CvType.CV_32FC1);
		Mat RGMin = new Mat(height, width, CvType.CV_32FC1);
		Mat RGMat = new Mat(height, width, CvType.CV_32FC1);
		Mat BYMat = new Mat(height, width, CvType.CV_32FC1);
		
		// Max(R,G,B)
		Core.max(R, G, tmp1);
		Core.max(B, tmp1, RGBMax);
		Core.max(RGBMax, new Scalar(0.0001), RGBMax); // to prevent dividing by 0
		// Min(R,G)
		Core.min(R, G, RGMin);
		
		// R-G
		Core.subtract(R, G, tmp1);
		// B-Min(R,G)
		Core.subtract(B, RGMin, tmp2);
		// RG = (R-G)/Max(R,G,B)
		Core.divide(tmp1, RGBMax, RGMat);
		// BY = (B-Min(R,G)/Max(R,G,B)
		Core.divide(tmp2, RGBMax, BYMat);
		
		// Clamp negative value to 0 for the RG and BY maps
		Core.max(RGMat, new Scalar(0), RGMat);
		Core.max(BYMat, new Scalar(0), BYMat);

		// Obtain [RG,BY] color opponency feature map by generating Gaussian pyramid and performing center-surround difference
		FMGaussianPyrCSD(RGMat, RGFM);
		FMGaussianPyrCSD(BYMat, BYFM);
		
		// release
		/*tmp1.release();
		tmp2.release();
		RGBMax.release();
		RGMin.release();
		RGMat.release();
		BYMat.release();*/
	}
	
	private void OFMGetFM(Mat I, Mat[] dst) {
		// Create gaussian pyramid
		Mat[] GaussianI = new Mat[9];
		FMCreateGaussianPyr(I, GaussianI);

		// Convolution Gabor filter with intensity feature maps to extract orientation feature
		Mat[] tempGaborOutput0 = new Mat[9];
		Mat[] tempGaborOutput45 = new Mat[9];
		Mat[] tempGaborOutput90 = new Mat[9];
		Mat[] tempGaborOutput135 = new Mat[9];
		
		for(int j=2; j<9; j++){
			int now_height = GaussianI[j].rows();
		    int now_width = GaussianI[j].cols();
		    tempGaborOutput0[j] = new Mat(now_height, now_width, CvType.CV_32FC1);
		    tempGaborOutput45[j] = new Mat(now_height, now_width, CvType.CV_32FC1);
		    tempGaborOutput90[j] = new Mat(now_height, now_width, CvType.CV_32FC1);
		    tempGaborOutput135[j] = new Mat(now_height, now_width, CvType.CV_32FC1);
		    Imgproc.filter2D(GaussianI[j], tempGaborOutput0[j], GaussianI[j].depth(), GaborKernel0);
		    Imgproc.filter2D(GaussianI[j], tempGaborOutput45[j], GaussianI[j].depth(), GaborKernel45);
		    Imgproc.filter2D(GaussianI[j], tempGaborOutput90[j], GaussianI[j].depth(), GaborKernel90);
		    Imgproc.filter2D(GaussianI[j], tempGaborOutput135[j], GaussianI[j].depth(), GaborKernel135);
		  }
		for(int j=0; j<9; j++) GaussianI[j].release();
		
		// calculate center surround difference for each orientation
		Mat[] temp0 = new Mat[6];
		Mat[] temp45 = new Mat[6];
		Mat[] temp90 = new Mat[6];
		Mat[] temp135 = new Mat[6];
		FMCenterSurroundDiff(tempGaborOutput0, temp0);
		FMCenterSurroundDiff(tempGaborOutput45, temp45);
		FMCenterSurroundDiff(tempGaborOutput90, temp90);
		FMCenterSurroundDiff(tempGaborOutput135, temp135);
		/*for(int i=2; i<9; i++){
			tempGaborOutput0[i].release();
			tempGaborOutput45[i].release();
			tempGaborOutput90[i].release();
			tempGaborOutput135[i].release();
			}*/

		// saving the 6 center-surround difference feature map of each angle configuration to the destination pointer
		for(int i=0; i<6; i++){
		    dst[i] = temp0[i];
		    dst[i+6] = temp45[i];
		    dst[i+12] = temp90[i];
		    dst[i+18] = temp135[i];
		  }
	}

	private void MFMGetFM(Mat I, Mat[] dst_x, Mat[] dst_y) {
		int height = I.rows();
		int width = I.cols();

		// convert
		//Mat I8U = new Mat(height, width, CvType.CV_32FC1);
		//Core.convertScaleAbs(I, I8U, 256,0);

		// obtain optical flow information
		Mat flowx = new Mat(height, width, CvType.CV_32FC1, new Scalar(0));
		Mat flowy = new Mat(height, width, CvType.CV_32FC1, new Scalar(0));

		// create Gaussian pyramid
		FMGaussianPyrCSD(flowx, dst_x);
		FMGaussianPyrCSD(flowy, dst_y);
		
		//flowx.release();
		//flowy.release();
		
	}

	private void FMGaussianPyrCSD(Mat src, Mat[] dst) {
		Mat[] GaussianMap = new Mat[9];
		FMCreateGaussianPyr(src, GaussianMap);
		FMCenterSurroundDiff(GaussianMap, dst);
		//for(int i=0; i<9; i++) GaussianMap[i].release();
	}


	private void FMCreateGaussianPyr(Mat src, Mat[] dst) {
		int i;
		dst[0] = src.clone();
		for (i=1;i<9;i++)
		{
			dst[i] = new Mat((dst[i-1].rows())/2,(dst[i-1].cols())/2,CvType.CV_32FC1);
			Imgproc.pyrDown(dst[i-1],dst[i],dst[i].size());
		}
	}
	
	private void FMCenterSurroundDiff(Mat[] GaussianMap, Mat[] dst) {
		int i=0;
		  for(int s=2; s<5; s++){
		    int now_height = GaussianMap[s].rows();
		    int now_width = GaussianMap[s].cols();
		    Mat tmp = new Mat(now_height, now_width, CvType.CV_32FC1);
		    dst[i] = new Mat(now_height, now_width, CvType.CV_32FC1);
		    dst[i+1] = new Mat(now_height, now_width, CvType.CV_32FC1);
		    Imgproc.resize(GaussianMap[s+3], tmp,tmp.size());
		    Core.absdiff(GaussianMap[s], tmp, dst[i]);
		    Imgproc.resize(GaussianMap[s+4], tmp,tmp.size());
		    Core.absdiff(GaussianMap[s], tmp, dst[i+1]);
		    //tmp.release();
		    i += 2;
		  }
		
	}
	
	private void normalizeFeatureMaps(Mat[] FM, Mat[] NFM, int num_maps, int height, int width) {
		 for(int i=0; i<num_maps; i++){
			  Mat normalizedImage = SMNormalization(FM[i]);
			  NFM[i] = new Mat(height, width, CvType.CV_32FC1);
			  Imgproc.resize(normalizedImage, NFM[i],NFM[i].size());
			  //normalizedImage.release();
			}
	}

	private Mat SMNormalization(Mat src) {
		Mat result = new Mat(src.height(), src.width(), CvType.CV_32FC1);
		
		// normalize so that the pixel value lies between 0 and 1
		Mat tempresult = SMRangeNormalize(src);
		//Core.normalize(src, tempResult);
		// single-peak emphasis / multi-peak suppression
		double lmaxmean = SMAvgLocalMax(tempresult);
		double normCoeff = (1-lmaxmean)*(1-lmaxmean);
		tempresult.convertTo(result,CvType.CV_32FC1,normCoeff);
		return result;
	}

	private Mat SMRangeNormalize(Mat src) {
		//MinMaxLocResult s = Core.minMaxLoc(mMask);
	    //mMask.convertTo(mMask, CvType.CV_8U,255.0/(s.maxVal-s.minVal),-s.minVal*255.0/(s.maxVal-s.minVal)); //CV_32FC1
	    MinMaxLocResult s = Core.minMaxLoc(src);  
	    Mat result = new Mat(src.height(),src.width(),CvType.CV_32FC1);
		if(s.maxVal!=s.minVal) src.convertTo(result, CvType.CV_32FC1, 1/(s.maxVal-s.minVal), s.minVal/(s.minVal-s.maxVal));
		else src.convertTo(result, CvType.CV_32FC1, 1, -s.minVal);

		  return result;
	}
	
	private double SMAvgLocalMax(Mat src) {
		int stepsize = step_local;
		int numlocal = 0;
		double lmaxmean = 0;
		Mat dilated = new Mat();
		Imgproc.dilate(src, dilated, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(stepsize,stepsize)));
		//da controllare se fa quel che dovrebbe fare
		for (int i=0; i<src.height(); i++)
		    for (int j=0; j<src.width(); j++)
		    {
		    	
		         if (src.get(i,j)[0] == dilated.get(i,j)[0])
		         {
		        	 //Log.i("max", "src.get(i,j)[0]"+dilated.get(i,j)[0]+"i"+i+"j"+j);
		        	 numlocal++;
		        	 lmaxmean=lmaxmean+src.get(i,j)[0]; //da controllare se parliamo solo di grigi o cosa
		         }
		             
		    }  
				  
		  return lmaxmean/numlocal;
	}
	
	private Mat ICMGetCM(Mat[] IFM, Size size) {
		int num_FMs = 6;

		// Normalize all intensity feature maps
		Mat[] NIFM = new Mat[6];
		/*for (int i=0;i<6;i++)
		{
			NIFM[i] = new Mat((int)size.height, (int) size.width, CvType.CV_32FC1,new Scalar(0));
		}*/
		normalizeFeatureMaps(IFM, NIFM, num_FMs,(int)size.height, (int) size.width);

		// Formulate intensity conspicuity map by summing up the normalized intensity feature maps
		Mat ICM = new Mat((int)size.height, (int) size.width, CvType.CV_32FC1, new Scalar(0));
		for (int i=0; i<num_FMs; i++){
		  Core.add(ICM, NIFM[i], ICM);
		 // NIFM[i].release();;
		}

		return ICM;
	}
	

	private Mat CCMGetCM(Mat[] CFM_RG, Mat[] CFM_BY, Size size) {
		Mat CCM_RG = new Mat((int)size.height, (int) size.width, CvType.CV_32FC1, new Scalar(0)); 
		Mat CCM_BY = new Mat((int)size.height, (int) size.width, CvType.CV_32FC1, new Scalar(0));
		CCM_RG = ICMGetCM(CFM_RG,size);
		CCM_BY = ICMGetCM(CFM_BY,size);
		
		Mat CCM = new Mat((int)size.height, (int) size.width, CvType.CV_32FC1, new Scalar(0));
		Core.add(CCM_BY, CCM_RG, CCM);

		//CCM_BY.release();
		//CCM_RG.release();

		return CCM;
		  
	}
	
	private Mat OCMGetCM(Mat[] OFM, Size size) {
	
		int num_FMs_perAngle = 6;
		//int num_angles = 4; non usato?
		//int num_FMs = num_FMs_perAngle * num_angles; non usato?

		// split feature maps into four sets
		Mat[] OFM0 = new Mat[6];
		Mat[] OFM45 = new Mat[6];
		Mat[] OFM90 = new Mat[6];
		Mat[] OFM135 = new Mat[6];
		for (int i=0; i<num_FMs_perAngle; i++){
		  OFM0[i] = OFM[0*num_FMs_perAngle+i];
		  OFM45[i] = OFM[1*num_FMs_perAngle+i];
		  OFM90[i] = OFM[2*num_FMs_perAngle+i];
		  OFM135[i] = OFM[3*num_FMs_perAngle+i];
		}

		// extract conspicuity map for each angle
		Mat[] NOFM_tmp= new Mat[4];
		NOFM_tmp[0] = ICMGetCM(OFM0,size);
		NOFM_tmp[1] = ICMGetCM(OFM45,size);
		NOFM_tmp[2] = ICMGetCM(OFM90,size);
		NOFM_tmp[3] = ICMGetCM(OFM135,size);

		// Normalize all orientation features map grouped by their orientation angles
		Mat[] NOFM = new Mat[4];
		for (int i=0; i<4; i++){
		  //Log.d("NOFMa "," "+NOFM[i]+i);
		  NOFM[i] = SMNormalization(NOFM_tmp[i]);
		  //Log.d("NOFMb "," "+NOFM_tmp[i]+i);
		 // NOFM_tmp[i].release();
		}

		// Sum up all orientation feature maps, and form orientation conspicuity map
		Mat OCM = new Mat((int)size.height, (int) size.width, CvType.CV_32FC1,new Scalar(0));
		for(int i=0; i<4; i++){
			//Log.d("NOFMc "," "+NOFM[i]+i);
		  Core.add(NOFM[i], OCM, OCM);
		  //Log.d("NOFMd "," "+NOFM[i]+i);
		  //NOFM[i].release();
		}

		return OCM;

	}
	
	private Mat MCMGetCM(Mat[] MFM_X, Mat[] MFM_Y, Size size) {
		return CCMGetCM(MFM_X, MFM_Y,size);
	}

}
