package com.example.leafrecognizer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
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

public class LeafProcessor extends ContextWrapper{
	
	private Mat mImgRGBA;
	private Mat mImgbRGBA;
	private Mat mImgLAB;
	private Mat mGray;
	private Mat mMask;
	private Mat mMask2;
	private Mat mMaskedGray;
	private Mat mMaskedRGB;
	private Mat mCanny;
	private Mat mImgCanny;
	private Mat mBackground;
	private Mat sal;
	private Mat salb;
	MatOfPoint leafContour;
	Point centroid;
	private int occurrences;
	private int width;
	private int height;
	
	private static final int VARTHRESH = 4;
	private static final int HISTORY = 3;
	private static final double LEARNING_RATE = 0.1 ;
	
	//private CameraActivity cam;
	private static final double EPSILON = 0.000000000001;
	CvSVM svm;
	final String appPath = Environment.getExternalStorageDirectory().getPath() + File.separator + "LeafRecog";
	final String modelPath = appPath + File.separator + "model.xml";
	final String namesPath = appPath + File.separator + "leaf_names.txt";
	
	List<String> leafNames;
	private double incircle = 0;
	private double excircle = 0;
	
	public LeafProcessor(Context base, Mat colorImg, Mat mSal, Mat mSalb){
		super(base);
		leafNames = new ArrayList<String>();
		getLeafNames();
		
		mImgRGBA = colorImg.clone();
		sal = mSal.clone();
		salb = mSalb.clone();
		width = mImgRGBA.cols();
		height = mImgRGBA.rows();
		
		mGray = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
		//sal = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
		mImgbRGBA = new Mat(height, width, CvType.CV_8UC4, new Scalar(0,0,0,0));
		
		mImgLAB = new Mat();
		//from RGB to gray 
	    Imgproc.cvtColor(mImgRGBA , mGray, Imgproc.COLOR_RGB2GRAY); 
	    Imgproc.cvtColor(mImgRGBA , mImgLAB, Imgproc.COLOR_RGB2Lab); 
	    
	    //Toast.makeText(this, "Numero elementi LAB: " + mImgLAB.get(0, 0).length, Toast.LENGTH_LONG).show();

	    
	    //Toast.makeText(this, "LAB pixel (0,0): (" + mImgLAB.get(0, 0)[0] + ", " + mImgLAB.get(0, 0)[1] + ", " + mImgLAB.get(0, 0)[2] + ")", Toast.LENGTH_LONG).show();

	    
		mMaskedGray = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
		mMaskedRGB = new Mat();
		mMask = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
		mImgCanny = mCanny();
		svm = new CvSVM();
		svm.load(modelPath, "Modellino");

		
	}
	
	public Mat bgDetect() {
		Mat frame = new Mat();
		Mat back = new Mat();
		Mat fore = new Mat();
		Mat hierarchy = new Mat();
		BackgroundSubtractorMOG2 bg = new BackgroundSubtractorMOG2(HISTORY, VARTHRESH, false);;
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		
		frame = mImgRGBA.clone();
		
		// convert frame to gray scale
		Imgproc.cvtColor(frame, back, Imgproc.COLOR_RGBA2RGB);
		
		bg.apply(back, fore, LEARNING_RATE); //apply() exports a gray image by definition
		
		return fore;
	}
	
	public Mat segmentLeafThresh(int thresh)
	{
		
		Mat v = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		
	    
	      
	    //binarization (threshold=100)
	    Imgproc.threshold(mGray, mMask, thresh, 255, Imgproc.THRESH_BINARY_INV);
	    
		/*Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5));
		Mat temp = new Mat(); 

		Imgproc.resize(mGray, temp, new Size(mGray.cols()/4, mGray.rows()/4));
		Imgproc.morphologyEx(temp, temp, Imgproc.MORPH_CLOSE, kernel);
		Imgproc.resize(temp, temp, new Size(mGray.cols(), mGray.rows()));

		Core.divide(mGray, temp, temp, 1, CvType.CV_32F); // temp will now have type CV_32F
		Core.normalize(temp, mGray, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
		Imgproc.threshold(mGray, mMask, -1, 255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU); */ 
		
	    //smoothing (median filter)
	    Imgproc.medianBlur(mMask, mMask, 5);
	      
	    // leave only the bigger connected component with holes filled
	    mMask2 = mMask.clone();
	    //save all contours in mMask2
	    Imgproc.findContours(mMask2, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
	      
	    int maxAreaIdx = -1;
	    double maxArea = 0;
	    for (int idx = 0; idx < contours.size(); idx++) {
	    	  
	        Mat contour = contours.get(idx);
	        double contourArea = Imgproc.contourArea(contour);
	        if (contourArea > maxArea) {
	        	maxAreaIdx = idx;
	            maxArea = contourArea;
	        }
	    }
	      
 
	    mMask.setTo(new Scalar(0,0,0)); 
	    mMaskedGray = mMask.clone();
	   
	      
	    if(maxAreaIdx>=0){
	    	//copy the bigger connected component, directly filled, on a new black image
	    	Imgproc.drawContours( mMask, contours, maxAreaIdx, new Scalar(255, 255, 255), -1);
	    }
	    else{
	    	//ESCI!
	    }
	    
	    leafContour = contours.get(maxAreaIdx);
	      
	    // obtain the masked image
	    mGray.copyTo(mMaskedGray, mMask);
	    
	    return mMaskedGray;
	}
	
	public Mat segmentLeafRG_gray()
	{
	    //coordinate del pixel centrale dell'immagine
	    int x = height/2;
	    int y = width/2;
	    
	    //Costruzione della maschera, costituita da un cerchio centrale di larghezza 1/10 rispetto all'immagine intera
	    Mat circleMask = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
	    Mat initRegion = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
	    Core.circle(circleMask, new Point(y, x), width/15, new Scalar(255), -1);
	    //Core.rectangle(circleMask, new Point(0, 0), new Point(width-1, height-1), new Scalar(255));
	    
	    // obtain the masked image
	    mGray.copyTo(initRegion, circleMask);
	    
	    
	    //Ottenimento istogramma con maschera (per ora inutile)
	    /*MatOfInt channels = new MatOfInt(0);
	    List<Mat> inputMat = new ArrayList<Mat>();
	    inputMat.add(mGray);
	    Mat histogram = new Mat();
	    MatOfInt histSize = new MatOfInt(256);
	    MatOfFloat ranges = new MatOfFloat(0f, 256f);
	    Imgproc.calcHist(inputMat, channels, circleMask, histogram, histSize, ranges);*/
	    
	    Scalar meanRegion = Core.mean(mGray, circleMask);
	    
	    Toast.makeText(this, "Media livelli di grigio: " + meanRegion.val[0], Toast.LENGTH_LONG).show();
	    
	    //distanza massima ammissibile di un pixel dalla media della regione
	    double reg_maxdist = 40;
	    
	    //valore medio della regione
	    double reg_mean = meanRegion.val[0];
	    
	    //numero di pixel della regione
	    int reg_size = Core.countNonZero(initRegion);
	    
	    Mat dilatedCircle = new Mat();
	    Mat se = Imgproc.getStructuringElement(2, new Size(3, 3));
	    
	    Imgproc.dilate(circleMask, dilatedCircle, se);
	    
	    //Toast.makeText(this, "SE, dimensioni: (" + se.height() + ", " + se.width() + "), numero bianchi: " + Core.countNonZero(se), Toast.LENGTH_LONG).show();

	    Mat neighMask = new Mat();
	    Mat firstNeighs = new Mat();
	    Core.absdiff(dilatedCircle, circleMask, neighMask);
	    
	    //estrazione dei primi neighbors
	    mGray.copyTo(firstNeighs, neighMask);
	    Mat idx = new Mat();
	    Core.findNonZero(neighMask, idx);
	    

	    int[] coord = {0, 0};
	    idx.get(123, 0, coord);
	    

	    //Lista dei pixel contigui a quelli della regione, ancora da valutare
	    List<Neighbor> neighbors = new ArrayList<Neighbor>();
	    Mat temp = new Mat(height, width, CvType.CV_8UC1, new Scalar(2));
	    Mat flag = new Mat();
	    temp.copyTo(flag, circleMask);
	    
	    for(int i=0; i<idx.rows(); i++)
	    {
	    	idx.get(i, 0, coord);
	    	//prova.put(coord[1],  coord[0], 120);
	    	neighbors.add(new Neighbor(coord[1], coord[0], firstNeighs.get(coord[1], coord[0])[0]));
	    	flag.put(coord[1], coord[0], 1);
	    }
	    
	    // Passi per i pixel del 4-intorno
	    int neigOffsets[][] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	    boolean isGrowing = true;
	    
	    //mMask � la maschera che alla fine rappresenter� la regione segmentata
	    circleMask.copyTo(mMask);
	    
/*	    int cont=0;
	    for(Neighbor nb : neighbors)
        {
	    	if (cont==200)
	    	    Toast.makeText(this, "Ci sono arrivato al 200esimo", Toast.LENGTH_LONG).show();

	    	cont++;
        }*/
	    
	    // finch� la distanza tra il possibile nuovo pixel e l'intensit� media della regione � inferiore a 10 && il numero di pixel della regione non ha raggiunto quella dell'immagine
	    while(reg_size < width*height && isGrowing && !neighbors.isEmpty())
	    {
		    List<Neighbor> toRemove = new ArrayList<Neighbor>();
		    List<Neighbor> toAdd = new ArrayList<Neighbor>();
	    	isGrowing=false;
	    	//ListIterator<Neighbor> itr = neighbors.listIterator();
	        for(Neighbor nb : neighbors)
	        {
	        	double dist = Math.abs(nb.value - reg_mean);
	        	if (dist <= reg_maxdist)
	        	{
	        		isGrowing=true;
	        		//aggiungi il pixel alla regione
	        		mMask.put(nb.x, nb.y, 255);
	        		//flagga il pixel come appartenente alla regione
	        		flag.put(nb.x, nb.y, 2);
	        		
	        		
	        		//aggiorna la media della regione
	        		reg_mean = (reg_mean * reg_size + nb.value)/(reg_size + 1);
	        		
	        		//incrementa il numero di pixel della regione
	        		reg_size++;
	        		
	        		//aggiungi alla lista dei neighbors i 4 pixel adiacenti a questo neighbor
	        		for (int j=0; j<4; j++)
	        	    {
	        			//calcola la coordinata
	        			int xn = nb.x + neigOffsets[j][0];
	        			int yn = nb.y + neigOffsets[j][1];

	        			// Controlla se il pixel � dentro l'immagine
	        			boolean ins = (xn >= 0) && (yn >= 0) && (xn < height) && (yn < width);

	        			// se il pixel � dentro l'immagine e non � gi� stato considerato
	        			if(ins && flag.get(xn, yn)[0] == 0)
	        			{
	        				//neg_list � una lista di info, una per ogni pixel: coordinata x, coordinata y, livello di intensit� del pixel													
	        				toAdd.add(new Neighbor(xn, yn, mGray.get(xn, yn)[0]));
	        				//viene anche indicato che ora quel pixel fa parte della lista dei neighbors
	        				flag.put(xn, yn, 1); 
	        			}
	        	    }
	        		
	        		//togli questo neighbor dalla lista neighbors
	        		toRemove.add(nb);

	        	}
	        	
	        }
	        
	        neighbors.addAll(toAdd);
	        neighbors.removeAll(toRemove);
	    }
	    
	    //********RICORDARSI DI FARE L'HOLE FILLING DELLA MASCHERA (mBin)************
	    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	    Mat v = new Mat();
	    Imgproc.findContours(mMask, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
	    Imgproc.drawContours(mMask, contours, 0, new Scalar(255, 255, 255), -1);
	    leafContour = contours.get(0);


	    Mat mMasked = new Mat();
	    // obtain the masked image
	    mGray.copyTo(mMasked, mMask);
	    Imgproc.drawContours(mMasked, contours, 0, new Scalar(255, 255, 255));

	    return mMasked;
	    //return circleMask;
	}
	
	public double scalarEuclDistance(Scalar s1, Scalar s2, Scalar w)
	{
		int i = 0;
		double sum = 0;
		while(i < s1.val.length && i < s2.val.length)
		{
			sum += w.val[i] * Math.pow(Math.abs(s1.val[i] - s2.val[i]), 2);
			i++;
		}
		return Math.sqrt( sum ); 
	}
	
	
	public Mat segmentLeafRG_RGB()
	{
		
	    //coordinate del pixel centrale dell'immagine
	    int x = height/2;
	    int y = width/2;
	    
	    //Costruzione della maschera, costituita da un cerchio centrale di larghezza 1/10 rispetto all'immagine intera
	    Mat circleMask = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
	    Mat initRegion = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));
	    Core.circle(circleMask, new Point(y, x), width/15, new Scalar(255), -1);
	    //Core.rectangle(circleMask, new Point(0, 0), new Point(width-1, height-1), new Scalar(255));
	    
	    // obtain the masked image
	    mImgRGBA.copyTo(initRegion, circleMask);
	    
	    Scalar meanRegion = Core.mean(mImgRGBA, circleMask);
	    
	    
	    //Toast.makeText(this, "Media RGB: (" + meanRegion.val[0] + ", " + meanRegion.val[1] + ", " + meanRegion.val[2] + ")", Toast.LENGTH_LONG).show();
	    //Toast.makeText(this, "Distanza: " + scalarEuclDistance(new Scalar(30, 30, 30), new Scalar(33, 30, 34)), Toast.LENGTH_LONG).show();
	    

	    //distanza massima ammissibile di un pixel dalla media della regione
	    double reg_maxdist = 80;
	    
	    
	    //numero di pixel della regione
	    int reg_size = Core.countNonZero(circleMask);
	    
	    Mat dilatedCircle = new Mat();
	    Mat se = Imgproc.getStructuringElement(2, new Size(3, 3));
	    
	    Imgproc.dilate(circleMask, dilatedCircle, se);
	    
	    //Toast.makeText(this, "SE, dimensioni: (" + se.height() + ", " + se.width() + "), numero bianchi: " + Core.countNonZero(se), Toast.LENGTH_LONG).show();

	    Mat neighMask = new Mat();
	    Mat firstNeighs = new Mat();
	    Core.absdiff(dilatedCircle, circleMask, neighMask);
	    
	    //estrazione dei primi neighbors
	    mImgRGBA.copyTo(firstNeighs, neighMask);
	    Mat idx = new Mat();
	    Core.findNonZero(neighMask, idx);
	    

	    int[] coord = {0, 0};	    

	    //Lista dei pixel contigui a quelli della regione, ancora da valutare
	    List<colNeighbor> neighbors = new ArrayList<colNeighbor>();
	    Mat temp = new Mat(height, width, CvType.CV_8UC1, new Scalar(2));
	    Mat flag = new Mat();
	    temp.copyTo(flag, circleMask);
	    
	    for(int i=0; i<idx.rows(); i++)
	    {
	    	idx.get(i, 0, coord);
	    	//prova.put(coord[1],  coord[0], 120);
	    	Scalar sc = new Scalar(firstNeighs.get(coord[1], coord[0])[0], firstNeighs.get(coord[1], coord[0])[1], firstNeighs.get(coord[1], coord[0])[2]);
	    	neighbors.add(new colNeighbor(coord[1], coord[0], sc));
	    	flag.put(coord[1], coord[0], 1);
	    }
	    
	    // Passi per i pixel del 4-intorno
	    int neigOffsets[][] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	    boolean isGrowing = true;
	    
	    //mMask � la maschera che alla fine rappresenter� la regione segmentata
	    circleMask.copyTo(mMask);
	    
	    
	    // finch� la distanza tra il possibile nuovo pixel e l'intensit� media della regione � inferiore a 10 && il numero di pixel della regione non ha raggiunto quella dell'immagine
	    while(reg_size < width*height && isGrowing && !neighbors.isEmpty())
	    {
		    List<colNeighbor> toRemove = new ArrayList<colNeighbor>();
		    List<colNeighbor> toAdd = new ArrayList<colNeighbor>();
	    	isGrowing=false;
	    	//ListIterator<colNeighbor> itr = neighbors.listIterator();
	        for(colNeighbor nb : neighbors)
	        {
	        	double dist = scalarEuclDistance(nb.value, meanRegion, new Scalar(1, 1, 1));
	        	int ex = nb.x;
	        	int ey = nb.y;
	        	boolean noedge = (mImgCanny.get(ex, ey)[0] == 0 && mImgCanny.get(ex, ey)[0] == 0 && mImgCanny.get(ex, ey)[0] == 0);
	        	//if (dist <= reg_maxdist)
	        	if (dist <= reg_maxdist && noedge)
	        	{
	        		isGrowing=true;
	        		//aggiungi il pixel alla regione
	        		mMask.put(nb.x, nb.y, 255);
	        		//flagga il pixel come appartenente alla regione
	        		flag.put(nb.x, nb.y, 2);
	        		
	        		
	        		//aggiorna la media della regione
	        		double newMeanR = (meanRegion.val[0]*reg_size + nb.value.val[0])/(reg_size + 1);
	        		double newMeanG = (meanRegion.val[1]*reg_size + nb.value.val[1])/(reg_size + 1);
	        		double newMeanB = (meanRegion.val[2]*reg_size + nb.value.val[2])/(reg_size + 1);
	        		meanRegion = new Scalar(newMeanR, newMeanG, newMeanB); 
	        		
	        		//incrementa il numero di pixel della regione
	        		reg_size++;
	        		
	        		//aggiungi alla lista dei neighbors i 4 pixel adiacenti a questo neighbor
	        		for (int j=0; j<4; j++)
	        	    {
	        			//calcola la coordinata
	        			int xn = nb.x + neigOffsets[j][0];
	        			int yn = nb.y + neigOffsets[j][1];

	        			// Controlla se il pixel � dentro l'immagine
	        			boolean ins = (xn >= 0) && (yn >= 0) && (xn < height) && (yn < width);

	        			// se il pixel � dentro l'immagine e non � gi� stato considerato
	        			if(ins && flag.get(xn, yn)[0] == 0 && noedge)
	        			{
	        				//neg_list � una lista di info, una per ogni pixel: coordinata x, coordinata y, livello di intensit� del pixel													
	        				double[] rgbVal = mImgRGBA.get(xn, yn);
	        				toAdd.add(new colNeighbor(xn, yn, new Scalar(rgbVal[0], rgbVal[1], rgbVal[2])));
	        				//viene anche indicato che ora quel pixel fa parte della lista dei neighbors
	        				flag.put(xn, yn, 1); 
	        			}
	        	    }
	        		
	        		//togli questo neighbor dalla lista neighbors
	        		toRemove.add(nb);

	        	}
	        	
	        }
	        
	        neighbors.addAll(toAdd);
	        neighbors.removeAll(toRemove);
	    }
	    
	    //********RICORDARSI DI FARE L'HOLE FILLING DELLA MASCHERA (mBin)************
	    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	    Mat v = new Mat();
	    Imgproc.findContours(mMask, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
	    Imgproc.drawContours(mMask, contours, 0, new Scalar(255, 255, 255), -1);
	    leafContour = contours.get(0);


	    // obtain the masked image
	    mImgRGBA.copyTo(mMaskedRGB, mMask);
	    Imgproc.drawContours(mMaskedRGB, contours, 0, new Scalar(255, 255, 255));
	    
	    Toast.makeText(this, "Dimensione regione: " + reg_size, Toast.LENGTH_LONG).show();

	    return mMaskedRGB;
	    //return circleMask;
	    
	    //return initRegion;
	}
	
	public Mat mCanny()
	{
		mCanny = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
		Imgproc.blur(mImgRGBA,mImgbRGBA,new Size(3,3));
		Imgproc.cvtColor(mImgbRGBA , mCanny, Imgproc.COLOR_BGRA2GRAY, 4);
		Imgproc.Canny(mCanny, mCanny, 20, 35);
		//ancora da trovare il settaggio migliore
	    return mCanny;
	}


    public Mat segmentLeafRG_LAB_NotCircle() {
        //coordinate del pixel centrale del rettangolo di partenza
        int x = height/2;
        int y = width/2;


        Mat circleMask = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
        Mat initRegion = new Mat();



        for(int i=0; i<sal.rows(); i++) {
            for(int j=0; j<sal.cols(); j++) {
                if(sal.get(i,j)[0]==255)
                    circleMask.put(i,j, 255);
                else
                    circleMask.put(i,j, 0);
            }
        }



        // obtain the masked image
        mImgRGBA.copyTo(initRegion, circleMask);

        Scalar meanRegion = Core.mean(mImgLAB, circleMask);


        //Toast.makeText(this, "Media LAB: (" + meanRegion.val[0] + ", " + meanRegion.val[1] + ", " + meanRegion.val[2] + ")", Toast.LENGTH_LONG).show();
        //Toast.makeText(this, "Distanza: " + scalarEuclDistance(new Scalar(30, 30, 30), new Scalar(33, 30, 34), new Scalar(0, 1, 1)), Toast.LENGTH_LONG).show();


        //distanza massima ammissibile di un pixel dalla media della regione 18 originale
        double reg_maxdist = 35;


        //numero di pixel della regione
        int reg_size = Core.countNonZero(circleMask);

        Toast.makeText(this, "Dimensione regione: " + reg_size, Toast.LENGTH_LONG).show();

        Mat dilatedCircle = new Mat();
        Mat se = Imgproc.getStructuringElement(2, new Size(3, 3));

        Imgproc.dilate(circleMask, dilatedCircle, se);

        //Toast.makeText(this, "SE, dimensioni: (" + se.height() + ", " + se.width() + "), numero bianchi: " + Core.countNonZero(se), Toast.LENGTH_LONG).show();

        Mat neighMask = new Mat();
        Mat firstNeighs = new Mat();
        Core.absdiff(dilatedCircle, circleMask, neighMask);

        //estrazione dei primi neighbors
        mImgLAB.copyTo(firstNeighs, neighMask);
        Mat idx = new Mat();
        Core.findNonZero(neighMask, idx);


        int[] coord = {0, 0};

        //Lista dei pixel contigui a quelli della regione, ancora da valutare
        List<colNeighbor> neighbors = new ArrayList<colNeighbor>();
        Mat temp = new Mat(height, width, CvType.CV_8UC1, new Scalar(2));
        Mat flag = new Mat();
        temp.copyTo(flag, circleMask);

        for(int i=0; i<idx.rows(); i++)
        {
            idx.get(i, 0, coord);
            //prova.put(coord[1],  coord[0], 120);
            Scalar sc = new Scalar(firstNeighs.get(coord[1], coord[0])[0], firstNeighs.get(coord[1], coord[0])[1], firstNeighs.get(coord[1], coord[0])[2]);
            neighbors.add(new colNeighbor(coord[1], coord[0], sc));
            flag.put(coord[1], coord[0], 1);
        }

        // Passi per i pixel del 4-intorno
        int neigOffsets[][] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        boolean isGrowing = true;

        //mMask � la maschera che alla fine rappresenter� la regione segmentata
        circleMask.copyTo(mMask);

        // finch� la distanza tra il possibile nuovo pixel e l'intensit� media della regione �
        //inferiore  a 10 && il numero di pixel della regione non ha raggiunto quella dell'immagine
        while(reg_size < width*height && isGrowing && !neighbors.isEmpty())
        {
            List<colNeighbor> toRemove = new ArrayList<colNeighbor>();
            List<colNeighbor> toAdd = new ArrayList<colNeighbor>();
            isGrowing=false;
            //ListIterator<colNeighbor> itr = neighbors.listIterator();
            for(colNeighbor nb : neighbors)
            {      // scalare originale (0,1,1)
                double dist = scalarEuclDistance(nb.value, meanRegion, new Scalar(0, 1, 1));
                //o l'uno o l'altro in questo caso uso la cosa binarizzata
                //if (salb.get(nb.x,nb.y)[0]==0)
                //reg_maxdist = 15;		//30   //ibrida 30-100	//proporzionale 35
                //else
                //	reg_maxdist = 35;		//70	//doppia soglia 15-35
                dist=dist/(sal.get(nb.x,nb.y)[0]/255);
                //Log.i("check", "sal: " + sal.get(nb.x,nb.y)[0]);
                //int ex = nb.x;
                //int ey = nb.y;
                //boolean noedge = (mImgCanny.get(ex, ey)[0] == 0);
                if (dist <= reg_maxdist)
                //if ((dist <= reg_maxdist) && noedge)
                {
                    isGrowing=true;
                    //aggiungi il pixel alla regione
                    mMask.put(nb.x, nb.y, 255);
                    //flagga il pixel come appartenente alla regione
                    flag.put(nb.x, nb.y, 2);


                    //aggiorna la media della regione
                    double newMeanL = (meanRegion.val[0]*reg_size + nb.value.val[0])/(reg_size + 1);
                    double newMeanA = (meanRegion.val[1]*reg_size + nb.value.val[1])/(reg_size + 1);
                    double newMeanB = (meanRegion.val[2]*reg_size + nb.value.val[2])/(reg_size + 1);
                    meanRegion = new Scalar(newMeanL, newMeanA, newMeanB);

                    //incrementa il numero di pixel della regione
                    reg_size++;

                    //aggiungi alla lista dei neighbors i 4 pixel adiacenti a questo neighbor
                    for (int j=0; j<4; j++)
                    {
                        //calcola la coordinata
                        int xn = nb.x + neigOffsets[j][0];
                        int yn = nb.y + neigOffsets[j][1];
                        //boolean nnoedge = false;
                        // Controlla se il pixel � dentro l'immagine
                        boolean ins = (xn >= 0) && (yn >= 0) && (xn < height) && (yn < width);
                        //if (ins) {
                        //nnoedge = (mImgCanny.get(xn, yn)[0] == 0); }
                        // se il pixel � dentro l'immagine e non � gi� stato considerato
                        if(ins && flag.get(xn, yn)[0] == 0)// && nnoedge)
                        {
                            //neg_list � una lista di info, una per ogni pixel: coordinata x, coordinata y, livello di intensit� del pixel
                            double[] labVal = mImgLAB.get(xn, yn);
                            toAdd.add(new colNeighbor(xn, yn, new Scalar(labVal[0], labVal[1], labVal[2])));
                            //viene anche indicato che ora quel pixel fa parte della lista dei neighbors
                            flag.put(xn, yn, 1);
                        }
                    }

                    //togli questo neighbor dalla lista neighbors
                    toRemove.add(nb);

                }

            }

            neighbors.addAll(toAdd);
            neighbors.removeAll(toRemove);
        }

        //********RICORDARSI DI FARE L'HOLE FILLING DELLA MASCHERA (mBin)************
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat v = new Mat();
        Imgproc.findContours(mMask, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        Imgproc.drawContours(mMask, contours, 0, new Scalar(255, 255, 255), -1);
        leafContour = contours.get(0);

        // obtain the masked image
        mImgRGBA.copyTo(mMaskedRGB, mMask);
        // obtain the masked image gray
        mGray.copyTo(mMaskedGray, mMask);
        Imgproc.drawContours(mMaskedRGB, contours, 0, new Scalar(255, 255, 255));


        return mMaskedRGB;
        //return circleMask;
        //return initRegion;
    }


	public Mat segmentLeafRG_LAB() {
	    //coordinate del pixel centrale del rettangolo di partenza
	    int x = height/2;
	    int y = width/2;


        Mat circleMask = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
        Mat initRegion = new Mat();


	    Core.circle(circleMask, new Point(y, x), width/15, new Scalar(255), -1);
	    //Core.rectangle(circleMask, new Point(0, 0), new Point(width-1, height-1), new Scalar(255))


	    // obtain the masked image
	    mImgRGBA.copyTo(initRegion, circleMask);
	    
	    Scalar meanRegion = Core.mean(mImgLAB, circleMask);
	    
	    
	    //Toast.makeText(this, "Media LAB: (" + meanRegion.val[0] + ", " + meanRegion.val[1] + ", " + meanRegion.val[2] + ")", Toast.LENGTH_LONG).show();
	    //Toast.makeText(this, "Distanza: " + scalarEuclDistance(new Scalar(30, 30, 30), new Scalar(33, 30, 34), new Scalar(0, 1, 1)), Toast.LENGTH_LONG).show();
	    

	    //distanza massima ammissibile di un pixel dalla media della regione 18 originale
	   	double reg_maxdist = 30;
	    
	    
	    //numero di pixel della regione
	    int reg_size = Core.countNonZero(circleMask);
	    
	    Toast.makeText(this, "Dimensione regione: " + reg_size, Toast.LENGTH_LONG).show();
	    
	    Mat dilatedCircle = new Mat();
	    Mat se = Imgproc.getStructuringElement(2, new Size(3, 3));
	    
	    Imgproc.dilate(circleMask, dilatedCircle, se);
	    
	    //Toast.makeText(this, "SE, dimensioni: (" + se.height() + ", " + se.width() + "), numero bianchi: " + Core.countNonZero(se), Toast.LENGTH_LONG).show();

	    Mat neighMask = new Mat();
	    Mat firstNeighs = new Mat();
	    Core.absdiff(dilatedCircle, circleMask, neighMask);
	    
	    //estrazione dei primi neighbors
	    mImgLAB.copyTo(firstNeighs, neighMask);
	    Mat idx = new Mat();
	    Core.findNonZero(neighMask, idx);
	    

	    int[] coord = {0, 0};	    

	    //Lista dei pixel contigui a quelli della regione, ancora da valutare
	    List<colNeighbor> neighbors = new ArrayList<colNeighbor>();
	    Mat temp = new Mat(height, width, CvType.CV_8UC1, new Scalar(2));
	    Mat flag = new Mat();
	    temp.copyTo(flag, circleMask);
	    
	    for(int i=0; i<idx.rows(); i++)
	    {
	    	idx.get(i, 0, coord);
	    	//prova.put(coord[1],  coord[0], 120);
	    	Scalar sc = new Scalar(firstNeighs.get(coord[1], coord[0])[0], firstNeighs.get(coord[1], coord[0])[1], firstNeighs.get(coord[1], coord[0])[2]);
	    	neighbors.add(new colNeighbor(coord[1], coord[0], sc));
	    	flag.put(coord[1], coord[0], 1);
	    }
	    
	    // Passi per i pixel del 4-intorno
	    int neigOffsets[][] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

	    boolean isGrowing = true;
	    
	    //mMask � la maschera che alla fine rappresenter� la regione segmentata
	    circleMask.copyTo(mMask);
	    
	    // finch� la distanza tra il possibile nuovo pixel e l'intensit� media della regione � 
	    //inferiore  a 10 && il numero di pixel della regione non ha raggiunto quella dell'immagine
	    while(reg_size < width*height && isGrowing && !neighbors.isEmpty())
	    {
		    List<colNeighbor> toRemove = new ArrayList<colNeighbor>();
		    List<colNeighbor> toAdd = new ArrayList<colNeighbor>();
	    	isGrowing=false;
	    	//ListIterator<colNeighbor> itr = neighbors.listIterator();
	        for(colNeighbor nb : neighbors)
	        {      // scalare originale (0,1,1)
	        	double dist = scalarEuclDistance(nb.value, meanRegion, new Scalar(0, 1, 1));
	        	//o l'uno o l'altro in questo caso uso la cosa binarizzata
	        	//if (salb.get(nb.x,nb.y)[0]==0)
	        		//reg_maxdist = 15;		//30   //ibrida 30-100	//proporzionale 35
	        	//else
	        	//	reg_maxdist = 35;		//70	//doppia soglia 15-35
	        	dist=dist/(sal.get(nb.x,nb.y)[0]/255);
	        	//Log.i("check", "sal: " + sal.get(nb.x,nb.y)[0]);
	        	//int ex = nb.x;
	        	//int ey = nb.y;
	        	//boolean noedge = (mImgCanny.get(ex, ey)[0] == 0);
	        	if (dist <= reg_maxdist)
	        	//if ((dist <= reg_maxdist) && noedge)
	        	{
	        		isGrowing=true;
	        		//aggiungi il pixel alla regione
	        		mMask.put(nb.x, nb.y, 255);
	        		//flagga il pixel come appartenente alla regione
	        		flag.put(nb.x, nb.y, 2);
	        		
	        		
	        		//aggiorna la media della regione
	        		double newMeanL = (meanRegion.val[0]*reg_size + nb.value.val[0])/(reg_size + 1);
	        		double newMeanA = (meanRegion.val[1]*reg_size + nb.value.val[1])/(reg_size + 1);
	        		double newMeanB = (meanRegion.val[2]*reg_size + nb.value.val[2])/(reg_size + 1);
	        		meanRegion = new Scalar(newMeanL, newMeanA, newMeanB); 
	        		
	        		//incrementa il numero di pixel della regione
	        		reg_size++;
	        		
	        		//aggiungi alla lista dei neighbors i 4 pixel adiacenti a questo neighbor
	        		for (int j=0; j<4; j++)
	        	    {
	        			//calcola la coordinata
	        			int xn = nb.x + neigOffsets[j][0];
	        			int yn = nb.y + neigOffsets[j][1];
	        			//boolean nnoedge = false;
	        			// Controlla se il pixel � dentro l'immagine
	        			boolean ins = (xn >= 0) && (yn >= 0) && (xn < height) && (yn < width);
	        			//if (ins) {
	        			//nnoedge = (mImgCanny.get(xn, yn)[0] == 0); }
	        			// se il pixel � dentro l'immagine e non � gi� stato considerato
	        			if(ins && flag.get(xn, yn)[0] == 0)// && nnoedge)
	        			{
	        				//neg_list � una lista di info, una per ogni pixel: coordinata x, coordinata y, livello di intensit� del pixel													
	        				double[] labVal = mImgLAB.get(xn, yn);
	        				toAdd.add(new colNeighbor(xn, yn, new Scalar(labVal[0], labVal[1], labVal[2])));
	        				//viene anche indicato che ora quel pixel fa parte della lista dei neighbors
	        				flag.put(xn, yn, 1); 
	        			}
	        	    }
	        		
	        		//togli questo neighbor dalla lista neighbors
	        		toRemove.add(nb);

	        	}
	        	
	        }
	        
	        neighbors.addAll(toAdd);
	        neighbors.removeAll(toRemove);
	    }
	    
	    //********RICORDARSI DI FARE L'HOLE FILLING DELLA MASCHERA (mBin)************
	    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	    Mat v = new Mat();
	    Imgproc.findContours(mMask, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
	    Imgproc.drawContours(mMask, contours, 0, new Scalar(255, 255, 255), -1);
	    leafContour = contours.get(0);

	    // obtain the masked image
	    mImgRGBA.copyTo(mMaskedRGB, mMask);
	    // obtain the masked image gray
	    mGray.copyTo(mMaskedGray, mMask);
	    Imgproc.drawContours(mMaskedRGB, contours, 0, new Scalar(255, 255, 255));

	    
	    return mMaskedRGB;
	    //return circleMask;
	    //return initRegion;
	}
	
	public Mat getMask()
	{
		return mMask;
	}
	
	public Mat getMaskedRGB()
	{
		return mMaskedRGB;
	}
	
	public Mat getContour()
	{
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
	    Mat v = new Mat();
	    Mat contourImage = new Mat(height, width, CvType.CV_8UC1, new Scalar(0));
	    Imgproc.findContours(mMask, contours, v, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
	    Imgproc.drawContours(contourImage, contours, 0, new Scalar(255, 255, 255), 2);
	    
	    return contourImage;
	    
	}
	
	public Mat extractFeatures(){
		
		Mat leafDataMat = new Mat(1, 71, CvType.CV_32FC1, new Scalar(0));
		
	    centroid = new Point();
	      
	      
	    //compute the center of mass
	    Moments moments = Imgproc.moments(mMask, true);
	    
	    centroid.x = moments.get_m10() / moments.get_m00();
	    centroid.y = moments.get_m01() / moments.get_m00();
		
	    
	    
	    double[] radii = centroidRadii();
	    
	    double[] shapeFeatures = shapeDescriptor();
	    
	    Mat hu = new Mat();
	    Imgproc.HuMoments(moments, hu);
	    double[] huMoments = new double[hu.rows()];
	    for(int i = 0; i < hu.rows(); i++){
	    	double[] moment = hu.get(i, 0);
	    	huMoments[i] = moment[0];
	    }
	    
		//Toast.makeText(this, "Momenti: " + huMoments[0] + ", " + huMoments[1] + ", " + huMoments[2] + ", " + huMoments[3] + ", " + huMoments[4] + ", " + huMoments[5] + ", " + huMoments[6], Toast.LENGTH_LONG).show();

	    double[] textureFeatures = getTextureFeaturesRidotto(true);

	    
	    //metto tutto nel Mat
		int j = 0;
		for(int k = 0; k < 36 && j < 71; k++, j++){
			leafDataMat.put(0, j, radii[k]);
		}
		for(int k = 0; k < 8 && j < 71; k++, j++){
			leafDataMat.put(0, j, shapeFeatures[k]);
		}
		for(int k = 0; k < 7 && j < 71; k++, j++){
			leafDataMat.put(0, j, huMoments[k]);
		}
		for(int k = 0; k < 80 && j < 71; k++, j++){
			leafDataMat.put(0, j, textureFeatures[k]);
		}
		
		//controllo visivo su maschera, convex hull e bounding box
		//return mMaskedGray;
		
		return leafDataMat;

	}
	
	public float classifyLeaf(Mat features)
	{
		return svm.predict(features);
	}
	
	private double[] centroidRadii(){
		//Mat cRadii = new Mat(36, 1, CvType.CV_32FC1, new Scalar(0, 0, 0));
		
		
		Point[] contourPoints = leafContour.toArray();
		//Toast.makeText(this, "Riga: " + contourPoints[0].x + ", Colonna: " + contourPoints[0].y, Toast.LENGTH_LONG).show();
		
		int dim = contourPoints.length;
		double[] fullRadii = new double[dim];
		double[] radii = new double[36];
		float range = dim/36;
		double min = 100000;
		double max = 0;
		double sum;
		int pos = 0;
		int start = 0;
		int stop = 0;

		for (int i = 0; i < dim; i++) 	{	
			fullRadii[i] = euclDist(contourPoints[i], centroid);
			if (fullRadii[i] > max){
				max = fullRadii[i];
				pos = i;
			}
			if (fullRadii[i] < min)
				min = fullRadii[i];
		}
		
		excircle = max;
		incircle = min;
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
			//i raggi vengono normalizzati
			radii[i] /= max;
			start = stop;	
		}	
		//Toast.makeText(this, "Righe: " + radii[5] + ", Colonne: " + radii[6], Toast.LENGTH_LONG).show();
		
		return radii;
		
	}

	private double euclDist(Point a, Point b){
		
		return Math.sqrt( Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2) );
		
	}
	
	private double[] shapeDescriptor(){
		
		double[] shapeFeatures = new double[8];
		MatOfInt hullIds = new MatOfInt();
		MatOfPoint convexHull = new MatOfPoint();
		MatOfPoint2f leafContour2f = new MatOfPoint2f(leafContour.toArray() );
		MatOfPoint minBoundRectPoints = new MatOfPoint();
		
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
		
		//draw the convex hull
		List<MatOfPoint> hulls = new ArrayList<MatOfPoint>();
		hulls.add(convexHull);	
		
		Mat mDraw = new Mat();
    	//from gray to RGB
    	Imgproc.cvtColor(mMaskedGray, mDraw, Imgproc.COLOR_GRAY2RGBA, 4);
    	//disegno il convex hull, per controllare
    	Imgproc.drawContours(mDraw, hulls, 0, new Scalar(0, 255, 0), 3);

		
		//leaf area
		double area = Imgproc.contourArea(leafContour);
		//leaf perimeter
		double perimeter = Imgproc.arcLength(leafContour2f, true);

		//convex hull area
		double areaCH = Imgproc.contourArea(convexHull);
		//convex hull perimeter
		MatOfPoint2f convexHull2f = new MatOfPoint2f( convexHull.toArray() );
		double perimeterCH = Imgproc.arcLength(convexHull2f, true);
		
		//Toast.makeText(this, "Perimetro: " + perimeter + ". Perimetro CH: " + perimeterCH, Toast.LENGTH_LONG).show();

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
		Imgproc.drawContours(mDraw, rects, 0, new Scalar(255, 0, 0), 3);
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
	
	private double[] getTextureFeaturesRidotto(boolean symmetric){
		int [][] offset = new int[][]{{1, 0},{1, -1},{0, -1},{-1, -1}};
		int dim = offset.length;
		double[] feature = new double[5*dim];
		double[] featurei;
		int [][] glcm;
		for (int i = 0; i < dim; i++){
			glcm = GLCM(new int[]{offset[i][0],offset[i][1]}, symmetric);
			featurei = textureDescriptorsRidotto(glcm);		
			for (int j = 0; j < 5; j++)
				feature[(i*5)+j] = featurei[j]; 				
		}
		return feature;
	}
	
	private int [][] GLCM(int[] offset, boolean symmetric){
		
		int [][] glcm = new int [256][256];
		int i2, j2;
		occurrences = 0;
		
		 Log.i("Check", "Height: " + height + ", Width: " + width);

		
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				i2 = i + offset[0];
				j2 = j + offset[1];
				if(i2>=0 && j2 >= 0 && i2 < height && j2 < width){
					int pixel1 = (int) mMaskedGray.get(i, j)[0];
					int pixel2 = (int) mMaskedGray.get(i2, j2)[0];
					glcm[pixel1][pixel2]++;
					occurrences++;
					if (symmetric){
						glcm[pixel2][pixel1]++;
						occurrences++;
					}
				}				
			}
		}	
		
		Log.i("Controllo", "Numero occorrenze: " + occurrences);

		Log.i("Controllo", "Numero in glcm[127][128]: " + glcm[127][128]);
		Log.i("Controllo", "Numero in glcm[128][127]: " + glcm[128][127]);
		occurrences -= glcm[0][0];
		Log.i("Controllo", "Numero occorrenze after: " + occurrences);

		
		
		
		
		glcm[0][0] = 0;
		return glcm;
	}
	
	private double[] textureDescriptors(int [][] GLCM){
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
	
	private double[] textureDescriptorsRidotto(int [][] GLCM){
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

	class Neighbor{
    	
    	public int x;
    	public int y;
    	public double value;
    	
    	public Neighbor(int x, int y, double value)
    	{
    		this.x = x;
    		this.y = y;
    		this.value = value;
    	}
    }
	
	class colNeighbor{
    	
    	public int x;
    	public int y;
    	public Scalar value;
    	
    	public colNeighbor(int x, int y, Scalar value)
    	{
    		this.x = x;
    		this.y = y;
    		this.value = value;
    	}
    }

	private void getLeafNames(){
		 
		 
		 File namesFile;
		 
		 try
		 {
			 namesFile = new File(namesPath);
			 //Log.i("Comunicazione", "Ok, trovato il file: " + namesFile.getName());
			 FileReader fr = new FileReader(namesFile);
			 BufferedReader br = new BufferedReader(fr);

			 String newLine;
			 
			 Log.i("Comunicazione", "Creato BufferedReader");
			 
			 while((newLine = br.readLine()) != null)
			 {
				 leafNames.add(newLine);
				 Log.i("Check", "Foglia: " + newLine);

			 }
			 
			 br.close();
			 fr.close();

		 }
		 catch (Exception e)
		 {
		 }
	 }

	public Mat saliency() {
		Mat coef = new Mat(height, width, CvType.CV_32FC2);
		ArrayList<Mat> coef_complex = new ArrayList<Mat>(2);
		Mat mag = new Mat(height, width,CvType.CV_32F);
		Mat signature = new Mat(height, width,CvType.CV_32FC2);
		// forward dft
		Core.dft(mImgRGBA,coef,Core.DFT_COMPLEX_OUTPUT,mImgRGBA.rows());
		// unify the magnitude
		
		Core.split(coef,coef_complex);
		
		Core.magnitude(coef_complex.get(0), coef_complex.get(1), mag);
		Core.divide(coef_complex.get(0),mag,coef_complex.get(0));
		Core.divide(coef_complex.get(1),mag,coef_complex.get(1));
		Core.merge(coef_complex, coef);
		// apply the inverse dft
		
		Core.dft(coef,signature,Core.DCT_INVERSE,mImgRGBA.rows());
		// take square and magnitude
		Core.split(signature, coef_complex);
		Core.magnitude(coef_complex.get(0), coef_complex.get(1), mag);
		Core.multiply(mag, mag, mag);		
		
		// add this to the result
		Core.add(sal, mag, sal); 
		return sal;
	}
}
