package com.example.leafrecognizer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.util.Log;

public class GMRsaliency extends ContextWrapper{

    //parameters
    private static final double compactness = 20;	//superpixels compactness
    private static final float alpha= 0.99f;		//balance the fittness and smoothness
    private static final float delta = 0.1f;		//contral the edge weight
    private static int spcounta = 225;		//actual superpixel number

    private Bitmap input;
    private Mat img;
    private Mat supLab;
    private Mat adj;
    private Mat optAff;
    private Mat W;
    private Mat tImg;
    private int[] wcut;


    public GMRsaliency(Context base, Bitmap image,Mat prova) {
        super(base);

        this.input = image;
        this.img = prova;
        // Scale to [0,1] and l*a*b colorspace
        // img.convertTo(img, CvType.CV_32F, 1/255.);
        //Mat imlab = new Mat();
        //Imgproc.cvtColor(img, imlab, Imgproc.COLOR_BGR2Lab);


    }

    private Mat GetSup() {
        SLIC slic = new SlicBuilder().buildSLIC();
        slic.createSuperpixel(input);
        return slic.GetLabels();
    }

    private Mat GetAdjLoop() {
        Mat adj = new Mat(spcounta,spcounta,CvType.CV_16U,new Scalar(0));
        for(int i=0;i<supLab.rows()-1;i++)
        {
            for(int j=0;j<supLab.cols()-1;j++)
            {
                if((int)supLab.get(i,j)[0]!=(int)supLab.get(i+1,j)[0])
                {
                    adj.put((int)supLab.get(i,j)[0],(int)supLab.get(i+1,j)[0],1);
                    adj.put((int)supLab.get(i+1,j)[0],(int)supLab.get(i,j)[0],1);
                }
                if((int)supLab.get(i,j)[0]!=(int)supLab.get(i,j+1)[0])
                {
                    adj.put((int)supLab.get(i,j)[0],(int)supLab.get(i,j+1)[0],1);
                    adj.put((int)supLab.get(i,j+1)[0],(int)supLab.get(i,j)[0],1);
                }
                if((int)supLab.get(i,j)[0]!=(int)supLab.get(i+1,j+1)[0])
                {
                    adj.put((int)supLab.get(i,j)[0],(int)supLab.get(i+1,j+1)[0],1);
                    adj.put((int)supLab.get(i+1,j+1)[0],(int)supLab.get(i,j)[0],1);
                }
                if((int)supLab.get(i+1,j)[0]!=(int)supLab.get(i,j+1)[0])
                {
                    adj.put((int)supLab.get(i+1,j)[0],(int)supLab.get(i,j+1)[0],1);
                    adj.put((int)supLab.get(i,j+1)[0],(int)supLab.get(i+1,j)[0],1);
                }
            }
        }

        List<Integer> bd_list = new ArrayList<Integer>();
        MatOfInt bd = new MatOfInt();
        int result;
        for(int i=0;i<supLab.cols();i++)
        {
            result=bd_list.lastIndexOf((int)supLab.get(0,i)[0]);
            if (result==-1 || result==bd_list.size())
                bd_list.add((int)supLab.get(0,i)[0]);
        }
        for(int i=0;i<supLab.cols();i++)
        {
            result=bd_list.lastIndexOf((int)supLab.get(supLab.rows()-1,i)[0]);
            if(result == -1 || result==bd_list.size())
                bd_list.add((int) supLab.get(supLab.rows()-1,i)[0]);
        }
        for(int i=0;i<supLab.rows();i++)
        {
            result=bd_list.lastIndexOf((int)supLab.get(i,0)[0]);
            if(result == -1 || result==bd_list.size())
                bd_list.add((int) supLab.get(i,0)[0]);
        }
        for(int i=0;i<supLab.rows();i++)
        {
            result=bd_list.lastIndexOf((int)supLab.get(i, supLab.cols()-1)[0]);
            if(result == -1 || result==bd_list.size())
                bd_list.add((int) supLab.get(i, supLab.cols()-1)[0]);
        }
        //bd.fromList(bd_list);
        //Log.d("check "," bd_list "+bd_list.size());

        int bdj,bdi;
        for (bdi=0;bdi<bd_list.size();bdi++)
            for (bdj=bdi+1;bdj<bd_list.size();bdj++) {
                adj.put(bd_list.get(bdi),bd_list.get(bdj),1);
                adj.put(bd_list.get(bdj),bd_list.get(bdi),1);
            }

        this.adj=adj;

		/*for(int i=0;i<adj.rows();i++)
		{
			for(int j=0;j<adj.cols();j++)
				Log.d("check "," i " + i+ " j " + j +"adj " + adj.get(i,j)[0]);
		}*/

        return adj;

    }

    private Mat GetWeight() {

        float[] supL = new float[spcounta];
        float[] supa = new float[spcounta];
        float[] supb = new float[spcounta];
        float[] pcount = new float[spcounta];

        for (int i=0;i<spcounta;i++){
            supL[i] = supa[i] = supb[i]=pcount[i]=0;
        }

        for(int i=0;i<tImg.rows();i++)
        {
            for(int j=0;j<tImg.cols();j++)
            {
                supL[(int)supLab.get(i,j)[0]]+=tImg.get(i,j)[0];
                supa[(int)supLab.get(i,j)[0]]+=tImg.get(i,j)[1];
                supb[(int)supLab.get(i,j)[0]]+=tImg.get(i,j)[2];
                pcount[(int)supLab.get(i,j)[0]]+=1.;
                //Log.d("check "," i "+i+" j "+j+" sup "+supLab.get(i,j)[0]);
                //Log.d("check "," i "+i+" j "+j+" img0 "+tImg.get(i,j)[0]+" img1 "+tImg.get(i,j)[1]+" img2 "+tImg.get(i,j)[2]);
            }
        }
        //Log.d("check "," tImg" + tImg + " rows "+tImg.rows() + " cols "+tImg.cols());
        //tImgMat [ 480*640*CV_32FC3, isCont=true, isSubmat=false, nativeObj=0x5e695a50, dataAddr=0x5eb43010 ] rows 480 cols 640


        for (int i=0;i<spcounta;i++)
        {
            supL[i]/=pcount[i];
            supa[i]/=pcount[i];
            supb[i]/=pcount[i];
            //Log.d("check "," i "+i+" supL "+supL[i]+" supa "+supa[i]+" supb "+supb[i] + " pcount "+pcount[i]);
        }

        Mat w= new Mat(adj.size(),CvType.CV_32F,new Scalar(-1));
        float minw=999999;
        float maxw=0;

        for(int i=0;i<spcounta;i++)
        {
            for(int j=0;j<spcounta;j++)
            {
                //Log.d("check ","adjget " + adj.get(i,j)[0]);
                if(adj.get(i,j)[0]==1.0)
                {
                    float dist=(float) Math.sqrt(Math.pow((supL[i]-supL[j]),2)+Math.pow((supa[i]-supa[j]),2)+Math.pow((supb[i]-supb[j]),2));
                    w.put(i,j,dist);
                    if(minw>dist)
                        minw=dist;
                    if(maxw<dist)
                        maxw=dist;
                    for(int k=0;k<spcounta;k++)
                    {
                        if(adj.get(j,k)[0]==1 && k!=i)
                        {
                            float dist2=(float) Math.sqrt(Math.pow((supL[i]-supL[k]),2)+Math.pow((supa[i]-supa[k]),2)+Math.pow((supb[i]-supb[k]),2));
                            w.put(i,k,dist2);
                            if(minw>dist2)
                                minw=dist2;
                            if(maxw<dist2)
                                maxw=dist2;
                        }
                    }
                }
            }
        }
        for(int i=0;i<spcounta;i++)
        {
            for(int j=0;j<spcounta;j++)
            {
                //Log.d("check ","wget " + w.get(i,j)[0]);
                if(w.get(i,j)[0]>-1)
                    w.put(i,j,Math.exp(-(w.get(i,j)[0]-minw)/((maxw-minw)*delta)));
                else
                    w.put(i,j,0);
            }
        }
        this.W=w;
		/*for(int i=0;i<w.rows();i++)
		{
			for(int j=0;j<w.cols();j++)
				Log.d("check "," i " + i + " j " + j+ " weight " + w.get(i,j)[0]);
		}*/
        return w;
    }

    private Mat GetOptAff()
    {
        Mat dd = new Mat(new Size(W.rows(),1),CvType.CV_32F);
        Core.reduce(W,dd,1,Core.REDUCE_SUM);
		/*for(int i=0;i<dd.rows();i++)
			{
				for(int j=0;j<dd.cols();j++)
					Log.d("check "," i " + i +" dd " + dd.get(i,j)[0]);
			}*/
        Mat D = new Mat(W.size(),CvType.CV_32F,new Scalar(0));
        //D=dd.diag();
        for(int i=0;i<D.rows();i++)
            D.put(i,i,dd.get(i,0)[0]);
        Mat optAff = new Mat();
        Core.multiply(W,new Scalar(alpha), optAff);
        Core.subtract(D,optAff, optAff);
        //Core.multiply(W, D, optAff);
        optAff=optAff.inv();

		/*for(int i=0;i<optAff.rows();i++)
		{
			for(int j=0;j<optAff.cols();j++)
				Log.d("check "," i " + i + " j "+j+" optAff " + optAff.get(i,j)[0]);
		}*/

        Mat B = new Mat();
        Core.subtract(Mat.ones(optAff.size(),CvType.CV_32F), Mat.eye(optAff.size(),CvType.CV_32F), B);
        optAff=optAff.mul(B);
        this.optAff=optAff;
        return optAff;
    }

    private Mat GetBdQuery(int type)
    {
        Mat y = new Mat(new Size(1,spcounta),CvType.CV_32F,new Scalar(0));
        switch(type)
        {
            case 1:
                for(int i=0;i<supLab.cols();i++)
                    y.put(0,(int)supLab.get(0,i)[0],1);
                break;
            case 2:
                for(int i=0;i<supLab.cols();i++)
                    y.put(0,(int)supLab.get(supLab.rows()-1,i)[0],1);
                break;
            case 3:
                for(int i=0;i<supLab.rows();i++)
                    y.put(0,(int)supLab.get(i,0)[0],1);
                break;
            case 4:
                for(int i=0;i<supLab.rows();i++)
                    y.put(0,(int)supLab.get(i,supLab.cols()-1)[0],1);
                break;
            default:
                Log.d("GetBdQuery ERROR: ","default choice");
        }
        return y;

    }

    private Mat RemoveFrame()
    {
        double thr=0.6;
        Mat grayimg = new Mat();
        Imgproc.cvtColor(img,grayimg,Imgproc.COLOR_BGR2GRAY);
        Mat edge = new Mat();
        Imgproc.Canny(grayimg,edge,150*0.4,150);

        int flagt=0;
        int flagd=0;
        int flagr=0;
        int flagl=0;
        int t=0;
        int d=0;
        int l=0;
        int r=0;

        int m=grayimg.rows();
        int n=grayimg.cols();

        int i=0;
        while(i<30)
        {
            Scalar pbt=Core.mean(new Mat(edge,new Range(i,i+1),new Range(0,n)));
            Scalar pbd=Core.mean(new Mat(edge,new Range(m-i-1,m-i),new Range(0,n)));
            Scalar pbl=Core.mean(new Mat(edge,new Range(0,m),new Range(i,i+1)));
            Scalar pbr=Core.mean(new Mat(edge,new Range(0,m),new Range(n-i-1,n-i)));
            if(pbt.val[0]/255>thr)
            {
                t=i;
                flagt=1;
            }
            if(pbd.val[0]/255>thr)
            {
                d=i;
                flagd=1;
            }
            if(pbl.val[0]/255>thr)
            {
                l=i;
                flagl=1;
            }
            if(pbr.val[0]/255>thr)
            {
                r=i;
                flagr=1;
            }
            i++;
        }
        int flagrm=flagt+flagd+flagl+flagr;
        Mat outimg;
        if(flagrm>1)
        {
            int maxwidth;
            maxwidth=(t>d)?t:d;
            maxwidth=(maxwidth>l)?maxwidth:l;
            maxwidth=(maxwidth>r)?maxwidth:r;
            if(t==0)
                t=maxwidth;
            if(d==0)
                d=maxwidth;
            if(l==0)
                l=maxwidth;
            if(r==0)
                r=maxwidth;
            outimg=new Mat (img,new Range(t,m-d),new Range(l,n-r));
            wcut[0]=m;
            wcut[1]=n;
            wcut[2]=t;
            wcut[3]=m-d;
            wcut[4]=l;
            wcut[5]=n-r;
        }
        else
        {
            wcut[0]=m;
            wcut[1]=n;
            wcut[2]=0;
            wcut[3]=m;
            wcut[4]=0;
            wcut[5]=n;
            outimg=img;
        }
        //Log.d("check ","RemoveFrame 0" +wcut[0]+" 1 "+wcut[1]+" 2 "+wcut[2]+" 3 "+wcut[3]+" 4 "+wcut[4]+" 5 "+wcut[5]);
        return outimg;
    }

    public Mat GetSal(int version, Mat customLabels, int nC) {

        wcut = new int[6];
        img = RemoveFrame();
        //Log.d("check ","RemoveFrame");

        this.supLab = new Mat(img.size(),CvType.CV_16U);

        if(version == 1) {
            this.supLab = GetSup();
            //Log.d("check ","GetSup");
        }
        else {
            customLabels.convertTo(this.supLab, CvType.CV_16U);
            spcounta = nC;
        }

        this.adj= new Mat(new Size(spcounta,spcounta),CvType.CV_16U);
        this.adj=GetAdjLoop();
        //Log.d("check ","GetAdjLoop");

        tImg=new Mat();
        img.convertTo(tImg,CvType.CV_32FC3,1.0/255);
        Imgproc.cvtColor(tImg, tImg, Imgproc.COLOR_BGR2Lab);

        this.W = new Mat(adj.size(),CvType.CV_32F);
        this.W=GetWeight();
        //Log.d("check ","GetWeight");

        this.optAff = new Mat(W.size(),CvType.CV_32F);
        this.optAff=GetOptAff();

        //Log.d("check ","GetOptAff");

        Mat salt = new Mat();
        Mat sald = new Mat();
        Mat sall = new Mat();
        Mat salr = new Mat();
        Mat sal1 = new Mat();
        Mat bdy = new Mat();

        bdy=GetBdQuery(1);
        //Log.d("check ","GetBdQuery1");
        Core.gemm(optAff,bdy, 1, bdy, 0, salt); //basic multiplication operator *
        Core.normalize(salt, salt, 0, 1, Core.NORM_MINMAX);
        sal1=salt.clone();
        sal1.setTo(new Scalar(1));
        Core.subtract(sal1, salt, salt);

        bdy=GetBdQuery(2);
        //Log.d("check ","GetBdQuery2");
        Core.gemm(optAff,bdy, 1, bdy, 0, sald);
        Core.normalize(sald, sald, 0, 1, Core.NORM_MINMAX);
        sal1=sald.clone();
        sal1.setTo(new Scalar(1));
        Core.subtract(sal1, sald, sald);

        bdy=GetBdQuery(3);
        //Log.d("check ","GetBdQuery3");
        Core.gemm(optAff,bdy, 1, bdy, 0, sall);
        Core.normalize(sall, sall, 0, 1, Core.NORM_MINMAX);
        sal1=sall.clone();
        sal1.setTo(new Scalar(1));
        Core.subtract(sal1, sall, sall);

        bdy=GetBdQuery(4);
        //Log.d("check ","GetBdQuery4");
        Core.gemm(optAff,bdy, 1, bdy, 0, salr);
        Core.normalize(salr, salr, 0, 1, Core.NORM_MINMAX);
        sal1=salr.clone();
        sal1.setTo(new Scalar(1));
        Core.subtract(sal1, salr, salr);

        Mat salb=new Mat();
        salb=salt.clone();
        salb=salb.mul(sald);
        salb=salb.mul(sall);
        salb=salb.mul(salr);

        Scalar thr=Core.mean(salb);
        Mat fgy = new Mat();
        Imgproc.threshold(salb,fgy,thr.val[0],1,Imgproc.THRESH_BINARY);

        Mat salf=new Mat();
        Core.gemm(optAff,fgy, 1, fgy, 0, salf);
        //Log.d("check ","Getsalf");
        //Core.multiply(optAff,fgy,salf); //opt 225*225 fgy 225*1

        Mat salMap= new Mat(img.size(),CvType.CV_32F);
        for(int i=0;i<salMap.rows();i++)
        {
            for(int j=0;j<salMap.cols();j++)
            {
                salMap.put(i,j,salf.get((int) supLab.get(i,j)[0],0));
            }
        }
        //Log.d("check ","salMap");
        Core.normalize(salMap, salMap, 0, 1, Core.NORM_MINMAX);

        Mat outMap= new Mat(new Size(wcut[1],wcut[0]),CvType.CV_32F,new Scalar(0));
        Mat subMap=new Mat(outMap,new Range(wcut[2],wcut[3]),new Range(wcut[4],wcut[5]));
        salMap.convertTo(subMap,subMap.type());
        //Log.d("check ","outMap");
        Core.multiply(outMap,new Scalar(255), outMap);
        return outMap; //outmap
    }
}
