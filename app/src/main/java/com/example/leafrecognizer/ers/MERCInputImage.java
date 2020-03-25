package com.example.leafrecognizer.ers;

import android.util.Log;

import org.opencv.core.Mat;

/**
 * Created by Cesare on 07/06/2016.
 */
public class MERCInputImage {
    private int width_;
    private int height_;
    private int nEdges_;
    private int nNodes_;
    public Edge[] edges_;

    public MERCInputImage() {
        this.width_ = 0;
        this.height_= 0;
    }

    public void readImage(Mat input, int conn8) {
        width_ = input.rows();
        height_ = input.cols();
        nNodes_ = width_ * height_;


        edges_ = new Edge[width_*height_*4];

        // inizializzazione
        for(int i = 0; i < edges_.length; i++) {
            edges_[i] = new Edge(0,0,0,0);
        }

        int num = 0;
        for (int y = 0; y < height_; y++) {
            for (int x = 0; x < width_; x++) {
                if (x < width_ - 1) {
                    edges_[num].setA_(y * width_ + x);
                    edges_[num].setB_(y * width_ + (x+1));
                    //edges_[num].w_ = DiffGrayImage(image, x, y, x+1, y);
                    edges_[num].setW_(Math.abs(1.0*(input.get(x,y)[0] - input.get(x+1,y)[0])));
                    num++;
                }

                if (y < height_ - 1) {
                    edges_[num].setA_(y * width_ + x);
                    edges_[num].setB_((y+1) * width_ + x);
                    //edges_[num].w_ = DiffGrayImage(image, x, y, x, y+1);
                    edges_[num].setW_(Math.abs(1.0*(input.get(x,y)[0] - input.get(x,y+1)[0])));
                    num++;
                }

                if(conn8 == 1) {
                    if ((x < (width_ - 1)) && (y < (height_ - 1))) {
                        edges_[num].setA_(y * width_ + x);
                        edges_[num].setB_((y+1) * width_ + (x+1));
                        //edges_[num].w_ = std::sqrt(2.0)*DiffGrayImage(image, x, y, x+1, y+1);
                        edges_[num].setW_(Math.sqrt(2.0)*(Math.abs(1.0*(input.get(x,y)[0] - input.get(x+1,y+1)[0]))));
                        num++;
                    }

                    if ((x < width_ - 1) && (y > 0)) {
                        edges_[num].setA_(y * width_ + x);
                        edges_[num].setB_((y-1) * width_ + (x+1));
                        //edges_[num].w_ = std::sqrt(2.0)*DiffGrayImage(image, x, y, x+1, y-1);
                        edges_[num].setW_(Math.sqrt(2.0)*(Math.abs(1.0*(input.get(x,y)[0] - input.get(x+1,y-1)[0]))));
                        num++;
                    }
                }
            }
        }

        nEdges_ = num;
    }

    public int get_nEdges() {
        return nEdges_;
    }

    public int get_nNodes() {
        return nNodes_;
    }
}
