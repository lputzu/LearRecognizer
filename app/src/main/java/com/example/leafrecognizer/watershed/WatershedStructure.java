package com.example.leafrecognizer.watershed;

/*
 * Watershed algorithm
 *
 * Copyright (c) 2003 by Christopher Mei (christopher.mei@sophia.inria.fr)
 *
 * This plugin is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this plugin; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

import android.graphics.Bitmap;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.lang.*;
import java.util.*;


/*
import ij.process.*;
import ij.*;
*/



/**
 *  WatershedStructure contains the pixels
 *  of the image ordered according to their
 *  grayscale value with a direct access to their
 *  neighbours.
 *
 **/

public class WatershedStructure {
    private Vector watershedStructure;

    public WatershedStructure(Mat image) {

        int width = image.cols();
        int offset, topOffset, bottomOffset;

        watershedStructure = new Vector(image.cols()*image.rows());

        /** The structure is filled with the pixels of the image. **/
        for(int i=0; i<image.rows(); i++) {
            for(int j=0; j<image.cols(); j++) {
                watershedStructure.add(new WatershedPixel(i, j, (byte) image.get(i,j)[0]));
            }
        }

        /** The WatershedPixels are then filled with the reference to their neighbours. **/
        for (int y=0; y<image.rows(); y++) {

            offset = y*width;
            topOffset = offset+width;
            bottomOffset = offset-width;

            for (int x=0; x<image.cols(); x++) {
                WatershedPixel currentPixel = (WatershedPixel)watershedStructure.get(x+offset);

                if(x+1<image.cols()) {
                    currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x+1+offset));

                    if(y-1>=0)
                        currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x+1+bottomOffset));

                    if(y+1<image.rows())
                        currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x+1+topOffset));
                }

                if(x-1>=0) {
                    currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x-1+offset));

                    if(y-1>=0)
                        currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x-1+bottomOffset));

                    if(y+1<image.rows())
                        currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x-1+topOffset));
                }

                if(y-1>=0)
                    currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x+bottomOffset));

                if(y+1<image.rows())
                    currentPixel.addNeighbour((WatershedPixel)watershedStructure.get(x+topOffset));
            }
        }

        Collections.sort(watershedStructure);
    }

    public String toString() {
        StringBuffer ret = new StringBuffer();

        for(int i=0; i<watershedStructure.size() ; i++) {
            ret.append((watershedStructure.get(i)).toString());
            ret.append("\n");
            ret.append("Neighbours :\n");

            Vector neighbours = ((WatershedPixel) watershedStructure.get(i)).getNeighbours();

            for(int j=0 ; j<neighbours.size() ; j++) {
                ret.append((neighbours.get(j)).toString());
                ret.append("\n");
            }
            ret.append("\n");
        }
        return ret.toString();
    }

    public int size() {
        return watershedStructure.size();
    }

    public WatershedPixel get(int i) {
        return (WatershedPixel) watershedStructure.get(i);
    }
}