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

/*
import ij.*;
import ij.plugin.filter.PlugInFilter;
import ij.process.*;
import ij.gui.*;
import ij.plugin.frame.PlugInFrame;
*/

import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.*;

/**
 *  This algorithm is an implementation of the watershed immersion algorithm
 *  written by Vincent and Soille (1991).
 *
 *  @Article{Vincent/Soille:1991,
 *   author =       "Lee Vincent and Pierre Soille",
 *   year =         "1991",
 *   keywords =     "IMAGE-PROC SKELETON SEGMENTATION GIS",
 *   institution =  "Harvard/Paris+Louvain",
 *   title =        "Watersheds in digital spaces: An efficient algorithm
 *                   based on immersion simulations",
 *   journal =      "IEEE PAMI, 1991",
 *   volume =       "13",
 *   number =       "6",
 *   pages =        "583--598",
 *   annote =       "Watershed lines (e.g. the continental divide) mark the
 *                  boundaries of catchment regions in a topographical map.
 *                  The height of a point on this map can have a direct
 *                  correlation to its pixel intensity. WIth this analogy,
 *                  the morphological operations of closing (or opening)
 *                  can be understood as smoothing the ridges (or filling
 *                  in the valleys). Develops a new algorithm for obtaining
 *                  the watershed lines in a graph, and then uses this in
 *                  developing a new segmentation approach based on the
 *                  {"}depth of immersion{"}.",
 *  }
 *
 *  A review of Watershed algorithms can be found at :
 *  http://www.cs.rug.nl/~roe/publications/parwshed.pdf
 *
 *  @Article{RoeMei00,
 *   author =       "Roerdink and Meijster",
 *   title =        "The Watershed Transform: Definitions, Algorithms and
 *                   Parallelization Strategies",
 *   journal =      "FUNDINF: Fundamenta Informatica",
 *   volume =       "41",
 *   publisher =    "IOS Press",
 *   year =         "2000",
 *  }
 **/

public class Watershed_Algorithm {
    private int threshold;
    final static int HMIN = 0;
    final static int HMAX = 256;

    public Mat run(Mat image) {
        /** First step : the pixels are sorted according to increasing grey values **/
        WatershedStructure watershedStructure = new WatershedStructure(image);

        /** Start flooding **/
        WatershedFIFO queue = new WatershedFIFO();
        int curlab = 0;

        int heightIndex1 = 0;
        int heightIndex2 = 0;

        for(int h=HMIN; h<HMAX; h++) /*Geodesic SKIZ of level h-1 inside level h */ {

            for(int pixelIndex = heightIndex1 ; pixelIndex<watershedStructure.size() ; pixelIndex++) /*mask all pixels at level h*/ {
                WatershedPixel p = watershedStructure.get(pixelIndex);

                if(p.getIntHeight() != h) {
                    /** This pixel is at level h+1 **/
                    heightIndex1 = pixelIndex;
                    break;
                }

                p.setLabelToMASK();

                Vector neighbours = p.getNeighbours();
                for(int i=0 ; i<neighbours.size() ; i++) {
                    WatershedPixel q = (WatershedPixel) neighbours.get(i);

                    if(q.getLabel()>=0) {/*Initialise queue with neighbours at level h of current basins or watersheds*/
                        p.setDistance(1);
                        queue.fifo_add(p);
                        break;
                    } // end if
                } // end for
            } // end for


            int curdist = 1;
            queue.fifo_add_FICTITIOUS();

            while(true) /** extend basins **/{
                WatershedPixel p = queue.fifo_remove();

                if(p.isFICTITIOUS())
                    if(queue.fifo_empty())
                        break;
                    else {
                        queue.fifo_add_FICTITIOUS();
                        curdist++;
                        p = queue.fifo_remove();
                    }

                Vector neighbours = p.getNeighbours();
                for(int i=0 ; i<neighbours.size() ; i++) /* Labelling p by inspecting neighbours */{
                    WatershedPixel q = (WatershedPixel) neighbours.get(i);

            /* Original algorithm :
               if( (q.getDistance() < curdist) &&
               (q.getLabel()>0 || q.isLabelWSHED()) ) {*/
                    if( (q.getDistance() <= curdist) &&
                            (q.getLabel()>=0) ) {
            /* q belongs to an existing basin or to a watershed */

                        if(q.getLabel() > 0) {
                            if( p.isLabelMASK() )
                                // Removed from original algorithm || p.isLabelWSHED() )
                                p.setLabel(q.getLabel());
                            else
                            if(p.getLabel() != q.getLabel())
                                p.setLabelToWSHED();
                        } // end if lab>0
                        else
                        if(p.isLabelMASK())
                            p.setLabelToWSHED();
                    }
                    else
                    if( q.isLabelMASK() && (q.getDistance() == 0) ) {
                        q.setDistance( curdist+1 );
                        queue.fifo_add( q );
                    }
                } // end for, end processing neighbours
            } // end while (loop)

        /* Detect and process new minima at level h */
            for(int pixelIndex = heightIndex2 ; pixelIndex<watershedStructure.size() ; pixelIndex++) {
                WatershedPixel p = watershedStructure.get(pixelIndex);

                if(p.getIntHeight() != h) {
                    /** This pixel is at level h+1 **/
                    heightIndex2 = pixelIndex;
                    break;
                }

                p.setDistance(0); /* Reset distance to zero */

                if(p.isLabelMASK()) { /* the pixel is inside a new minimum */
                    curlab++;
                    p.setLabel(curlab);
                    queue.fifo_add(p);


                    while(!queue.fifo_empty()) {
                        WatershedPixel q = queue.fifo_remove();

                        Vector neighbours = q.getNeighbours();

                        for(int i=0 ; i<neighbours.size() ; i++) /* inspect neighbours of p2*/{
                            WatershedPixel r = (WatershedPixel) neighbours.get(i);

                            if( r.isLabelMASK() ) {
                                r.setLabel(curlab);
                                queue.fifo_add(r);
                            }
                        }
                    } // end while
                } // end if
            } // end for
        } /** End of flooding **/

        /** Put the result in a new image **/
        Mat outputImage = image.clone();

        for(int pixelIndex = 0 ; pixelIndex<watershedStructure.size() ; pixelIndex++) {
            WatershedPixel p = watershedStructure.get(pixelIndex);

            if(p.isLabelWSHED() && !p.allNeighboursAreWSHED())
                outputImage.put(p.getX(), p.getY(), 255);
            else
                outputImage.put(p.getX(), p.getY(), 0);
        }

        return outputImage;
    }
}
