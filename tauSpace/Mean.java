/*
 * Copyright (C) 2014 Alex Gavryushkin <alex@gavruskin.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import beast.evolution.tree.Tree;

import java.util.Date;
import java.util.Random;

/**
 * Computing Frechet mean.
 *
 * Created on 24/11/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */
public class Mean {

    public static TauTree mean(TauTree[] trees, int numIter, int matchLength, double epsilon) {

        int CauchyMax = 0; // This is for testing. Can be removed.
        int convergenceCounter = 0;  // Counts number of iterations the two means are no further apart than epsilon.
        // Find the first mean candidate randomly:
        Random random = new Random();
        int r = random.nextInt(trees.length);
        TauTree meanOld = new TauTree();
        for (TauPartition m : trees[r].tauPartitions) {
            meanOld.addPartition(m);
        }
        int i = 1;
        while (i < numIter) {
            int j = random.nextInt(trees.length);
            TauTree meanNew = Geodesic.geodesic(meanOld, trees[j], (double) 1 / (i + 1)).tree;
            double tmpDistance = Geodesic.geodesic(meanOld, meanNew, 0.5).geoLength; // 0.5 here is a random choice. We only need the geo.geoLength.
            if (i % 10000 == 0 || convergenceCounter >= 50) {
                System.out.println("The test distance is " + tmpDistance + ".");
                Date date = new Date();
                System.out.println("The convergence counter is " + convergenceCounter + " on " + date + ".");
                meanNew.labelMap = trees[0].labelMap;
                Tree meanNewBeast = TauTree.constructFromTauTree(meanNew);
                System.out.println("The running mean tree is");
                System.out.println(meanNewBeast.getRoot().toNewick());
                System.out.print("\n");
            }
            // Check for convergence:
            if (tmpDistance < epsilon) {
                convergenceCounter++;
                if (convergenceCounter > CauchyMax) {
                    CauchyMax = convergenceCounter;
                }

                if (convergenceCounter == matchLength) {
                    System.out.println("\nThe total number of Sturm iterations is " + i + ".\n");
                    double firstTau = 0.0;
                    for (int k = 0; k < trees.length; k++) {
                        firstTau += trees[k].firstTau;
                    }
                    meanOld.firstTau = firstTau / trees.length;
                    return meanOld;
                }
            } else {
                convergenceCounter = 0;
            }
            meanOld.tauPartitions.clear();
            for (TauPartition m: meanNew.tauPartitions) {
                meanOld.addPartition(m);
            }
            i++;
        }
        System.out.println("\nAfter " + i +" iterations, the Sturm sequence has not converged! Try a longer sequence.\n");
        System.out.println("The maximal length of Cauchy sequence is " + CauchyMax + ".\n");
        double firstTau = 0.0;
        for (int k = 0; k < trees.length; k++) {
            firstTau += trees[k].firstTau;
        }
        meanOld.firstTau = firstTau / trees.length;
        return meanOld;
    }
}
