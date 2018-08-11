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

import java.util.ArrayList;
import java.util.List;


/**
 * This is the main class where the geodesic gets computed.
 *
 * Created on 21/10/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */

public class Geodesic {

    public TauTree tree;
    public double geoLength;

    // areCompatible[0] returns true iff partitions part1 and part2 are compatible. areCompatible[1] returns true iff
    // part1 implies part2. areCompatible[2] returns true iff part2 implies part1.
    // areCompatible[3] returns true iff partitions part1 and part2 are identical.

    public static boolean[] areCompatible(boolean[][] part1, boolean[][] part2) {

        boolean[] tmp = new boolean[4];
        tmp[0] = false;
        boolean tmpCompatible = true;
        outerloop:
        for (int i = 0; i < part1.length; i++) {
            for (int j = i; j < part1[i].length; j++) {
                if (!part1[i][j] || part2[i][j]) {
                    tmpCompatible = true;
                } else {
                    tmpCompatible = false;
                    break outerloop;
                }
            }
        }
        if (tmpCompatible) {
            tmp[0] = true;
            tmp[1] = true;
        } else {
            tmp[1] = false;
            tmpCompatible = true;
        }
        outerloop:
        for (int i = 0; i < part1.length; i++) {
            for (int j = i; j < part1[i].length; j++) {
                if (part1[i][j] || !part2[i][j]) {
                    tmpCompatible = true;
                } else {
                    tmpCompatible = false;
                    break outerloop;
                }
            }
        }
        if (tmpCompatible) {
            tmp[0] = true;
            tmp[2] = true;
        } else {
            tmp[2] = false;
        }

        if (tmp[1] && tmp[2]) {
            tmp[3] = true;
        } else {
            tmp[3] = false;
        }
        return tmp;
    }

    // inCompatibilityGraph returns the incompatibility graph given two tau trees.

    private static Graph inCompatibilityGraph(TauTree tree1, TauTree tree2) {

        Graph tmpGraph = new Graph();
        for (TauPartition i : tree1.tauPartitions) {
            Vertex k = new Vertex(tree1.numTaxa);
            k.matrix = new boolean[i.matrix.length][i.matrix.length];
            for (int r = 0; r < i.matrix.length; r++) {
                for (int l = 0; l < i.matrix[r].length; l++) {
                    k.matrix[r][l] = i.matrix[r][l];
                }
            }
            //k.matrix = i.matrix;
            k.weight = i.tauCoordinate;
            tmpGraph.verticesA.add(k);
        }
        for (TauPartition j : tree2.tauPartitions) {
            Vertex l = new Vertex(tree2.numTaxa);
            l.matrix = new boolean[j.matrix.length][j.matrix.length];
            for (int k = 0; k < j.matrix.length; k++) {
                for (int r = 0; r < j.matrix[k].length; r++) {
                    l.matrix[k][r] = j.matrix[k][r];
                }
            }
            //l.matrix = j.matrix;
            l.weight = j.tauCoordinate;
            tmpGraph.verticesB.add(l);
        }

        // incidenceMatrix is not a usual incidence matrix: rows are the vertices of the left-side partition,
        // columns---right-side.
        tmpGraph.incidenceMatrix = new boolean[tmpGraph.verticesA.size()][tmpGraph.verticesB.size()];// HERE!!!!

        for (Vertex k : tmpGraph.verticesA) {
            for (Vertex l : tmpGraph.verticesB) {
                if (!areCompatible(k.matrix, l.matrix)[0]) {
                    k.adjacent.add(l);
                    l.adjacent.add(k);
                    tmpGraph.incidenceMatrix[tmpGraph.verticesA.indexOf(k)][tmpGraph.verticesB.indexOf(l)] = true;
                }
            }
        }
        return tmpGraph;
    }

    // geodesic, given two trees and a real number t between 0 and 1, returns the tree traversed by the geodesic at time t.

    public static Geodesic geodesic(TauTree tree1, TauTree tree2, double t) {
        if (t < 0 || t > 1){
            System.out.println("The time t must be within the interval [0,1]! Terminating...");
            System.exit(0);
        }
        Geodesic tmpGeo = new Geodesic();
        TauTree tmpTree = new TauTree();
        double length = 0;

        ArrayList<TauTree> tmpGeodesic = new ArrayList<>();
        List<List<List<Integer>>> graphGeodesic = new ArrayList<>();        // graphGeodesic is a list of pairs of
        // lists of vertices, where 0th element of
        // a pair is the A_i, the 1st---B_i.

        // Checking if the trees have all partitions identical. If so, return the corresponding geodesic:
        boolean allIdentical = false;
        for (TauPartition i : tree1.tauPartitions) {
            allIdentical = false;
            for (TauPartition j : tree2.tauPartitions) {
                if (areCompatible(i.matrix,j.matrix)[3]) {
                    allIdentical = true;
                    break;
                }
            }
            if (!allIdentical) {
                break;
            }
        }
        if (allIdentical) {
            for (TauPartition j : tree2.tauPartitions) {
                allIdentical = false;
                for (TauPartition i : tree1.tauPartitions) {
                    if (areCompatible(i.matrix, j.matrix)[3]) {
                        allIdentical = true;
                        break;
                    }
                }
                if (!allIdentical) {
                    break;
                }
            }
        }
        if (allIdentical) {
            TauPartition J = new TauPartition();
            for (TauPartition i : tree1.tauPartitions) {
                for (TauPartition j : tree2.tauPartitions) {
                    if (areCompatible(i.matrix,j.matrix)[3]) {
                        J.matrix = new boolean[j.matrix.length][j.matrix.length];
                        for (int k = 0; k < j.matrix.length; k++) {
                            for (int l = 0; l < j.matrix[k].length; l++) {
                                J.matrix[k][l] = j.matrix[k][l];
                            }
                        }
                        J.tauCoordinate = j.tauCoordinate;
                        break;
                    }
                }
                tmpTree.addPartition(i);
                tmpTree.tauPartitions.get(tmpTree.tauPartitions.size() - 1).tauCoordinate =
                        (1-t)*i.tauCoordinate + t*J.tauCoordinate;
                length = length +
                        Math.pow(i.tauCoordinate - J.tauCoordinate, 2);
            }
            tmpGeo.tree = tmpTree;
            tmpGeo.geoLength = Math.sqrt(length);
            return tmpGeo;
        }

        // Checking if the tree1 is a subtree of tree2. If so, return the corresponding geodesic:
        boolean tree1subtree2 = false;
        for (TauPartition i : tree1.tauPartitions) {
            tree1subtree2 = false;
            for (TauPartition j : tree2.tauPartitions) {
                if (areCompatible(i.matrix,j.matrix)[3]) {
                    tree1subtree2 = true;
                    break;
                }
            }
            if (!tree1subtree2) {
                break;
            }
        }
        if (tree1subtree2 && t == 0) {
            for (TauPartition i : tree1.tauPartitions) {
                tmpTree.addPartition(i);
            }
            tmpGeo.tree = tmpTree; // TODO: Here and further, I have to add the length of the geodesic to the return.
            return tmpGeo;
        }
        if (tree1subtree2 && t == 1) {
            for (TauPartition j : tree2.tauPartitions) {
                tmpTree.addPartition(j);
            }
            tmpGeo.tree = tmpTree;
            return  tmpGeo;
        }
        if (tree1subtree2 && t != 0 && t != 1) {
            TauPartition I = new TauPartition();
            boolean thisOne = false;
            for (TauPartition j : tree2.tauPartitions) {
                for (TauPartition i : tree1.tauPartitions) {
                    thisOne = false;
                    if (areCompatible(i.matrix,j.matrix)[3]) {
                        I.matrix = new boolean[i.matrix.length][i.matrix.length];
                        for (int k = 0; k < i.matrix.length; k++) {
                            for (int l = 0; l < i.matrix[k].length; l++) {
                                I.matrix[k][l] = i.matrix[k][l];
                            }
                        }
                        I.tauCoordinate = i.tauCoordinate;
                        thisOne = true;
                        break;
                    }
                }
                if (thisOne) {
                    tmpTree.addPartition(j);
                    tmpTree.tauPartitions.get(tmpTree.tauPartitions.size() - 1).tauCoordinate =
                            (1-t)*I.tauCoordinate + t*j.tauCoordinate;
                    length += Math.pow(I.tauCoordinate - j.tauCoordinate, 2);
                } else {
                    tmpTree.addPartition(j);
                    tmpTree.tauPartitions.get(tmpTree.tauPartitions.size() - 1).tauCoordinate =
                            t*j.tauCoordinate;
                    length += Math.pow(j.tauCoordinate, 2);
                }
            }
            tmpGeo.tree = tmpTree;
            tmpGeo.geoLength = Math.sqrt(length);
            return tmpGeo;
        }

        // Checking if the tree2 is a subtree of tree1. If so, return the corresponding geodesic:
        boolean tree2subtree1 = false;
        for (TauPartition j : tree2.tauPartitions) {
            tree2subtree1 = false;
            for (TauPartition i : tree1.tauPartitions) {
                if (areCompatible(i.matrix,j.matrix)[3]) {
                    tree2subtree1 = true;
                    break;
                }
            }
            if (!tree2subtree1) {
                break;
            }
        }

        if (tree2subtree1 && t == 0) {
            for (TauPartition i : tree1.tauPartitions) {
                tmpTree.addPartition(i);
            }
            tmpGeo.tree = tmpTree;
            return tmpGeo;
        }
        if (tree2subtree1 && t == 1) {
            for (TauPartition j : tree2.tauPartitions) {
                tmpTree.addPartition(j);
            }
            tmpGeo.tree = tmpTree;
            return tmpGeo;
        }
        if (tree2subtree1 && t != 0 && t != 1) {
            TauPartition J = new TauPartition();
            boolean thisOne = false;
            for (TauPartition i : tree1.tauPartitions) {
                for (TauPartition j : tree2.tauPartitions) {
                    thisOne = false;
                    if (areCompatible(i.matrix,j.matrix)[3]) {
                        J.matrix = new boolean[j.matrix.length][j.matrix.length];
                        for (int k = 0; k < j.matrix.length; k++) {
                            for (int l = 0; l < j.matrix[k].length; l++) {
                                J.matrix[k][l] = j.matrix[k][l];
                            }
                        }
                        J.tauCoordinate = j.tauCoordinate;
                        thisOne = true;
                        break;
                    }
                }
                if (thisOne) {
                    tmpTree.addPartition(i);
                    tmpTree.tauPartitions.get(tmpTree.tauPartitions.size() - 1).tauCoordinate =
                            (1-t)*i.tauCoordinate + t*J.tauCoordinate;
                    length += Math.pow(i.tauCoordinate - J.tauCoordinate, 2);
                } else {
                    tmpTree.addPartition(i);
                    tmpTree.tauPartitions.get(tmpTree.tauPartitions.size() - 1).tauCoordinate =
                            (1-t)*i.tauCoordinate;
                    length += Math.pow(i.tauCoordinate, 2);
                }
            }
            tmpGeo.tree = tmpTree;
            tmpGeo.geoLength = Math.sqrt(length);
            return tmpGeo;
        }

        TauTree treeIdentical = new TauTree();
        TauTree tree1noIdentical = new TauTree();
        TauTree tree2noIdentical = new TauTree();

        boolean[] identicalFrom1 = new boolean[tree1.tauPartitions.size()];
        for (int i = 0; i < identicalFrom1.length; i++) {
            identicalFrom1[i] = false;
        }
        boolean[] identicalFrom2 = new boolean[tree2.tauPartitions.size()];
        for (int i = 0; i < identicalFrom2.length; i++) {
            identicalFrom2[i] = false;
        }
        for (TauPartition i : tree1.tauPartitions) {
            for (TauPartition j : tree2.tauPartitions) {
                if (areCompatible(i.matrix,j.matrix)[3]) {
                    TauPartition tmpPrt = new TauPartition();
                    tmpPrt.matrix = new boolean[i.matrix.length][i.matrix.length];
                    for (int k = 0; k < i.matrix.length; k++) {
                        for (int l = 0; l < i.matrix[k].length; l++) {
                            tmpPrt.matrix[k][l] = i.matrix[k][l];
                        }
                    }
                    tmpPrt.tauCoordinate = (1-t)*i.tauCoordinate + t*j.tauCoordinate;;
                    treeIdentical.addPartition(tmpPrt);
                    length = length +
                            Math.pow(i.tauCoordinate - j.tauCoordinate, 2);
                    identicalFrom1[tree1.tauPartitions.indexOf(i)] = true;
                    identicalFrom2[tree2.tauPartitions.indexOf(j)] = true;
                }
            }
        }

        for (TauPartition i : tree1.tauPartitions) {
            if (!identicalFrom1[tree1.tauPartitions.indexOf(i)]) {
                tree1noIdentical.addPartition(i);
            }
        }

        for (TauPartition j : tree2.tauPartitions) {
            if (!identicalFrom2[tree2.tauPartitions.indexOf(j)]) {
                tree2noIdentical.addPartition(j);
            }
        }

        Graph incmptGraph = inCompatibilityGraph(tree1noIdentical, tree2noIdentical);

        double[] Aweight = new double[tree1noIdentical.tauPartitions.size()];
        double[] Bweight = new double[tree2noIdentical.tauPartitions.size()];
        for (int i = 0; i < tree1noIdentical.tauPartitions.size(); i++) {
            Aweight[i] = tree1noIdentical.tauPartitions.get(i).tauCoordinate;
            incmptGraph.verticesA.get(i).weight = tree1noIdentical.tauPartitions.get(i).tauCoordinate;
        }
        for (int i = 0; i < tree2noIdentical.tauPartitions.size(); i++) {
            Bweight[i] = tree2noIdentical.tauPartitions.get(i).tauCoordinate;
            incmptGraph.verticesB.get(i).weight = tree2noIdentical.tauPartitions.get(i).tauCoordinate;
        }
        BipartiteGraph bprtGraph = new BipartiteGraph(incmptGraph.incidenceMatrix, Aweight, Bweight);

        int[] indicesA = new int[incmptGraph.verticesA.size()];
        int[] indicesB = new int[incmptGraph.verticesB.size()];
        for (int i = 0; i < incmptGraph.verticesA.size(); i++) {
            indicesA[i] = i;
        }
        for (int i = 0; i < incmptGraph.verticesB.size(); i++) {
            indicesB[i] = i;
        }
        ArrayList<Integer> cover1list = new ArrayList<>();
        for (int i = 0; i < indicesA.length; i++) {
            cover1list.add(indicesA[i]);
        }
        ArrayList<Integer> cover2list = new ArrayList<>();
        for (int i = 0; i < indicesB.length; i++) {
            cover2list.add(indicesB[i]);
        }
        List<List<Integer>> cover = new ArrayList<>();
        cover.add(cover1list);
        cover.add(cover2list);
        graphGeodesic.add(cover);

        // here the main loop comes:
        boolean ShortcutProperty = true;

        while (ShortcutProperty) {
            ShortcutProperty = false;
            for (int i = 0; i < graphGeodesic.size(); i++) {
                indicesA = new int[graphGeodesic.get(i).get(0).size()];
                indicesB = new int[graphGeodesic.get(i).get(1).size()];
                for (int j = 0; j < graphGeodesic.get(i).get(0).size(); j++) {
                    int tempInt = graphGeodesic.get(i).get(0).get(j);
                    indicesA[j] = tempInt;
                }
                for (int j = 0; j < graphGeodesic.get(i).get(1).size(); j++) {
                    int tempInt = graphGeodesic.get(i).get(1).get(j);
                    indicesB[j] = tempInt;
                }

                ArrayList<Integer> cover1listLoop = bprtGraph.vertex_cover1list(indicesA, indicesB);
                ArrayList<Integer> cover4list = bprtGraph.vertex_cover2list(indicesA, indicesB);
                ArrayList<Integer> cover2listLoop = new ArrayList<>();
                for (int j = 0; j < indicesB.length; j++) {
                        if (!cover4list.contains(indicesB[j])) cover2listLoop.add(indicesB[j]);
                }
                ArrayList<Integer> cover3list = new ArrayList<>();
                for (int j = 0; j < indicesA.length; j++) {
                        if (!cover1listLoop.contains(indicesA[j])) cover3list.add(indicesA[j]);
                }

                double testWeight = 0.0;
                for (int j = 0; j < cover1listLoop.size(); j++) {
                    testWeight += bprtGraph.AweightNormalised[cover1listLoop.get(j)];
                }
                for (int j = 0; j < cover4list.size(); j++) {
                    testWeight += bprtGraph.BweightNormalised[cover4list.get(j)];
                }

                if (testWeight < 1 && cover1listLoop.size() > 0 && cover2listLoop.size() > 0 && cover3list.size() > 0 && cover4list.size() > 0) {
                    ShortcutProperty = true;

                    List<List<Integer>> cover1 = new ArrayList<>();
                    List<List<Integer>> cover2 = new ArrayList<>();
                    cover1.add(cover1listLoop);
                    cover1.add(cover2listLoop);
                    cover2.add(cover3list);
                    cover2.add(cover4list);

                    graphGeodesic.remove(i);
                    graphGeodesic.add(i, cover2);
                    graphGeodesic.add(i, cover1);
                }

            }
        }

        // Computing the norms of the partitions:
        double[] A = new double[graphGeodesic.size()];
        double[] B = new double[graphGeodesic.size()];
        for (int i = 0; i < graphGeodesic.size(); i++) {
            for (int j = 0; j < graphGeodesic.get(i).get(0).size(); j++) {
                 A[i] += bprtGraph.Avertex[graphGeodesic.get(i).get(0).get(j)].weight;
                //A[i] += tree1.tauPartitions.get(graphGeodesic.get(i).get(0).get(j)).tauCoordinate;
            }
            A[i] = Math.sqrt(A[i]);
            for (int j = 0; j < graphGeodesic.get(i).get(1).size(); j++) {
                B[i] += bprtGraph.Bvertex[graphGeodesic.get(i).get(1).get(j)].weight;
                //B[i] += tree2.tauPartitions.get(graphGeodesic.get(i).get(1).get(j)).tauCoordinate;
            }
            B[i] = Math.sqrt(B[i]);
            length += Math.pow(A[i] + B[i], 2) ;
        }

        // Adding trees to the geodesic:
        for (int i = 0; i <= graphGeodesic.size(); i++) {
            TauTree additionalTree = new TauTree();

            // Adding identical partitions:
            for (TauPartition j : treeIdentical.tauPartitions) {
                additionalTree.addPartition(j);
            }

            // Adding partition corresponding to the destination-tree:
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < graphGeodesic.get(j).get(1).size(); k++) {
                    int b = graphGeodesic.get(j).get(1).get(k);
                    //if (t/(1-t) >= A[graphGeodesic.size() - 1] / B[graphGeodesic.size() - 1]) {
                    //    tree2noIdentical.tauPartitions.get(b).tauCoordinate =
                    //            (t*B[j] - (1-t)*A[j])*tree2noIdentical.tauPartitions.get(b).tauCoordinate / B[j];
                    //}
                    //if (i < graphGeodesic.size() - 1 && A[i] / B[i] <= t / (1-t) && t / (1-t) < A[i+1] / B[i+1]) {
                    //    tree2noIdentical.tauPartitions.get(b).tauCoordinate =
                    //            (t*B[j] - (1-t)*A[j])*tree2noIdentical.tauPartitions.get(b).tauCoordinate / B[j];
                    //}
                    tree2noIdentical.tauPartitions.get(b).tauCoordinate =
                            Math.max((t * B[j] - (1 - t) * A[j]) * tree2noIdentical.tauPartitions.get(b).tauCoordinate / B[j], 0); // TODO: This is a dodgy line I don't understand, but without it it's not workin :(
                    additionalTree.addPartition(tree2noIdentical.tauPartitions.get(b));
                }
            }

            // Adding partition corresponding to the origin-tree:
            for (int j = i; j < graphGeodesic.size(); j++) {
                for (int k = 0; k < graphGeodesic.get(j).get(0).size(); k++) {
                    int a = graphGeodesic.get(j).get(0).get(k);
                    //if (t/(1-t) <= A[0] / B[0]) {
                    //    tree1noIdentical.tauPartitions.get(a).tauCoordinate =
                    //            ((1-t)*A[j] - t*B[j])*tree1noIdentical.tauPartitions.get(a).tauCoordinate / A[j];
                    //}
                    //if (i < graphGeodesic.size() - 1 && A[i] / B[i] < t / (1-t) && t / (1-t) <= A[i+1] / B[i+1]) {
                    //    tree1noIdentical.tauPartitions.get(a).tauCoordinate =
                    //            ((1-t)*A[j] - t*B[j])*tree1noIdentical.tauPartitions.get(a).tauCoordinate / A[j];
                    //}
                    tree1noIdentical.tauPartitions.get(a).tauCoordinate =
                            Math.max(((1-t)*A[j] - t*B[j])*tree1noIdentical.tauPartitions.get(a).tauCoordinate / A[j] , 0); // TODO: This is a dodgy line I don't understand, but without it it's not workin :(
                    additionalTree.addPartition(tree1noIdentical.tauPartitions.get(a));
                }
            }
            tmpGeodesic.add(additionalTree);
        }

        // Choosing the tree at time t (the tau-coordinates are already assigned with respect to that tree):
        if (t/(1-t) <= A[0]/B[0]) {
            tmpTree = tmpGeodesic.get(0);
        }
        if (t/(1-t) > A[A.length - 1] / B[B.length - 1]) {
            tmpTree = tmpGeodesic.get(tmpGeodesic.size() - 1);
        }
        for (int i = 0; i < A.length - 1; i++) {
            if (A[i] / B[i] > A[i+1] / B[i+1]) {throw new RuntimeException("Wrong A[i]/B[i] sequence.");}
            if (A[i] / B[i] < t/(1-t) && t/(1-t) <= A[i+1] / B[i+1]) {
                tmpTree = tmpGeodesic.get(i+1);
            }
        }

        tmpGeo.tree = tmpTree;
        length = Math.sqrt(length);
        tmpGeo.geoLength = length;

        return tmpGeo;
    }
}
