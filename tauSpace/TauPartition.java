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


import beast.evolution.tree.Node;
import beast.evolution.tree.Tree;
import beast.evolution.tree.TreeUtils;

import java.util.Map;
import java.util.TreeSet;

/**
 * A tau-partition is a boolean matrix with a real tau-coordinate.
 *
 * Created 22/10/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */

public class TauPartition {
    public boolean[][] matrix;
    public double tauCoordinate;

    public TauPartition() {
    }
    
    public TauPartition(Node node, double tau, Node[][] ancestors) {
        int numTaxa = node.getTree().getLeafNodeCount();
        matrix = new boolean[numTaxa][numTaxa];
        tauCoordinate = tau;
        for (int i = 0; i < numTaxa; i++) {
            for (int j = 0; j < numTaxa; j++) {
                if (i == j) {
                    matrix[i][j] = true;
                } else matrix[i][j] = (ancestors[i][j].getHeight() <= node.getHeight());
            }
        }
    }

    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append("tau="+tauCoordinate+" matrix:\n");
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                builder.append((matrix[i][j] ? 1 : 0) + " ");
            }
            builder.append("\n");
        }
        return builder.toString();
    }

    public boolean equalMatrix(TauPartition p) {

        boolean[][] m1 = matrix;
        boolean[][] m2 = p.matrix;

        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[i].length; j++) {
                if (m1[i][j] != m2[i][j]) return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {

        boolean[][] m1 = {{true, false}, {true, false}};
        boolean[][] m2 = {{true, false}, {true, false}};

        System.out.println(m1.equals(m2));
    }
}
