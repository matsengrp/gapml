/*
 * Copyright (C) 2014 Alex Gavryushkin <alex@gavruskin.com> and Alexei Drummond
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
import beast.util.TreeParser;

import java.util.*;

/**
 * Methods for converting BEAST-tree to tau-tree and back are in this class.
 *
 * Created on 22/10/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 * @author Alexei Drummond
 */

public class TauTree {

    int numTaxa;
    ArrayList<TauPartition> tauPartitions = new ArrayList<TauPartition>();
    Node[][] ancestors;
    double firstTau;

    public TauTree() {
    }

    public TauTree(Tree tree) {
        constructFromTree(tree);
    }

    public Map<String, Integer> labelMap = new TreeMap<String, Integer>();

    private void constructFromTree(Tree tree) {
        numTaxa = tree.getLeafNodeCount();
        List<Node> nodes = tree.getInternalNodes();
        List<String> labels = tree.getTaxonset().asStringList();
        // Collections.sort(labels);
        Integer s = 0;
        for (String label : labels) {
            labelMap.put(label, s);
            s = s + 1;
        }
        ancestors = new Node[numTaxa][numTaxa];
        fillAncestors(tree, tree.getRoot(), ancestors, labelMap);
        Collections.sort(nodes, (o1, o2) -> Double.compare(o1.getHeight(), o2.getHeight()));
        for (int i = 0; i < nodes.size(); i++) {
            Node node = nodes.get(i);
            if (!node.isRoot()) {
                double tau = nodes.get(i + 1).getHeight() - node.getHeight();
                // tauPartitions.add(new TauPartition(node, tau, ancestors));
                addPartition(new TauPartition(node, tau, ancestors));
            }
        }
        firstTau = nodes.get(0).getHeight();
    }

    public void addPartition(TauPartition p) {

        for (TauPartition t : tauPartitions) {
            if (t.equalMatrix(p)){
 	        throw new RuntimeException("Trying to add identical partition!");
		// System.out.println("again???!!! weird");
	        // return;
	    }
        }
        TauPartition tmp = new TauPartition();
        tmp.matrix = new boolean[p.matrix.length][p.matrix[0].length];
        for (int i = 0; i < p.matrix.length; i++) {
            for (int j = 0; j < p.matrix[i].length; j++) {
                tmp.matrix[i][j] = p.matrix[i][j];
            }
        }
        tmp.tauCoordinate = p.tauCoordinate;
        tauPartitions.add(tmp);
    }

    private static List<Integer> fillAncestors(Tree tree, Node node, Node[][] ancestors, Map<String, Integer> labelMap) {
        if (node.isLeaf()) {
            ArrayList<Integer> l = new ArrayList<Integer>();
            l.add(labelMap.get(node.getID()));
            return l;
        } else {
            List<Integer> left = fillAncestors(tree, node.getChild(0), ancestors, labelMap);
            List<Integer> right = fillAncestors(tree, node.getChild(1), ancestors, labelMap);

            for (int i : left) {
                for (int j : right) {
                    if (ancestors[i][j] == null) {
                        ancestors[i][j] = node;
                        ancestors[j][i] = node;
                    }
                }
            }

            left.addAll(right);
            return left;
        }
    }

    public static Tree constructFromTauTree(TauTree tauTree) {
        Tree tmpTree = new Tree();

        //Sorting tau-partitions just in case:
        Comparator<TauPartition> comparator = new Comparator<TauPartition>() {
            @Override
            public int compare(TauPartition o1, TauPartition o2) {
                boolean[] result = Geodesic.areCompatible(o1.matrix, o2.matrix);
                //if (result[3]) return 0; TODO: This is not needed unless a partition is added without using ^ addPartition.
                if (result[1] && !result[2]) return -1;
                if (result[2] && !result[1]) return 1;
                throw new RuntimeException("Incomparable or identical matrices!");
            }
        };
        Collections.sort(tauTree.tauPartitions, comparator);

        TauPartition previous = null;
        ArrayList<Node> leafNodes = new ArrayList<>();
        ArrayList<Node> activeNodes = new ArrayList<>();
        for (String label : tauTree.labelMap.keySet()) {
            try {
                Node node = new Node(label);
                node.setHeight(0.0);
                node.setNr(tauTree.labelMap.get(label));
                leafNodes.add(node);
                activeNodes.add(node);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        //System.out.println("Starting with " + leafNodes.size() + " nodes.");

        Collections.sort(leafNodes, (o1, o2) -> Integer.compare(o1.getNr(), o2.getNr()));

        double time = tauTree.getFirstTau();
        int nodeNum = activeNodes.size();
        for (TauPartition current : tauTree.tauPartitions) {

            //System.out.println(current.toString());

            Set<Node> newNodes = getNew(current, previous, leafNodes);

            //System.out.println(newNodes.size() + " new nodes: " + newNodes);

            Node parent = new Node();
            parent.setNr(nodeNum);
            parent.setHeight(time);
            for (Node child : newNodes) {
                activeNodes.remove(child);
                parent.addChild(child);
            }
            activeNodes.add(parent);
            previous = current;
            time += current.tauCoordinate;
            nodeNum += 1;
        }

        if (activeNodes.size() > 1) {
            Node root = new Node();
            root.setHeight(time);
            root.setNr(nodeNum);
            for (Node child : activeNodes) {
                root.addChild(child);
            }
            tmpTree.setRoot(root);

            return tmpTree;

        } else {
            throw new RuntimeException("Unexpected size of activeList (" + activeNodes.size() + "), should be >1.");
        }
    }

    private static Set<Node> getNew(TauPartition current, TauPartition previous, List<Node> leafNodes) {

        if (previous != null && Geodesic.areCompatible(current.matrix, previous.matrix)[3]) {
            throw new RuntimeException("Matrices are identical!");
        }

        HashSet<Node> newNodes = new HashSet<Node>();

        for (int i = 0; i < current.matrix.length; i++) {
            for (int j = i + 1; j < current.matrix[i].length; j++) {
                if (previous == null || !previous.matrix[i][j]) {
                    if (current.matrix[i][j]) {
                        newNodes.add(getOldestParent(leafNodes.get(i)));
                        newNodes.add(getOldestParent(leafNodes.get(j)));
                    }
                }
            }
        }
        return newNodes;
    }

    private static Node getOldestParent(Node node) {
        if (node.getParent() == null) return node;
        return getOldestParent(node.getParent());
    }

    public static void main(String[] args) throws Exception {

        String newick1 = "(((2:2,3:2):2,(1:3,6:3):1):5,(4:1,5:1):8);";
        TreeParser tree1 = new TreeParser(newick1);
        TauTree ttree1 = new TauTree(tree1);

        Tree tree = constructFromTauTree(ttree1);

        System.out.println(tree.getRoot().toNewick());
    }

    public double getFirstTau() {
        return firstTau;
    }
}
