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
// import beast.util.NexusParser;
import beast.util.TreeParser;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.Date;
import java.util.List;
import java.util.Arrays;

/**
 * Parsing the file, calling mean, returning the mean.
 *
 * Created on 14/11/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */
public class Main {

    public static void main(String[] args) throws Exception {

        if (args.length > 0) {
            Path p = Paths.get(args[0]);
	    String newick = new String(Files.readAllBytes(p));
            Path p1 = Paths.get(args[1]);
	    String newick1 = new String(Files.readAllBytes(p1));
            // NexusParser parser = new NexusParser();
            TreeParser parser = new TreeParser(newick);
            TreeParser parser1 = new TreeParser(newick1);
            // File treeFile = new File(args[0]);
            // parser.parseFile(treeFile);
            List<Tree> trees = Arrays.asList(parser, parser1);

            // Putting trees into an array to save 2 seconds per tree when converting them to tau-trees:
            Tree[] treesArray = new Tree[trees.size()];
            for (int i = 0; i < trees.size(); i++) {
                treesArray[i] = trees.get(i);
            }

            // Converting the trees to tau-trees:
            TauTree[] tauTrees = new TauTree[trees.size()];
            int numberOfTreesPassed = 0;
            for (int i = 0; i < treesArray.length; i++) {
                tauTrees[i] = new TauTree(treesArray[i]);
                numberOfTreesPassed=i+1;
            }

            System.out.println("The number of trees passed is " + numberOfTreesPassed + ".\n");
            Date date = new Date();
            System.out.println("The tau-trees have been created on " + date + ". Start computing distance\n");
            System.out.println(
                Geodesic.geodesic(tauTrees[0],tauTrees[1],0.9).geoLength);
        }
    }
}
