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

import java.lang.reflect.Array;
import java.util.ArrayList;

import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * vertex_cover returns int[3][n] where n = max(Aindex.geoLength(),Bindex.geoLength()) such that
 * vertex_cover[0][0] = ``the number of elements in the cover on the left-side'';
 * vertex_cover[0][i] = 0;
 * vertex_cover[1][0] = ``the number of elements in the cover on the right-side'';
 * vertex_cover[1][i] = 0;
 * vertex_cover[2][i] = ``one-dimensional array of vertices in the cover on the left-side'';
 * vertex_cover[3][i] = ``one-dimensional array of vertices in the cover on the left-side''.
 *
 * vertex_cover operates on indices, that is, it takes indices of the vertices as an input and
 * outputs indices of the vertices in the min-weight vertex cover.
 */

/* BipartiteGraph constructs and manipulates
 * a bipartite node-weighted graph, starting as
 * input with the node-node incidence matrix of
 * the graph plus the (squared) weights on the nodes sets.
 *
 * Most of this class is taken from Megan Owen's implementation of her geodesic algorithm for BHV space.
 * Created on 24/10/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */

public class BipartiteGraph {

    public double[] AweightNormalised;
    public double[] BweightNormalised;
        private boolean[][] edge; //node-node incidence matrix
        private int nA,nB,n,i,j;  //nA=#A-side nodes, nB=#B-side nodes, n=max(nA,nB)
        public VertexSTcut[] Avertex, Bvertex; //keeps information on the vertices

        public BipartiteGraph(boolean IncidenceMatrix[][], double Aweight[], double Bweight[]) {
            nA= Array.getLength(Aweight); nB=Array.getLength(Bweight); n=max(nA,nB);
            edge = IncidenceMatrix;
            Avertex = new VertexSTcut[n];
            Bvertex = new VertexSTcut[n];
            for (i=0;i<=nA-1;i++) Avertex[i]= new VertexSTcut(Aweight[i]);
            for (j=0;j<=nB-1;j++) Bvertex[j]= new VertexSTcut(Bweight[j]);
        }

        public int[][] vertex_cover(int[] Aindex, int[] Bindex) {

            int nAVC=Array.getLength(Aindex),nBVC=Array.getLength(Bindex); //nAVC,nBVC=size of A and B
            double total;
            double[][] ABflow=new double[nA][nB];
            int i, j, k, AScanListSize, BScanListSize, augmentingPathEnd=-1, Apathnode, Bpathnode;
            int[][] CD=new int[4][n]; //output: incidence vectors of vertex covers, CD[0]=Aside; CD[1]=Bside;
            int[] AScanList=new int[nA], BScanList=new int[nB]; //list of newly scanned nodes

		/* First set normalized weights */
            total=0;
            for(i=0;i<=nAVC-1;i++) total=total+Avertex[Aindex[i]].weight;
            for(i=0;i<=nAVC-1;i++) Avertex[Aindex[i]].residual=Avertex[Aindex[i]].weight/total;
            total=0;
            for(j=0;j<=nBVC-1;j++) total=total+Bvertex[Bindex[j]].weight;
            for(j=0;j<=nBVC-1;j++) Bvertex[Bindex[j]].residual=Bvertex[Bindex[j]].weight/total;

            AweightNormalised = new double[Avertex.length];// These two are for the `main loop.'
            BweightNormalised = new double[Bvertex.length];
            for (int l = 0; l < nAVC; l++) {
                AweightNormalised[l] = Avertex[Aindex[l]].residual;
            }

            for (int l = 0; l < nBVC; l++) {
                BweightNormalised[l] = Bvertex[Bindex[l]].residual;
            }

		/* Now comes the flow algorithm
		 * Flow on outside arcs are represented by Vertex[i].residual
		 * Flow on inside arcs are represented by ABflow
		 * Initialize ABflow to 0, start scanlist
		 */

            for(i=0;i<=nA-1;i++) for(j=0;j<=nB-1;j++) ABflow[i][j]=0;
            total=1; //flow augmentation in last stage
            while(total>0){

                //Scan Phase
                //Set labels
                total=0;
                for(i=0;i<=nAVC-1;i++) {Avertex[Aindex[i]].label=-1; Avertex[Aindex[i]].pred=-1;}
                for(j=0;j<=nBVC-1;j++) {Bvertex[Bindex[j]].label=-1; Bvertex[Bindex[j]].pred=-1;}
                AScanListSize=0;
                for(i=0;i<=nAVC-1;i++){
                    if (Avertex[Aindex[i]].residual>0){
                        Avertex[Aindex[i]].label=Avertex[Aindex[i]].residual;
                        AScanList[AScanListSize]=Aindex[i]; AScanListSize++;
                    }
                    else Avertex[Aindex[i]].label=-1;
                }
                for(i=0;i<=nBVC-1;i++) Bvertex[i].label=-1;

                // scan for an augmenting path
                scanning: while(AScanListSize!=0) {
				/* Scan the A side nodes*/
                    BScanListSize=0;
                    for(i=0;i<=AScanListSize-1;i++)
                        for(j=0;j<=nBVC-1;j++)
                            if (edge[AScanList[i]][Bindex[j]] && Bvertex[Bindex[j]].label==-1){
                                Bvertex[Bindex[j]].label=Avertex[AScanList[i]].label; Bvertex[Bindex[j]].pred=AScanList[i];
                                BScanList[BScanListSize]=Bindex[j]; BScanListSize++;
                            }

				/* Scan the B side nodes*/
                    AScanListSize=0;
                    for(j=0;j<=BScanListSize-1;j++)
                        if (Bvertex[BScanList[j]].residual>0) {
                            total=min(Bvertex[BScanList[j]].residual,Bvertex[BScanList[j]].label);
                            augmentingPathEnd=BScanList[j];
                            break scanning;
                        }
                        else for(i=0;i<=nAVC-1;i++)
                            if (edge[Aindex[i]][BScanList[j]] && Avertex[Aindex[i]].label==-1 && ABflow[Aindex[i]][BScanList[j]]>0) {
                                Avertex[Aindex[i]].label=min(Bvertex[BScanList[j]].label,ABflow[Aindex[i]][BScanList[j]]);
                                Avertex[Aindex[i]].pred=BScanList[j];
                                AScanList[AScanListSize]=Aindex[i];AScanListSize++;
                            }
                }//scanning procedure

                if (total>0) { // flow augmentation
                    Bvertex[augmentingPathEnd].residual=Bvertex[augmentingPathEnd].residual-total;
                    Bpathnode=augmentingPathEnd; Apathnode=Bvertex[Bpathnode].pred;

                    ABflow[Apathnode][Bpathnode]=ABflow[Apathnode][Bpathnode]+total;
                    while (Avertex[Apathnode].pred!=-1) {
                        Bpathnode=Avertex[Apathnode].pred;
                        ABflow[Apathnode][Bpathnode]=ABflow[Apathnode][Bpathnode]-total;
                        Apathnode=Bvertex[Bpathnode].pred;
                        ABflow[Apathnode][Bpathnode]=ABflow[Apathnode][Bpathnode]+total;

                    }
                    Avertex[Apathnode].residual=Avertex[Apathnode].residual-total;

                }
                else { //min vertex cover found, unlabeled A's, labeled B's
                    k=0;
                    for (i=0;i<=nAVC-1;i++)
                        if (Avertex[Aindex[i]].label==-1) {
                            CD[2][k]=Aindex[i];
                            k++;
                        }
                    CD[0][0]=k;
                    k=0;
                    for (j=0;j<=nBVC-1;j++)
                        if (Bvertex[Bindex[j]].label>=0) {
                            CD[3][k]=Bindex[j];
                            k++;
                        }
                    CD[1][0]=k;
                }
            }//flow algorithm
            return CD;

        } //vertex_cover

    public ArrayList<Integer> vertex_cover1list(int[] Aindex, int[] Bindex) {
        ArrayList<Integer> tmpList = new ArrayList<>();
        int[][] vcover = vertex_cover(Aindex,Bindex);
        for (int k = 0; k < vcover[0][0]; k++) {
            tmpList.add(vcover[2][k]);
        }
        return tmpList;
    }

    public ArrayList<Integer> vertex_cover2list(int[] Aindex, int[] Bindex) {
        ArrayList<Integer> tmpList = new ArrayList<>();
        int[][] vcover = vertex_cover(Aindex,Bindex);
        for (int k = 0; k < vcover[1][0]; k++) {
            tmpList.add(vcover[3][k]);
        }
        return tmpList;
    }



}