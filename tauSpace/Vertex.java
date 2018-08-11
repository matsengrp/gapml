import java.util.ArrayList;

/**
 * Created on 22/10/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */

public class Vertex  {
    public boolean[][] matrix;
    public ArrayList<Vertex> adjacent = new ArrayList<Vertex>();
    public double weight;

    public Vertex(int numTaxa) {
        matrix = new boolean[numTaxa][numTaxa];
    }
}
