/**
 * Taken from Megan Owen's implementation of her geodesic algorithm for BHV space.
 * Created 29/10/14.
 * @author Alex Gavryushkin <alex@gavruskin.com>
 */

public class VertexSTcut {
    public double label, weight, residual; // -1 unlabeled, otherwise max flow to that vertex
    public int pred; //-1 = unscanned, otherwise predecessor

    public VertexSTcut(double weight) {this.weight=weight*weight;};

}