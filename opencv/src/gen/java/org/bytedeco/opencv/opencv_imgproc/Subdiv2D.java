// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_imgproc;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;
import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgproc.*;


/** \} imgproc_hist
 <p>
 *  \addtogroup imgproc_subdiv2d
 *  \{ */

@Namespace("cv") @NoOffset @Properties(inherit = org.bytedeco.opencv.presets.opencv_imgproc.class)
public class Subdiv2D extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Subdiv2D(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Subdiv2D(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public Subdiv2D position(long position) {
        return (Subdiv2D)super.position(position);
    }
    @Override public Subdiv2D getPointer(long i) {
        return new Subdiv2D((Pointer)this).offsetAddress(i);
    }

    /** Subdiv2D point location cases */
    /** enum cv::Subdiv2D:: */
    public static final int /** Point location error */
 PTLOC_ERROR        = -2,
           /** Point outside the subdivision bounding rect */
           PTLOC_OUTSIDE_RECT = -1,
           /** Point inside some facet */
           PTLOC_INSIDE       = 0,
           /** Point coincides with one of the subdivision vertices */
           PTLOC_VERTEX       = 1,
           /** Point on some edge */
           PTLOC_ON_EDGE      = 2;

    /** Subdiv2D edge type navigation (see: getEdge()) */
    /** enum cv::Subdiv2D:: */
    public static final int NEXT_AROUND_ORG   = 0x00,
           NEXT_AROUND_DST   = 0x22,
           PREV_AROUND_ORG   = 0x11,
           PREV_AROUND_DST   = 0x33,
           NEXT_AROUND_LEFT  = 0x13,
           NEXT_AROUND_RIGHT = 0x31,
           PREV_AROUND_LEFT  = 0x20,
           PREV_AROUND_RIGHT = 0x02;

    /** creates an empty Subdiv2D object.
    To create a new empty Delaunay subdivision you need to use the #initDelaunay function.
     */
    public Subdiv2D() { super((Pointer)null); allocate(); }
    private native void allocate();

    /** \overload
    <p>
    @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
    <p>
    The function creates an empty Delaunay subdivision where 2D points can be added using the function
    insert() . All of the points to be added must be within the specified rectangle, otherwise a runtime
    error is raised.
     */
    public Subdiv2D(@ByVal Rect rect) { super((Pointer)null); allocate(rect); }
    private native void allocate(@ByVal Rect rect);

    /** \brief Creates a new empty Delaunay subdivision
    <p>
    @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
     <p>
     */
    public native void initDelaunay(@ByVal Rect rect);

    /** \brief Insert a single point into a Delaunay triangulation.
    <p>
    @param pt Point to insert.
    <p>
    The function inserts a single point into a subdivision and modifies the subdivision topology
    appropriately. If a point with the same coordinates exists already, no new point is added.
    @return the ID of the point.
    <p>
    \note If the point is outside of the triangulation specified rect a runtime error is raised.
     */
    public native int insert(@ByVal Point2f pt);

    /** \brief Insert multiple points into a Delaunay triangulation.
    <p>
    @param ptvec Points to insert.
    <p>
    The function inserts a vector of points into a subdivision and modifies the subdivision topology
    appropriately.
     */
    public native void insert(@Const @ByRef Point2fVector ptvec);

    /** \brief Returns the location of a point within a Delaunay triangulation.
    <p>
    @param pt Point to locate.
    @param edge Output edge that the point belongs to or is located to the right of it.
    @param vertex Optional output vertex the input point coincides with.
    <p>
    The function locates the input point within the subdivision and gives one of the triangle edges
    or vertices.
    <p>
    @return an integer which specify one of the following five cases for point location:
    -  The point falls into some facet. The function returns #PTLOC_INSIDE and edge will contain one of
       edges of the facet.
    -  The point falls onto the edge. The function returns #PTLOC_ON_EDGE and edge will contain this edge.
    -  The point coincides with one of the subdivision vertices. The function returns #PTLOC_VERTEX and
       vertex will contain a pointer to the vertex.
    -  The point is outside the subdivision reference rectangle. The function returns #PTLOC_OUTSIDE_RECT
       and no pointers are filled.
    -  One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error
       processing mode is selected, #PTLOC_ERROR is returned.
     */
    public native int locate(@ByVal Point2f pt, @ByRef IntPointer edge, @ByRef IntPointer vertex);
    public native int locate(@ByVal Point2f pt, @ByRef IntBuffer edge, @ByRef IntBuffer vertex);
    public native int locate(@ByVal Point2f pt, @ByRef int[] edge, @ByRef int[] vertex);

    /** \brief Finds the subdivision vertex closest to the given point.
    <p>
    @param pt Input point.
    @param nearestPt Output subdivision vertex point.
    <p>
    The function is another function that locates the input point within the subdivision. It finds the
    subdivision vertex that is the closest to the input point. It is not necessarily one of vertices
    of the facet containing the input point, though the facet (located using locate() ) is used as a
    starting point.
    <p>
    @return vertex ID.
     */
    public native int findNearest(@ByVal Point2f pt, Point2f nearestPt/*=0*/);
    public native int findNearest(@ByVal Point2f pt);

    /** \brief Returns a list of all edges.
    <p>
    @param edgeList Output vector.
    <p>
    The function gives each edge as a 4 numbers vector, where each two are one of the edge
    vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3].
     */
    public native void getEdgeList(@ByRef Vec4fVector edgeList);

    /** \brief Returns a list of the leading edge ID connected to each triangle.
    <p>
    @param leadingEdgeList Output vector.
    <p>
    The function gives one edge ID for each triangle.
     */
    public native void getLeadingEdgeList(@StdVector IntPointer leadingEdgeList);
    public native void getLeadingEdgeList(@StdVector IntBuffer leadingEdgeList);
    public native void getLeadingEdgeList(@StdVector int[] leadingEdgeList);

    /** \brief Returns a list of all triangles.
    <p>
    @param triangleList Output vector.
    <p>
    The function gives each triangle as a 6 numbers vector, where each two are one of the triangle
    vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
     */
    public native void getTriangleList(@ByRef Vec6fVector triangleList);

    /** \brief Returns a list of all Voronoi facets.
    <p>
    @param idx Vector of vertices IDs to consider. For all vertices you can pass empty vector.
    @param facetList Output vector of the Voronoi facets.
    @param facetCenters Output vector of the Voronoi facets center points.
     <p>
     */
    public native void getVoronoiFacetList(@StdVector IntPointer idx, @ByRef Point2fVectorVector facetList,
                                         @ByRef Point2fVector facetCenters);
    public native void getVoronoiFacetList(@StdVector IntBuffer idx, @ByRef Point2fVectorVector facetList,
                                         @ByRef Point2fVector facetCenters);
    public native void getVoronoiFacetList(@StdVector int[] idx, @ByRef Point2fVectorVector facetList,
                                         @ByRef Point2fVector facetCenters);

    /** \brief Returns vertex location from vertex ID.
    <p>
    @param vertex vertex ID.
    @param firstEdge Optional. The first edge ID which is connected to the vertex.
    @return vertex (x,y)
     <p>
     */
    public native @ByVal Point2f getVertex(int vertex, IntPointer firstEdge/*=0*/);
    public native @ByVal Point2f getVertex(int vertex);
    public native @ByVal Point2f getVertex(int vertex, IntBuffer firstEdge/*=0*/);
    public native @ByVal Point2f getVertex(int vertex, int[] firstEdge/*=0*/);

    /** \brief Returns one of the edges related to the given edge.
    <p>
    @param edge Subdivision edge ID.
    @param nextEdgeType Parameter specifying which of the related edges to return.
    The following values are possible:
    -   NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge)
    -   NEXT_AROUND_DST next around the edge vertex ( eDnext )
    -   PREV_AROUND_ORG previous around the edge origin (reversed eRnext )
    -   PREV_AROUND_DST previous around the edge destination (reversed eLnext )
    -   NEXT_AROUND_LEFT next around the left facet ( eLnext )
    -   NEXT_AROUND_RIGHT next around the right facet ( eRnext )
    -   PREV_AROUND_LEFT previous around the left facet (reversed eOnext )
    -   PREV_AROUND_RIGHT previous around the right facet (reversed eDnext )
    <p>
    ![sample output](pics/quadedge.png)
    <p>
    @return edge ID related to the input edge.
     */
    public native int getEdge( int edge, int nextEdgeType );

    /** \brief Returns next edge around the edge origin.
    <p>
    @param edge Subdivision edge ID.
    <p>
    @return an integer which is next edge ID around the edge origin: eOnext on the
    picture above if e is the input edge).
     */
    public native int nextEdge(int edge);

    /** \brief Returns another edge of the same quad-edge.
    <p>
    @param edge Subdivision edge ID.
    @param rotate Parameter specifying which of the edges of the same quad-edge as the input
    one to return. The following values are possible:
    -   0 - the input edge ( e on the picture below if e is the input edge)
    -   1 - the rotated edge ( eRot )
    -   2 - the reversed edge (reversed e (in green))
    -   3 - the reversed rotated edge (reversed eRot (in green))
    <p>
    @return one of the edges ID of the same quad-edge as the input edge.
     */
    public native int rotateEdge(int edge, int rotate);
    public native int symEdge(int edge);

    /** \brief Returns the edge origin.
    <p>
    @param edge Subdivision edge ID.
    @param orgpt Output vertex location.
    <p>
    @return vertex ID.
     */
    public native int edgeOrg(int edge, Point2f orgpt/*=0*/);
    public native int edgeOrg(int edge);

    /** \brief Returns the edge destination.
    <p>
    @param edge Subdivision edge ID.
    @param dstpt Output vertex location.
    <p>
    @return vertex ID.
     */
    public native int edgeDst(int edge, Point2f dstpt/*=0*/);
    public native int edgeDst(int edge);
}
