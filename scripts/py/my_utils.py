import numpy as np
import pandas as pd
from geopandas import gpd
from shapely.geometry import Point, Polygon, LineString, MultiLineString, mapping
from shapely.ops import linemerge
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from itertools import product
import networkx as nx
from osmnx import graph_to_gdfs, gdfs_to_graph, save_and_show, get_paths_to_simplify

def clean_intersections_graph(G, tolerance=15, dead_ends=False):
    """
    Clean-up intersections comprising clusters of nodes by merging them and
    returning a modified graph.

    Divided roads are represented by separate centerline edges. The intersection
    of two divided roads thus creates 4 nodes, representing where each edge
    intersects a perpendicular edge. These 4 nodes represent a single
    intersection in the real world. This function cleans them up by buffering
    their points to an arbitrary distance, merging overlapping buffers, and
    taking their centroid. For best results, the tolerance argument should be
    adjusted to approximately match street design standards in the specific
    street network.

    Parameters
    ----------
    G : networkx multidigraph
    tolerance : float
        nodes within this distance (in graph's geometry's units) will be
        dissolved into a single intersection
    dead_ends : bool
        if False, discard dead-end nodes to return only street-intersection
        points

    Returns
    ----------
    Networkx graph with the new aggregated vertices and induced edges
    """

    # if dead_ends is False, discard dead-end nodes to only work with edge
    # intersections
    if not dead_ends:
        if 'streets_per_node' in G.graph:
            streets_per_node = G.graph['streets_per_node']
        else:
            streets_per_node = count_streets_per_node(G)

        dead_end_nodes = [node for node, count in streets_per_node.items() if count <= 1]
        G = G.copy()
        G.remove_nodes_from(dead_end_nodes)

    # create a GeoDataFrame of nodes, buffer to passed-in distance, merge
    # overlaps
    gdf_nodes, gdf_edges = graph_to_gdfs(G)
    buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
    if isinstance(buffered_nodes, Polygon):
        # if only a single node results, make it iterable so we can turn it into
        # a GeoSeries
        buffered_nodes = [buffered_nodes]

    # Buffer points by tolerance and union the overlapping ones
    gdf_nodes, gdf_edges = graph_to_gdfs(G)
    buffered_nodes = gdf_nodes.buffer(15).unary_union
    unified_intersections = gpd.GeoSeries(list(buffered_nodes))
    unified_gdf = gpd.GeoDataFrame(unified_intersections).rename(columns={0:'geometry'}).set_geometry('geometry')
    unified_gdf.crs = gdf_nodes.crs

    ### Merge original nodes with the aggregated shapes
    intersections = gpd.sjoin(gdf_nodes, unified_gdf, how="right", op='intersects')
    intersections['geometry_str'] = intersections['geometry'].map(lambda x: str(x))
    intersections['new_osmid'] = intersections.groupby('geometry_str')['index_left'].transform('min').astype(str)
    intersections['num_osmid_agg'] = intersections.groupby('geometry_str')['index_left'].transform('count')

    ### Create temporary lookup with the agg osmid and the new one
    lookup = intersections[intersections['num_osmid_agg']>1][['osmid', 'new_osmid', 'num_osmid_agg']]
    lookup = lookup.rename(columns={'osmid': 'old_osmid'})
    intersections = intersections[intersections['osmid'].astype(str)==intersections['new_osmid']]
    intersections = intersections.set_index('index_left')

    ### Make everything else similar to original node df
    intersections = intersections[gdf_nodes.columns]
    intersections['geometry'] = intersections.geometry.centroid
    intersections['x'] = intersections.geometry.x
    intersections['y'] = intersections.geometry.y
    del intersections.index.name
    intersections.gdf_name = gdf_nodes.gdf_name

    # Replace aggregated osimid with the new ones
    # 3 cases - 1) none in lookup, 2) either u or v in lookup, 3) u and v in lookup
    # Ignore case 1. Append case 3 to case 2. ignore distance but append linestring.

    # removed .astype(str) from merger after u a nd v

    agg_gdf_edges = pd.merge(gdf_edges.assign(u=gdf_edges.u),
                        lookup.rename(columns={'new_osmid': 'new_osmid_u', 'old_osmid': 'old_osmid_u'}),
                        left_on='u', right_on='old_osmid_u', how='left')
    agg_gdf_edges = pd.merge(agg_gdf_edges.assign(v=agg_gdf_edges.v),
                        lookup.rename(columns={'new_osmid': 'new_osmid_v', 'old_osmid': 'old_osmid_v'}),
                        left_on='v', right_on='old_osmid_v', how='left')

    # Remove all u-v edges that are between the nodes that are aggregated together (case 3)
    agg_gdf_edges_c3 = agg_gdf_edges[((agg_gdf_edges['new_osmid_v'].notnull()) &
        (agg_gdf_edges['new_osmid_u'].notnull()) &
        (agg_gdf_edges['new_osmid_u'] == agg_gdf_edges['new_osmid_v']))]

    agg_gdf_edges = agg_gdf_edges[~agg_gdf_edges.index.isin(agg_gdf_edges_c3.index)]

    # Create a self loop containing all the joint geometries of the aggregated nodes where both u and v are agg
    # Set onway to false to prevent duplication if someone were to create bidrectional edges
    agg_gdf_edges_int = agg_gdf_edges_c3[~((agg_gdf_edges_c3['new_osmid_u'] == agg_gdf_edges_c3['u']) |
                                        (agg_gdf_edges_c3['new_osmid_v'] == agg_gdf_edges_c3['v']))]
    agg_gdf_edges_int = agg_gdf_edges_int.dissolve(by=['new_osmid_u', 'new_osmid_v']).reset_index()
    agg_gdf_edges_int['u'] = agg_gdf_edges_int['new_osmid_u']
    agg_gdf_edges_int['v'] = agg_gdf_edges_int['new_osmid_v']
    agg_gdf_edges_int = agg_gdf_edges_int[gdf_edges.columns]
    agg_gdf_edges_int['oneway'] = False

    # Simplify by removing edges that do not involve the chosen agg point
    # at least one of them must contain the new u or new v
    agg_gdf_edges_c3 = agg_gdf_edges_c3[(agg_gdf_edges_c3['new_osmid_u'] == agg_gdf_edges_c3['u']) |
                                        (agg_gdf_edges_c3['new_osmid_v'] == agg_gdf_edges_c3['v'])]

    agg_gdf_edges_c3 = agg_gdf_edges_c3[['geometry', 'u', 'v', 'new_osmid_u', 'new_osmid_v']]
    agg_gdf_edges_c3.columns = ['old_geometry', 'old_u', 'old_v', 'new_osmid_u', 'new_osmid_v']

    # Merge back the linestring for case 2
    # Ignore u and v if they are on the merging / agg node
    # Copy over the linestring only on the old node
    subset_gdf = agg_gdf_edges_c3[agg_gdf_edges_c3['new_osmid_v']!=agg_gdf_edges_c3['old_v']]
    agg_gdf_edges = pd.merge(agg_gdf_edges, subset_gdf[['old_geometry', 'old_v']],
                             how='left', left_on='u', right_on='old_v')

    geom = agg_gdf_edges[['geometry', 'old_geometry']].values.tolist()
    agg_gdf_edges['geometry'] = [linemerge([r[0], r[1]]) if isinstance(r[1], (LineString, MultiLineString)) else r[0] for r in geom]
    agg_gdf_edges.drop(['old_geometry', 'old_v'], axis=1, inplace=True)

    # If new osmid matches on u, merge in the existing u-v string
    # where u is the aggregated vertex and v is the old one to be removed

    subset_gdf = agg_gdf_edges_c3[agg_gdf_edges_c3['new_osmid_u']!=agg_gdf_edges_c3['old_u']]
    agg_gdf_edges = pd.merge(agg_gdf_edges, subset_gdf[['old_geometry', 'old_u']],
                             how='left', left_on='v', right_on='old_u')

    geom = agg_gdf_edges[['geometry', 'old_geometry']].values.tolist()
    agg_gdf_edges['geometry'] = [linemerge([r[0], r[1]]) if isinstance(r[1], (LineString, MultiLineString)) else r[0] for r in geom]
    agg_gdf_edges.drop(['old_geometry', 'old_u'], axis=1, inplace=True)

    agg_gdf_edges['u'] = np.where(agg_gdf_edges['new_osmid_u'].notnull(), agg_gdf_edges['new_osmid_u'], agg_gdf_edges['u'])
    agg_gdf_edges['v'] = np.where(agg_gdf_edges['new_osmid_v'].notnull(), agg_gdf_edges['new_osmid_v'], agg_gdf_edges['v'])
    agg_gdf_edges = agg_gdf_edges[gdf_edges.columns]
    agg_gdf_edges = gpd.GeoDataFrame(pd.concat([agg_gdf_edges, agg_gdf_edges_int], ignore_index=True),
                                     crs=agg_gdf_edges.crs)

    agg_gdf_edges['u'] = agg_gdf_edges['u'].astype(np.int64)
    agg_gdf_edges['v'] = agg_gdf_edges['v'].astype(np.int64)

    return gdfs_to_graph(intersections, agg_gdf_edges)

def plot_graph_mls(G, bbox=None, fig_height=6, fig_width=None, margin=0.02,
               axis_off=True, equal_aspect=False, bgcolor='w', show=True,
               save=False, close=True, file_format='png', filename='temp',
               dpi=300, annotate=False, node_color='#66ccff', node_size=15,
               node_alpha=1, node_edgecolor='none', node_zorder=1,
               edge_color='#999999', edge_linewidth=1, edge_alpha=1,
               use_geom=True):
    """
    Plot a networkx spatial graph. Modified to accept MultiLineString.

    Parameters
    ----------
    G : networkx multidigraph
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    equal_aspect : bool
        if True set the axis aspect ratio equal
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node

    Returns
    -------
    fig, ax : tuple
    """

    node_Xs = [float(x) for _, x in G.nodes(data='x')]
    node_Ys = [float(y) for _, y in G.nodes(data='y')]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        west, south, east, north = edges.total_bounds
    else:
        north, south, east, west = bbox

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north-south)/(east-west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio

    # create the figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
    ax.set_facecolor(bgcolor)

    # draw the edges as lines from node to node
    # start_time = time.time()
    lines = []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry' in data and use_geom:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            if isinstance(data['geometry'], MultiLineString):
                lines += [list(t) for t in mapping(data['geometry'])['coordinates']]
            else:
                lines += [list(mapping(data['geometry'])['coordinates'])]

        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha, edgecolor=node_edgecolor, zorder=node_zorder)

    # set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    # configure axis appearance
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()

    xaxis.get_major_formatter().set_useOffset(False)
    yaxis.get_major_formatter().set_useOffset(False)

    # if axis_off, turn off the axis display set the margins to zero and point
    # the ticks in so there's no space around the plot
    if axis_off:
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(which='both', direction='in')
        xaxis.set_visible(False)
        yaxis.set_visible(False)
        fig.canvas.draw()

    if equal_aspect:
        # make everything square
        ax.set_aspect('equal')
        fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x'], data['y']))

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax