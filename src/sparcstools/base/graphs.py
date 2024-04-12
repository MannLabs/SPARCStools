from graph_tool import Graph
from graph_tool.topology import shortest_distance
from networkx import Graph as nxGraph

#calcultion of centers is equivalent to method in networkx: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.center.html
def get_center_nodes(g):

    """Calculate the center node of the graph.
    
    The center is the set of nodes with eccentricity equal to radius.
    
    Parameters
    ----------
    g : graph-tool Graph
        The graph for which to calculate the center node.   
    
    Returns
    -------
    centers : int
        index of the center node of the graph.
    """

    nodes = g.get_vertices()
    
    if len(nodes) > 2:
        
        distances = shortest_distance(g)
        eccentricities = [x.a.max() for x in distances]
        
        centers = nodes[eccentricities.index(min(eccentricities))]

        return(centers)

    #if only one node or two nodes are present in the graph the one with the lower index is returned    
    else:
        return(nodes[0]) 

#code for conversion of networkx graph to graph-tool graph
#adapted from https://gist.github.com/bbengfort/a430d460966d64edc6cad71c502d7005

def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    if isinstance(key, str):
        # Ensure the key is in ASCII format
        key = key.encode('ascii', errors='replace').decode('ascii')

    # Deal with the value
    if isinstance(value, bool):
        tname = 'bool'

    elif isinstance(value, int):
        tname = 'float'
        value = float(value)

    elif isinstance(value, float):
        tname = 'float'

    elif isinstance(value, str):
        tname = 'string'
        value = value.encode('ascii', errors='replace').decode('ascii')

    elif isinstance(value, dict):
        tname = 'object'

    else:
        tname = 'string'
        value = str(value)

    return tname, value, key


def nx2gt(nxG):
    """
    Converts a networkx graph to a graph-tool graph.

    This implementation is based on a gist by Benjamin Bengfort: https://gist.github.com/bbengfort/a430d460966d64edc6cad71c502d7005
    
    Parameters
    ----------
    nxG : networkx.Graph
        The networkx graph to convert.
    
    Returns
    -------
    gtG : graph-tool.Graph
        The graph-tool graph.
    """

    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = Graph(directed=nxG.is_directed())

    # Add the Graph properties as "internal properties"
    for key, value in nxG.graph.items():
        # Convert the value and key into a type for graph-tool
        tname, value, key = get_prop_type(value, key)

        prop = gtG.new_graph_property(tname) # Create the PropertyMap
        gtG.graph_properties[key] = prop     # Set the PropertyMap
        gtG.graph_properties[key] = value    # Set the actual value

    # Phase 1: Add the vertex and edge property maps
    # Go through all nodes and edges and add seen properties
    # Add the node properties first
    nprops = set() # cache keys to only add properties once
    for node, data in nxG.nodes(data=True):

        # Go through all the properties if not seen and add them.
        for key, val in data.items():
            if key in nprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key  = get_prop_type(val, key)

            prop = gtG.new_vertex_property(tname) # Create the PropertyMap
            gtG.vertex_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            nprops.add(key)

    # Also add the node id: in NetworkX a node can be any hashable type, but
    # in graph-tool node are defined as indices. So we capture any strings
    # in a special PropertyMap called 'id' -- modify as needed!
    gtG.vertex_properties['id'] = gtG.new_vertex_property('string')

    # Add the edge properties second
    eprops = set() # cache keys to only add properties once
    for src, dst, data in nxG.edges(data=True):

        # Go through all the edge properties if not seen and add them.
        for key, val in data.items():
            if key in eprops: continue # Skip properties already added

            # Convert the value and key into a type for graph-tool
            tname, _, key = get_prop_type(val, key)

            prop = gtG.new_edge_property(tname) # Create the PropertyMap
            gtG.edge_properties[key] = prop     # Set the PropertyMap

            # Add the key to the already seen properties
            eprops.add(key)

    # Phase 2: Actually add all the nodes and vertices with their properties
    # Add the nodes
    vertices = {} # vertex mapping for tracking edges later
    for node, data in nxG.nodes(data=True):

        # Create the vertex and annotate for our edges later
        v = gtG.add_vertex()
        vertices[node] = v

        # Set the vertex properties, not forgetting the id property
        data['id'] = str(node)
        for key, value in data.items():
            gtG.vp[key][v] = value # vp is short for vertex_properties

    # Add the edges
    for src, dst, data in nxG.edges(data=True):

        # Look up the vertex structs from our vertices mapping and add edge.
        e = gtG.add_edge(vertices[src], vertices[dst])

        # Add the edge properties
        for key, value in data.items():
            gtG.ep[key][e] = value # ep is short for edge_properties

    # Done, finally!
    return gtG



def gt2nx(gtG):
    """
    Converts a graph-tool graph to a NetworkX graph.

    Parameters
    ----------
    gtG : graph-tool.Graph
        The graph-tool graph to convert.

    Returns
    -------
    nxG : networkx.Graph
        The NetworkX graph.
    """
    # Create a new NetworkX graph
    nxG = nxGraph()

    # Add graph properties
    for key, value in gtG.graph_properties.items():
        nxG.graph[key] = value

    # Add vertex properties
    for key, value in gtG.vertex_properties.items():
        nxG.graph[key] = value

    # Add vertices
    vertex_id_map = {}  # Mapping from graph-tool vertex index to NetworkX node ID
    for v in gtG.vertices():
        vertex_id = int(gtG.vp['id'][v])
        vertex_id_map[v] = vertex_id
        nxG.add_node(vertex_id, **{key: value[v] for key, value in gtG.vp.items() if key != 'id'})

    # Add edges
    for e in gtG.edges():
        source = vertex_id_map[e.source()]
        target = vertex_id_map[e.target()]
        nxG.add_edge(source, target, **{key: value[e] for key, value in gtG.ep.items()})

    return nxG
