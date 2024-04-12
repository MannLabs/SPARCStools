"""_custom_ashalar_funcs

functions adapted from https://github.com/labsyspharm/ashlar to provide specific functionality required for this package.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

#added requirements for parallelization
from threading import Lock
import numpy.typing as npt
from tqdm.auto import tqdm
from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy as copy
import sys
import concurrent.futures

import multiprocessing
from tqdm import tqdm

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ashlar import utils as utils  # Import your utils module here

def draw_mosaic_image(ax, aligner, img, **kwargs):
    if img is None:
        img = [[0]]
    h, w = aligner.mosaic_shape
    ax.imshow(img, extent=(-0.5, w-0.5, h-0.5, -0.5), **kwargs)

def plot_edge_quality(
    aligner, outdir, img=None, show_tree=True, pos='metadata', im_kwargs=None, nx_kwargs=None
):
    if pos == 'metadata':
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == 'aligner':
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    if im_kwargs is None:
        im_kwargs = {}
    if nx_kwargs is None:
        nx_kwargs = {}
    final_nx_kwargs = dict(width=2, node_size=100, font_size=6)
    final_nx_kwargs.update(nx_kwargs)
    if show_tree:
        nrows, ncols = 1, 2
        if aligner.mosaic_shape[1] * 2 / aligner.mosaic_shape[0] > 2 * 4 / 3:
            nrows, ncols = ncols, nrows
    else:
        nrows, ncols = 1, 1
    
    fig = plt.figure(figsize = (100, 100))
    ax = plt.subplot(nrows, ncols, 1)
    draw_mosaic_image(ax, aligner, img, **im_kwargs)
    error = np.array([aligner._cache[tuple(sorted(e))][1]
                      for e in aligner.neighbors_graph.edges])
    
    # Manually center and scale data to 0-1, except infinity which is set to -1.
    # This lets us use the purple-green diverging color map to color the graph
    # edges and cause the "infinity" edges to disappear into the background
    # (which is itself purple).
    infs = error == np.inf
    error[infs] = -1
    if not infs.all():
        error_f = error[~infs]
        emin = np.min(error_f)
        emax = np.max(error_f)
        if emin == emax:
            # Always true when there's only one edge. Otherwise it's unlikely
            # but theoretically possible.
            erange = 1
        else:
            erange = emax - emin
        error[~infs] = (error_f - emin) / erange
    # Neighbor graph colored by edge alignment quality (brighter = better).
    nx.draw(
        aligner.neighbors_graph, ax=ax, with_labels=True,
        pos=np.fliplr(centers), edge_color=error, edge_vmin=-1, edge_vmax=1,
        edge_cmap=plt.get_cmap('PRGn'), **final_nx_kwargs
    )
    if show_tree:
        ax = plt.subplot(nrows, ncols, 2)
        draw_mosaic_image(ax, aligner, img, **im_kwargs)
        # Spanning tree with nodes at original tile positions.
        nx.draw(
            aligner.spanning_tree, ax=ax, with_labels=True,
            pos=np.fliplr(centers), edge_color='royalblue',
            **final_nx_kwargs
        )
    fig.set_facecolor('black')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "QC_edge_quality.pdf"))

def plot_edge_scatter(aligner, outdir, annotate=True):
    
    import seaborn as sns
    xdata = aligner.all_errors
    ydata = np.clip(
        [np.linalg.norm(v[0]) for v in aligner._cache.values()], 0.01, np.inf
    )

    #remove inf values if present
    if np.inf in ydata:
        ydata[ydata == np.inf] = np.max(ydata[ydata != np.inf]) * 2
    if np.inf in xdata:
        xdata[xdata == np.inf] = np.max(xdata[xdata != np.inf]) * 2

    pdata = np.clip(aligner.errors_negative_sampled, 0, 10) #by clipping no inf values can remain

    g = sns.JointGrid(x = xdata, y = ydata)
    g.plot_joint(sns.scatterplot, alpha=0.5)
    
    _, xbins = np.histogram(np.hstack([xdata, pdata]), bins=40)
    
    sns.distplot(
        xdata, ax=g.ax_marg_x, kde=False, bins = xbins, norm_hist=True
    )

    sns.distplot(
        pdata, ax=g.ax_marg_x, kde=False, bins=xbins, norm_hist=True,
        hist_kws=dict(histtype='step')
    )
    g.ax_joint.axvline(aligner.max_error, c='k', ls=':')
    g.ax_joint.axhline(aligner.max_shift_pixels, c='k', ls=':')
    g.ax_joint.set_yscale('log')
    g.set_axis_labels('error', 'shift')
    if annotate:
        for pair, x, y in zip(aligner.neighbors_graph.edges, xdata, ydata):
            plt.annotate(str(pair), (x, y), alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "QC_edge_scatter.pdf"))


from ashlar.reg import LayerAligner, EdgeAligner, warn_data, Mosaic
import ashlar.utils as utils

#helper functions for paralellization
def _execute_indexed_parallel(
    func: Callable, *, args: list, tqdm_args: dict = None, n_threads: int = 10
) -> list:
    if tqdm_args is None:
        tqdm_args = {}

    results = [None for _ in range(len(args))]
    with ThreadPoolExecutor(n_threads) as executor:
        with tqdm(total=len(args), **tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    return results


def _execute_parallel(func: Callable, *, args: list, tqdm_args: dict = None, n_threads: int = 10):
    if tqdm_args is None:
        tqdm_args = {}

    with ThreadPoolExecutor(n_threads) as executor:
        with tqdm(total=len(args), **tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for _ in as_completed(futures):
                pbar.update(1)
    
    return None

def parallel_eccentricity(G, batch_size=200, num_threads=20):
    order = G.order()
    
    def _calculate(nodes):
        results = {}
        for node in nodes:
            length = nx.shortest_path_length(G, source=node)
            L = len(length)
            
            if L != order:
                if G.is_directed():
                    msg = (
                        "Found infinite path length because the digraph is not"
                        " strongly connected"
                    )
                else:
                    msg = "Found infinite path length because the graph is not" " connected"
                
                sys.exit(msg)
            
            results[node] = max(length.values())
        
        return results

    # Split nodes into batches (this limits the number of threads that need to formed making it more efficient)
    node_batches = [list(G.nodes())[i:i + batch_size] for i in range(0, len(G.nodes()), batch_size)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks for each node batch to the thread pool
        futures = [executor.submit(_calculate, batch) for batch in node_batches]
        
        # Gather results as they complete
        results = {}
        for future in concurrent.futures.as_completed(futures):
            batch_results = future.result()
            results.update(batch_results)
    
    return results


import networkx as nx
from graph_tool import graph_tool as gt
from graph_tool import topology as topology
from graph_tool.draw import graph_draw
from graph_tool.search import dijkstra_search, dijkstra_iterator
import sklearn.linear_model

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
    """
    # Phase 0: Create a directed or undirected graph-tool Graph
    gtG = gt.Graph(directed=nxG.is_directed())

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

def get_center_nodes(g):
    
    nodes = g.get_vertices()
    if len(nodes) > 2:
        
        distances = topology.shortest_distance(g)
        eccentricities = [x.a[x.a < 1e100].max() for x in distances]
        
        centers = nodes[eccentricities.index(min(eccentricities))]

        return(centers)
        
    else:
        return(nodes[0]) 
    
class ParallelLayerAligner(LayerAligner):

    def __init__(self, reader, reference_aligner, n_threads = 20, channel=None, max_shift=15,
                 filter_sigma=0.0, verbose=False, *args, **kwargs):
        super().__init__(reader = reader, reference_aligner = reference_aligner, channel=channel, max_shift=max_shift,
                 filter_sigma=filter_sigma, verbose=verbose, *args, **kwargs)
        self.n_threads = n_threads

    def register_all(self):
        n = self.metadata.num_images
        args = [copy.deepcopy((i,)) for i in range(n)]
        results = _execute_indexed_parallel(
            self.register,
            args=args,
            tqdm_args=dict(
                file=sys.stdout,
                disable=not self.verbose,
                desc="                  aligning tile",
            ),
            n_threads = self.n_threads
        )
        shift, error = list(zip(*results))
        self.shifts = np.array(shift)
        self.errors = np.array(error)
        assert self.shifts.shape == (n, 2)
        assert self.errors.shape == (n,)

        if self.verbose:
            print()

class ParallelEdgeAligner(EdgeAligner):

    def __init__(
        self, reader, n_threads = 20, channel=0, max_shift=15, alpha=0.01, max_error=None,
        randomize=False, filter_sigma=0.0, do_make_thumbnail=True, verbose=False
    ):
        super().__init__(reader = reader, channel=channel, max_shift = max_shift, alpha = alpha, max_error = max_error,
        randomize=randomize, filter_sigma=filter_sigma, do_make_thumbnail=do_make_thumbnail, verbose=verbose)

        self.n_threads = n_threads

    def compute_threshold(self):
        # Compute error threshold for rejecting aligments. We generate a
        # distribution of error scores for many known non-overlapping image
        # regions and take a certain percentile as the maximum allowable error.
        # The percentile becomes our accepted false-positive ratio.
        edges = self.neighbors_graph.edges
        num_tiles = self.metadata.num_images
        # If not enough tiles overlap to matter, skip this whole thing.
        if len(edges) <= 1:
            self.errors_negative_sampled = np.empty(0)
            self.max_error = np.inf
            return
        widths = np.array([self.intersection(t1, t2).shape.min() for t1, t2 in edges])
        w = widths.max()
        max_offset = self.metadata.size[0] - w
        # Number of possible pairs minus number of actual neighbor pairs.
        num_distant_pairs = num_tiles * (num_tiles - 1) // 2 - len(edges)
        # Reduce permutation count for small datasets -- there are fewer
        # possible truly distinct strips with fewer tiles. The calculation here
        # is just a heuristic, not rigorously derived.
        n = 1000 if num_distant_pairs > 8 else (num_distant_pairs + 1) * 10
        pairs = np.empty((n, 2), dtype=int)
        offsets = np.empty((n, 2), dtype=int)
        # Generate n random non-overlapping image strips. Strips are always
        # horizontal, across the entire image width.
        max_tries = 100
        if self.randomize is False:
            random_state = np.random.RandomState(0)
        else:
            random_state = np.random.RandomState()
        for i in range(n):
            # Limit tries to avoid infinite loop in pathological cases.
            for current_try in range(max_tries):
                t1, t2 = random_state.randint(self.metadata.num_images, size=2)
                o1, o2 = random_state.randint(max_offset, size=2)
                # Check for non-overlapping strips and abort the retry loop.
                if t1 != t2 and (t1, t2) not in edges:
                    # Different, non-neighboring tiles -- always OK.
                    break
                elif t1 == t2 and abs(o1 - o2) > w:
                    # Same tile OK if strips don't overlap within the image.
                    break
                elif (t1, t2) in edges:
                    # Neighbors OK if either strip is entirely outside the
                    # expected overlap region (based on nominal positions).
                    its = self.intersection(t1, t2, np.repeat(w, 2))
                    ioff1, ioff2 = its.offsets[:, 0]
                    if (
                        its.shape[0] > its.shape[1]
                        or o1 < ioff1 - w
                        or o1 > ioff1 + w
                        or o2 < ioff2 - w
                        or o2 > ioff2 + w
                    ):
                        break
            else:
                # Retries exhausted. This should be very rare.
                warn_data("Could not find non-overlapping strips in {max_tries} tries")
            pairs[i] = t1, t2
            offsets[i] = o1, o2

        def register(t1, t2, offset1, offset2):
            img1 = self.reader.read(t1, self.channel)[offset1 : offset1 + w, :]
            img2 = self.reader.read(t2, self.channel)[offset2 : offset2 + w, :]
            _, error = utils.register(img1, img2, self.filter_sigma, upsample=1)
            return error

        # prepare arguments for executor
        args = []
        for (t1, t2), (offset1, offset2) in zip(pairs, offsets):
            arg = (t1, t2, offset1, offset2)
            args.append(copy.deepcopy(arg))

        errors = _execute_indexed_parallel(
            register,
            args=args,
            tqdm_args=dict(
                file=sys.stdout,
                disable=not self.verbose,
                desc="    quantifying alignment error",
            ),
            n_threads=self.n_threads
        )

        errors = np.array(errors)
        self.errors_negative_sampled = errors
        self.max_error = np.percentile(errors, self.alpha * 100)

    def register_all(self):
        args = []
        for t1, t2 in self.neighbors_graph.edges:
            arg = (t1, t2)
            args.append(copy.deepcopy(arg))

        _execute_parallel(
            self.register_pair,
            args=args,
            tqdm_args=dict(
                file=sys.stdout,
                disable=not self.verbose,
                desc="                  aligning edge",
            ),
            n_threads=self.n_threads
        )

        self.all_errors = np.array([x[1] for x in self._cache.values()])

        # Set error values above the threshold to infinity.
        for k, v in self._cache.items():
            if v[1] > self.max_error or np.any(np.abs(v[0]) > self.max_shift_pixels):
                self._cache[k] = (v[0], np.inf)

    def build_spanning_tree(self):
        g = nx.Graph()
        g.add_nodes_from(self.neighbors_graph)
        g.add_weighted_edges_from(
            (t1, t2, error)
            for (t1, t2), (_, error) in self._cache.items()
            if np.isfinite(error)
        )

        gtG = nx2gt(g)

        spanning_tree = gt.Graph(gtG)
        spanning_tree.clear_edges()

        # label the components in a property map
        c = gt.topology.label_components(gtG)[0]
        components = np.unique(c.a)

        centers = []
        for i in components:
            u = gt.GraphView(gtG, vfilt=c.a == i)
            #graph_draw(u, vertex_text=u.vertex_index, ink_scale = 0.5)
            
            center = get_center_nodes(u)
            centers.append(center)
            vertices = list(u.vertices())
            for vertix in vertices:
                vlist, elist = gt.topology.shortest_path(u, center, vertix)
                spanning_tree.add_edge_list(elist)

        gt.generation.remove_parallel_edges(spanning_tree)
        self.spanning_tree = spanning_tree
        self.centers_spanning_tree = centers

    def calculate_positions(self, batch_size = 200):
        shifts = {}
        _components = []

        # label the components in a property map
        c = gt.topology.label_components(self.spanning_tree)[0]
        components = np.unique(c.a)

        for ix, i in enumerate(components):
            u = gt.GraphView(self.spanning_tree, vfilt=c.a == i)
            nodes = list(u.get_vertices())
            _components.append(nodes)

            center = self.centers_spanning_tree[ix]
            
            shifts[center] = np.array([0, 0])
            
            if len(nodes) > 1:
                for edge in gt.search.bfs_iterator(u, source=center):
                    source, dest = edge
                    source = int(source)
                    dest = int(dest)
                    if source not in shifts:
                        source, dest = dest, source
                    shift = self.register_pair(source, dest)[0]
                    shifts[dest] = shifts[source] + shift

            if shifts:
                self.shifts = np.array([s for _, s in sorted(shifts.items())])
                self.positions = self.metadata.positions + self.shifts
                self.components_spanning_tree = _components
            else:
                # TODO: fill in shifts and positions with 0x2 arrays
                raise NotImplementedError("No images")    

    def fit_model(self):
        components = self.components_spanning_tree
        components = sorted(
            components,
            key=len, reverse=True
        )

        # Fit LR model on positions of largest connected component.
        cc0 = list(components[0])
        self.lr = sklearn.linear_model.LinearRegression()
        self.lr.fit(self.metadata.positions[cc0], self.positions[cc0])
        # Fix up degenerate transform matrix. This happens when the spanning
        # tree is completely edgeless or cc0's metadata positions fall in a
        # straight line. In this case we fall back to the identity transform.
        if np.linalg.det(self.lr.coef_) < 1e-3:
            # FIXME We should probably exit here, not just warn. We may provide
            # an option to force it anyway.
            warn_data(
                "Could not align enough edges, proceeding anyway with original"
                " stage positions."
            )
            self.lr.coef_ = np.diag(np.ones(2))
            self.lr.intercept_ = np.zeros(2)
        
        # Adjust position of remaining components so their centroids match
        # the predictions of the model.
        for cc in components[1:]:
            nodes = list(cc)
            centroid_m = np.mean(self.metadata.positions[nodes], axis=0)
            centroid_f = np.mean(self.positions[nodes], axis=0)
            shift = self.lr.predict([centroid_m])[0] - centroid_f
            self.positions[nodes] += shift
        # Adjust positions and model intercept to put origin at 0,0.
        self.origin = self.positions.min(axis=0)
        self.positions -= self.origin
        self.lr.intercept_ -= self.origin
        self.centers = self.positions + self.metadata.size / 2

class ParallelMosaic(Mosaic):

    def __init__(self, aligner, shape, n_threads=20, channels=None, ffp_path=None, dfp_path=None,
                 flip_mosaic_x=False, flip_mosaic_y=False, barrel_correction=None,
                 verbose=False):

        super().__init__(aligner=aligner, shape=shape, channels=channels, ffp_path=ffp_path, dfp_path=dfp_path,
                         flip_mosaic_x=flip_mosaic_x, flip_mosaic_y=flip_mosaic_y, barrel_correction=barrel_correction,
                         verbose=verbose)

        self.n_threads = n_threads

    def assemble_channel_parallel(
            self,
            channel,
            positions,
            reader,
            out=None,
            tqdm_args=None,
    ):
        if tqdm_args is None:
            tqdm_args = dict(
                file=sys.stdout,
                disable= not self.verbose,
                desc="assembling tiles",
            )

        if out is None:
            out = np.zeros(self.shape, self.dtype)
        else:
            if out.shape != self.shape:
                raise ValueError(
                    f"out array shape {out.shape} does not match Mosaic"
                    f" shape {self.shape}"
                )

        def assemble_single(si_position):
            si, position = si_position
            img = reader.read(c=channel, series=si)
            img = self.correct_illumination(img, channel)
            utils.paste(out, img, position, func=utils.pastefunc_blend)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            list(tqdm(executor.map(assemble_single, enumerate(positions)), total=len(positions), **tqdm_args))
        
        # Memory-conserving axis flips.
        if self.flip_mosaic_x:
            for i in range(len(out)):
                out[i] = out[i, ::-1]
        if self.flip_mosaic_y:
            for i in range(len(out) // 2):
                out[[i, -i - 1]] = out[[-i - 1, i]]
        return out
    