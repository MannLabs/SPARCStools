"""_custom_ashalar_funcs

functions adapted from https://github.com/labsyspharm/ashlar to provide specific functionality required for this package.
"""

import sys
import numpy as np
import copy as copy

#added requirements for parallelization
from networkx import Graph as nxGraph

# Import your utils module here
import sklearn.linear_model

from graph_tool.topology import shortest_path, label_components
from graph_tool import Graph as gtGraph
from graph_tool import GraphView
from graph_tool.generation import remove_parallel_edges
from graph_tool.search import bfs_iterator

from ashlar.reg import LayerAligner, EdgeAligner, warn_data, Mosaic
from ashlar import utils as utils  

from sparcstools.base.parallelilzation import execute_indexed_parallel, execute_parallel
from sparcstools.base.graphs import nx2gt, get_center_nodes

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class ParallelLayerAligner(LayerAligner):

    def __init__(self, reader, reference_aligner, n_threads = 20, channel=None, max_shift=15,
                 filter_sigma=0.0, verbose=False, *args, **kwargs):
        super().__init__(reader = reader, reference_aligner = reference_aligner, channel=channel, max_shift=max_shift,
                 filter_sigma=filter_sigma, verbose=verbose, *args, **kwargs)
        self.n_threads = n_threads

    def register_all(self):
        n = self.metadata.num_images
        args = [copy.deepcopy((i,)) for i in range(n)]
        results = execute_indexed_parallel(
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

        errors = execute_indexed_parallel(
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

        execute_parallel(
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

        self.cached_errors = self._cache.copy() # save as a backup

    def build_spanning_tree(self):
        g = nxGraph()
        g.add_nodes_from(self.neighbors_graph)
        g.add_weighted_edges_from(
            (t1, t2, error)
            for (t1, t2), (_, error) in self.cached_errors.items()
            if np.isfinite(error)
        )

        gtG = nx2gt(g)

        spanning_tree = gtGraph(gtG)
        spanning_tree.clear_edges()

        # label the components in a property map
        c = label_components(gtG)[0]
        components = np.unique(c.a)

        centers = []
        for i in components:
            u = GraphView(gtG, vfilt=c.a == i)
            #graph_draw(u, vertex_text=u.vertex_index, ink_scale = 0.5)
            
            center = get_center_nodes(u)
            centers.append(center)
            vertices = list(u.vertices())
            for vertix in vertices:
                vlist, elist = shortest_path(u, center, vertix)
                spanning_tree.add_edge_list(elist)

        remove_parallel_edges(spanning_tree)
        self.spanning_tree = spanning_tree
        self.centers_spanning_tree = centers

    def calculate_positions(self, batch_size = 200):
        shifts = {}
        _components = []

        # label the components in a property map
        c = label_components(self.spanning_tree)[0]
        components = np.unique(c.a)

        for ix, i in enumerate(components):
            u = GraphView(self.spanning_tree, vfilt=c.a == i)
            nodes = list(u.get_vertices())
            _components.append(nodes)

            center = self.centers_spanning_tree[ix]
            
            shifts[center] = np.array([0, 0])
            
            if len(nodes) > 1:
                for edge in bfs_iterator(u, source=center):
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
        
        tqdm_args = dict(
                file=sys.stdout,
                disable= not self.verbose,
                desc="assembling tiles",
                total=len(positions),
            )
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            list(tqdm(executor.map(assemble_single, enumerate(positions)), **tqdm_args))
        
        # Memory-conserving axis flips.
        if self.flip_mosaic_x:
            for i in range(len(out)):
                out[i] = out[i, ::-1]
        if self.flip_mosaic_y:
            for i in range(len(out) // 2):
                out[[i, -i - 1]] = out[[-i - 1, i]]
        return out
    