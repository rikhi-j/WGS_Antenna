import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon

from scipy.ndimage import label
from scipy.spatial.distance import cdist
import time
from collections import defaultdict


###############################################################################
# Visualization
###############################################################################
def flag_polygon(grid):
    flag_map = grid.copy()
    flag_map[flag_map==1]=-1
    M,N = grid.shape
    flag = 1

    def recursive_flagging(i,j):
        if flag_map[i,j] != -1:
            return
        flag_map[i,j] = flag
        #Go Down
        if i+1 <M:
            recursive_flagging(i+1,j)
        #Go Right
        if j-1 >=0:
            recursive_flagging(i,j-1)
        #Go Up
        if i-1 >=0:
            recursive_flagging(i-1,j)
        #Go Left
        if j+1 <N:
            recursive_flagging(i,j+1)
        return    

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if flag_map[i,j] == -1:
                recursive_flagging(i,j)
                flag +=1

    return flag_map

def display(grid, title=""):
    fig,ax =plt.subplots(figsize=(8,6))
    ax.imshow(grid, cmap='gray_r', vmin=0,vmax=1)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            text = ax.text(j,i,grid[i,j],ha="center",va="center")
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def heatmap(grid,title=""):
    fig,ax = plt.subplots(figsize=(8,6))
    ax.imshow(grid,cmap='hot')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            text = ax.text(j,i,grid[i,j],ha="center",va="center")
    plt.title(title)
    plt.axis('off')
    plt.show()        

###############################################################################
# Random Grid Generation
###############################################################################

def generate_grid(M, N, conductor_probability=0.5):

    grid= np.random.choice(
        [0,1],
        size=(M,N),
        p=[1-conductor_probability, conductor_probability]
    )

    return grid.astype(int)


###############################################################################
# Checkerboard Detection
###############################################################################

def illegal_blocks(grid):

    blocks = []

    M,N = grid.shape

    for i in range(M-1):
        for j in range(N-1):

            block = grid[i:i+2,j:j+2]

            if np.array_equal(block,
                              np.array([[1,0],
                                        [0,1]])):
                blocks.append((i,j))

            elif np.array_equal(block,
                                np.array([[0,1],
                                          [1,0]])):
                blocks.append((i,j))

    return blocks


###############################################################################
# Checkerboard Repair
###############################################################################

def remove_checkerboards(grid):

    grid = grid.copy()

    changed = True

    while changed:

        changed = False

        blocks = illegal_blocks(grid)

        for i,j in blocks:

            block = grid[i:i+2,j:j+2]

            #
            # Flip lower-right pixel
            #
            if block[0,0] == block[1,1]:

                grid[i+1,j+1] = block[0,1]

            else:

                grid[i+1,j+1] = block[0,0]

            changed = True

    return grid


###############################################################################
# Connected Component Utilities
###############################################################################

STRUCTURE = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
])


def connected_components(binary_grid):

    labels, ncomp = label(binary_grid, STRUCTURE)

    return labels, ncomp


###############################################################################
# Remove Small Conductor Islands
###############################################################################

def remove_small_conductor_islands(grid,
                                   min_area=4):

    labels, ncomp = connected_components(grid)

    new_grid = grid.copy()

    for k in range(1, ncomp+1):

        mask = labels == k

        if np.sum(mask) < min_area:

            new_grid[mask] = 0

    return new_grid


###############################################################################
# Remove Small Lakes
###############################################################################

def remove_small_lakes(grid,
                       min_area=4):

    water = 1-grid

    labels, ncomp = connected_components(water)

    new_grid = grid.copy()

    for k in range(1, ncomp+1):

        mask = labels == k

        if np.sum(mask) < min_area:

            new_grid[mask] = 1

    return new_grid


###############################################################################
# Bridge Drawing
###############################################################################

def draw_bridge(grid,
                start,
                end):

    r1,c1 = start
    r2,c2 = end

    #
    # horizontal/vertical Manhattan path
    #

    while r1 != r2:

        grid[r1,c1] = 1

        if r2 > r1:
            r1 += 1
        else:
            r1 -= 1

    while c1 != c2:

        grid[r1,c1] = 1

        if c2 > c1:
            c1 += 1
        else:
            c1 -= 1

    grid[r2,c2] = 1


###############################################################################
# Connect All Conductor Components
###############################################################################

def connect_components(grid):

    grid = grid.copy()

    while True:

        labels, ncomp = connected_components(grid)

        if ncomp <= 1:
            break

        #
        # find largest component
        #

        sizes = np.bincount(labels.ravel())
        sizes[0] = 0

        largest_label = np.argmax(sizes)

        main_pixels = np.argwhere(
            labels == largest_label
        )

        #
        # connect each smaller component
        #

        for k in range(1, ncomp+1):

            if k == largest_label:
                continue

            comp_pixels = np.argwhere(
                labels == k
            )

            #
            # closest pair
            #

            D = cdist(
                comp_pixels,
                main_pixels
            )

            idx = np.unravel_index(
                np.argmin(D),
                D.shape
            )

            p1 = tuple(comp_pixels[idx[0]])
            p2 = tuple(main_pixels[idx[1]])

            draw_bridge(grid,
                        p1,
                        p2)

            break

    return grid


###############################################################################
# Statistics
###############################################################################

def count_components(grid):

    _, ncomp = connected_components(grid)

    return ncomp


def count_checkerboards(grid):

    return len(illegal_blocks(grid))


###############################################################################
# Main Repair Function
###############################################################################

def repair_mask(grid,
                min_island_area=4,
                min_lake_area=2,
                verbose=True):

    grid = grid.copy()

    iteration = 0

    while True:

        old_grid = grid.copy()

        #
        # Step 1
        #

        grid = remove_checkerboards(grid)

        #
        # Step 2
        #

        grid = remove_small_conductor_islands(
            grid,
            min_island_area
        )

        #
        # Step 3
        #

        grid = connect_components(grid)

        #
        # Step 4
        #

        grid = remove_small_lakes(
            grid,
            min_lake_area
        )

        iteration += 1

        if verbose:

            print(
                f"Iteration {iteration:2d} | "
                f"Components={count_components(grid)} | "
                f"Checkerboards={count_checkerboards(grid)}"
            )

        if np.array_equal(old_grid,
                          grid):
            break

    return grid

def extract_rectilinear_polygons(grid):
    M, N = grid.shape
    edges = []

    def add_edge(p1, p2):
        edges.append((p1, p2))

    for i in range(M):
        for j in range(N):
            if grid[i, j] == 1:
                x, y = j, i  # IMPORTANT: column=x, row=y

                # cell corners
                p00 = (x,     y)
                p10 = (x + 1, y)
                p01 = (x,     y + 1)
                p11 = (x + 1, y + 1)

                # check 4 neighbors → only draw exposed edges

                # top
                if i == 0 or grid[i-1, j] == 0:
                    add_edge(p00, p10)

                # bottom
                if i == M-1 or grid[i+1, j] == 0:
                    add_edge(p01, p11)

                # left
                if j == 0 or grid[i, j-1] == 0:
                    add_edge(p00, p01)

                # right
                if j == N-1 or grid[i, j+1] == 0:
                    add_edge(p10, p11)

    return edges

def edges_to_HFSS_polygons(edges):
    polys = edges_to_polygons(edges)
    shapely_polys = []

    for poly in polys:
        clean = [(float(x), float(y)) for x, y in poly]

        if len(clean)<3:
            continue

        try:
            shapely_polys.append(Polygon(clean))
        except Exception as e:
            print("Failed polygon:", clean)
            raise e
    return shapely_polys    
def edges_to_polygons(edges):
    adj = defaultdict(list)

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()
    polygons = []

    def walk(start):
        poly = []
        current = start
        prev = None

        while True:
            poly.append(current)

            neighbors = adj[current]

            next_node = None
            for n in neighbors:
                e = tuple(sorted((current, n)))
                if e not in visited_edges:
                    next_node = n
                    visited_edges.add(e)
                    break

            if next_node is None or next_node == start:
                break

            prev = current
            current = next_node

        return poly

    for v in list(adj.keys()):
        for n in adj[v]:
            e = tuple(sorted((v, n)))
            if e not in visited_edges:
                visited_edges.add(e)
                poly = walk(v)
                if len(poly) > 2:
                    polygons.append(poly)

    return polygons

###############################################################################
# Demo
###############################################################################

if __name__ == "__main__":

    np.random.seed(int(time.time()))
    #np.random.seed(42)

    M = 20
    N = 30

    grid = generate_grid(
        M,
        N,
        conductor_probability=0.50
    )

    display(grid,
            "Original Random Grid")

    repaired = repair_mask(
        grid,
        min_island_area=5,
        min_lake_area=5
    )

    display(repaired,
            "Manufacturable Grid")

    holes = 1-repaired
    
    
    poly_map = flag_polygon(holes)

    heatmap(poly_map, "Polygon Map Padded")

    labels,ncomp = label(holes, STRUCTURE)
    edges = extract_rectilinear_polygons(holes)

    polys = edges_to_polygons(edges)

    verts = []
    for poly in polys:
        #poly.append(poly[0])  # close loop

        verts.append([(x, y, 0) for x, y in poly])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poly_1 = []
    poly_1.append(verts[0])
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.5))

    ax.set_xlim(0, holes.shape[1])
    ax.set_ylim(0, holes.shape[0])
    ax.set_zlim(0, 1)

    ax.set_box_aspect([1,1,0.2])
    plt.show()

    print()
    print("Final Statistics")
    print("----------------")
    print(
        "Connected Components:",
        count_components(repaired)
    )
    print(
        "Checkerboards:",
        count_checkerboards(repaired)
    )