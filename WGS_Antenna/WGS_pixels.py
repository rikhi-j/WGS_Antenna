# =============================================================================
# MODULE: WGS_pixels.py
# PURPOSE:
# Defines the WGS_Antenna class, which is the project's internal representation
# of an antenna. This class stores the antenna geometry as an unfolded 2D grid,
# while still preserving the relationship between each grid cell and its
# corresponding 3D surface on the physical WGS antenna.
#
# Rather than modifying HFSS geometry directly, the optimization process
# modifies WGS_Antenna objects. These objects are later repaired, validated,
# converted into ANSYS geometry, and simulated in HFSS.
# -----------------------------------------------------------------------------
# HOW IT IS USED:
# - optimization_controller.py creates a base WGS_Antenna object.
# - PSO design vectors are converted into WGS_Antenna conductor patterns.
# - Repair and validation modules operate directly on WGS_Antenna objects.
# - wgs_to_ansys_geometry.py converts the final repaired WGS_Antenna into
#   ANSYS geometry commands.
# -----------------------------------------------------------------------------
# OVERALL MODULE PROCESS:
# 1. Construct the unfolded 2D representation of the WGS antenna.
# 2. Map every unfolded grid cell to its corresponding 3D antenna surface.
# 3. Create the editable conductor grid used by the optimizer.
# 4. Create masks identifying which cells may or may not be modified.
# 5. Build a neighbor map describing how every cell connects across folded
#    antenna surfaces.
# 6. Provide helper functions for visualization and (legacy) mutation/crossover.
# -----------------------------------------------------------------------------
# HOW THE ANTENNA IS REPRESENTED:
# The physical WGS antenna is unfolded into a single 2D grid containing all six
# outer faces of the antenna. Each grid location represents one rectangular
# conductor cell on the antenna surface.
#
# Every editable cell can exist in one of two states:
#     True  = Metal (conductor present)
#     False = Empty (conductor removed)
#
# The optimizer never edits the physical HFSS geometry directly. Instead, it
# changes these Boolean conductor states, which are later converted into HFSS
# cutout operations.
# -----------------------------------------------------------------------------
# KEY DATA STORED:
# map_grid:
#     Stores which physical antenna face each unfolded grid cell belongs to
#     (Top, Bottom, Left, Right, Near Cap, or Far Cap).
#
# conductor_grid:
#     Stores the actual antenna design. Every Boolean value indicates whether a
#     conductor exists at that WGS cell.
#
# map_mask:
#     Identifies which conductor cells are editable by the optimizer. Cells
#     outside the editable region remain fixed throughout optimization.
#
# neighbor_map:
#     Stores neighboring cells while accounting for how the unfolded 2D layout
#     wraps around the 3D antenna. This allows repair and validation algorithms
#     to determine true physical neighbors.
#
# pad_ring_t:
#     Defines the thickness of the fixed border surrounding the editable WGS
#     antenna region. This border is created during initialization and is never
#     modified during optimization.
# -----------------------------------------------------------------------------
# MAIN INPUTS:
# - Physical antenna dimensions (size).
# - Grid resolution.
# - Padding-ring thickness.
# - Optional randomized initialization.
# - Alpha value controlling initial random conductor generation.
# -----------------------------------------------------------------------------
# MAIN OUTPUTS:
# - Fully initialized WGS_Antenna objects.
# - Editable conductor grids.
# - Neighbor connectivity information.
# - 3D visualization of the antenna (display functions).
# -----------------------------------------------------------------------------
# IMPORTANT NOTES:
# - This module defines the antenna representation used by the entire project.
# - The optimizer, repair system, validator, and ANSYS exporter all operate on
#   WGS_Antenna objects.
# - The conductor_grid contains the actual antenna design that is optimized.
# - map_mask prevents fixed conductor regions from being modified.
# - child() and mutate() are legacy helper functions from an earlier
#   genetic-algorithm workflow and are not used by the current PSO optimizer.
# =============================================================================

import pylab as plt
import numpy as np
import random
import copy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

class WGS_Antenna:
    # constructor
    def __init__(self, size=np.array([0,0,0]), pad_ring_t = np.array([0,0]), resolution=np.array([1,1,1]), randomized=False, alpha = 0.5):

        # physical dimensions of antenna body
        self.a = size[0]; self.b =  size[1]; self.c =  size[2]

        # stores grid resolution -> num of grid cells in each direction
        self.A = int(resolution[0]); self.B = int(resolution[1]); self.C = int(resolution[2])
        self.pad_ring_t = pad_ring_t
        self.alpha =  alpha

        # makes sure size and res are valid
        assert (self.a >0 and self.b>0 and self.c>0), f"Sizes must be greater than 0. {size}"
        assert (self.A >0 and self.B>0 and self.C>0), f"Resolutions must be greater than 0. {resolution}"

        # unfolded 2D layout of box
        # map_grid[i][j] says what part of the 3D antenna that 2D grid cell represents.
        self.map_grid = []

        # stores which cells are allowed to change during mutation
        self.map_mask = []        

        for i in range(2*self.B+self.C):
            temp_row = []
            for j in range(2*(self.A+self.B)):
                side = self.__get_side(i,j)
                if side != "X" or side !="":
                    temp_row.append(side)
            self.map_grid.append(temp_row)
        grid_i_min = 0
        grid_i_max = len(self.map_grid)
        i_min = grid_i_min + self.B+self.C*self.pad_ring_t[0]/self.c
        i_max = grid_i_max - (self.B+self.C*self.pad_ring_t[1]/self.c)

        # defines where antenna should force conductor material instead of allowing random choices
        self.pad_ring_t_indexes = np.array([int(i_min),int(i_max)])

        # actual grid
        self.conductor_grid = []
        self.randomizer_mask = []

        if randomized:
            temp_alpha = self.alpha
        else:
            temp_alpha = 0
        for i in range(len(self.map_grid)):
            temp_row = []
            temp_mask_row = []
            for j in range(len(self.map_grid[i])):
                side = self.__get_side(i,j)
                if side == "X":
                    continue
                if side == "FC": #or side == "LS" or side == "RS":
                    temp_row.append(True)
                    temp_mask_row.append(False)
                elif side == "NC":
                    temp_row.append(False)
                    temp_mask_row.append(False)
                elif i<=self.pad_ring_t_indexes[0] or i>=self.pad_ring_t_indexes[1]:
                    temp_row.append(True)
                    temp_mask_row.append(False)
                elif self.pad_ring_t_indexes[0]<i<self.pad_ring_t_indexes[1]:
                    temp_mask_row.append(True)    
                    if random.random() > temp_alpha:
                        temp_row.append(True)
                    else:
                        temp_row.append(False)
                else:
                    temp_row.append(False)        
            self.conductor_grid.append(temp_row)
            self.map_mask.append(temp_mask_row)

        # neighbor map
        self.neighbor_map = []
        for i in range(len(self.map_grid)):
            temp_row = []
            for j in range(len(self.map_grid[i])):
                temp_row.append(self.__get_neighbors(i,j))
            self.neighbor_map.append(temp_row)

    def display_shapley(self, top_face = True, bottom_face = True, left_side = True, right_side = True, near_cap = True, far_cap = True, mask = False):
        fig=plt.figure(layout='constrained')
        ax = plt.axes(projection='3d')
        plt.axis('equal')
        ax.set_box_aspect((self.c, self.a, self.b))
        
        top_face_colors = [(1,0,0),(1,0.5,0.5),(0.5,0,0),(0.5,0.25,0.25)]
        bottom_face_colors = [(0,1,0),(0.5,1,0.5),(0,0.5,0),(0.25,0.5,0.25)]
        left_side_colors = [(0,0,1),(0.5,0.5,1),(0,0,0.5),(0.25,0.25,0.5)]
        right_side_colors = [(1,1,0),(1,1,0.5),(0.5,0.5,0),(0.5,0.5,0.25)]
        near_cap_colors = [(0,1,1),(0.5,1,1),(0,0.5,0.5),(0.25,0.5,0.5)]
        far_cap_colors = [(1,0,1),(1,0.5,1),(0.5,0,0.5),(0.5,0.25,0.5)]
        TF_COLORS_N = len(top_face_colors)
        BF_COLORS_N = len(bottom_face_colors)
        LS_COLORS_N = len(left_side_colors)
        RS_COLORS_N = len(right_side_colors)
        NC_COLORS_N = len(near_cap_colors)
        FC_COLORS_N = len(far_cap_colors)
        verts = []
        colors = []
        poly_TF = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
        poly_BF = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
        poly_LS = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
        poly_RS = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
        poly_NC = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
        poly_FC = np.array([0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
        if mask:
            self.display(top_face,bottom_face,left_side,right_side,near_cap,far_cap,mask)
        else:
            for i in range(len(self.map_mask)):
                for j in range(len(self.map_mask[i])):
                    pass
            pass    

    def display(self, top_face = True, bottom_face = True, left_side = True, right_side = True, near_cap = True, far_cap = True, mask = False):
        fig=plt.figure(layout='constrained')
        ax = plt.axes(projection='3d')
        plt.axis('equal')
        ax.set_box_aspect((self.c, self.a, self.b))
        
        top_face_colors = [(1,0,0),(1,0.5,0.5),(0.5,0,0),(0.5,0.25,0.25)]
        bottom_face_colors = [(0,1,0),(0.5,1,0.5),(0,0.5,0),(0.25,0.5,0.25)]
        left_side_colors = [(0,0,1),(0.5,0.5,1),(0,0,0.5),(0.25,0.25,0.5)]
        right_side_colors = [(1,1,0),(1,1,0.5),(0.5,0.5,0),(0.5,0.5,0.25)]
        near_cap_colors = [(0,1,1),(0.5,1,1),(0,0.5,0.5),(0.25,0.5,0.5)]
        far_cap_colors = [(1,0,1),(1,0.5,1),(0.5,0,0.5),(0.5,0.25,0.5)]
        TF_COLORS_N = len(top_face_colors)
        BF_COLORS_N = len(bottom_face_colors)
        LS_COLORS_N = len(left_side_colors)
        RS_COLORS_N = len(right_side_colors)
        NC_COLORS_N = len(near_cap_colors)
        FC_COLORS_N = len(far_cap_colors)
        verts = []
        colors = []
        if mask:
            for i in range(len(self.map_mask)):
                for j in range(len(self.map_mask[i])):

                    if self.map_mask[i][j]:
                        side = self.__get_side(i,j)
                        match side:
                            case "NC":
                                if near_cap: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(near_cap_colors[np.mod(i+j,NC_COLORS_N)])  
                                
                            case "FC":
                                if far_cap: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(far_cap_colors[np.mod(i+j,FC_COLORS_N)]) 
                            case "TF":
                                if top_face: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(top_face_colors[np.mod(i+j,TF_COLORS_N)]) 
                            case "BF":
                                if bottom_face: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(bottom_face_colors[np.mod(i+j,BF_COLORS_N)])    
                            case "LS":
                                if left_side: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(left_side_colors[np.mod(i+j,LS_COLORS_N)])  
                            case "RS":
                                if right_side: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(right_side_colors[np.mod(i+j,RS_COLORS_N)])
                            case _:
                                pass
                    else:
                        verts.append(self.__make_segment(i,j))
                        colors.append((0,0,0))    
        else:
            for i in range(len(self.conductor_grid)):
                for j in range(len(self.conductor_grid[i])):

                    if self.conductor_grid[i][j]:
                        side = self.__get_side(i,j)
                        match side:
                            case "NC":
                                if near_cap: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(near_cap_colors[np.mod(i+j,NC_COLORS_N)])  
                                
                            case "FC":
                                if far_cap: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(far_cap_colors[np.mod(i+j,FC_COLORS_N)]) 
                            case "TF":
                                if top_face: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(top_face_colors[np.mod(i+j,TF_COLORS_N)]) 
                            case "BF":
                                if bottom_face: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(bottom_face_colors[np.mod(i+j,BF_COLORS_N)])    
                            case "LS":
                                if left_side: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(left_side_colors[np.mod(i+j,LS_COLORS_N)])  
                            case "RS":
                                if right_side: 
                                    verts.append(self.__make_segment(i,j))
                                    colors.append(right_side_colors[np.mod(i+j,RS_COLORS_N)])
                            case _:
                                pass    
        
                                     
        for i in range(len(verts)):
            ax.add_collection3d(Poly3DCollection([verts[i]],facecolors=[colors[i]]))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        plt.show()            

    def __convert_xyz_ij(self, x,y,z):
        a = self.a; b = self.b; c =self.c
        A = self.A; B = self.B; C = self.C
        da = a/A; db = b/B; dc = c/C
        
        position = {"NC": False, "FC": False,"LS": False,"RS": False, "TF": False, "BF": False,}
        if x == -c/2 and -a/2<= y <=a/2 and -b/2 <=z<=b/2:
            position["FC"] = True 
        elif x == self.c/2 and -a/2<=y<=a/2 and -b/2 <=z<=b/2:
            position["NC"] = True
        elif -c/2 <= x <= c/2 and y==-a/2 and -b/2 <=z<=b/2:
            position["LS"] =True
        elif -c/2 <=x <=c/2 and y==a/2 and -b/2 <=z<=b/2:
            position["RS"] = True
        elif -c/2 <=x <=c/2 and -a/2<=y<=a/2  and z == -b/2:
            position["BF"] = True
        elif -c/2 <=x <=c/2 and -a/2<=y<=a/2  and z == b/2:
            position["TF"] = True

        indexes = []
        if position["NC"]:
            i = int((z+b/2)/db); j = int((-y+a/2)/da)
            indexes.append((i,j))
            check = ("NC"==self.__get_side(i,j))
        if position["FC"]:
            i = int((z-b/2)/db)+B+C; j = int((-y+a/2)/da)
            indexes.append((i,j))
            check = ("FC"==self.__get_side(i,j))
        if position["TF"]:
            i = int((z+b/2)/db)+B; j = int((-y+a/2)/da) 
            indexes.append((i,j))
            check = ("TF"==self.__get_side(i,j))
        if position["BF"]:
            i = int((-x+c/2)/dc)+B; j = int((y+a/2)/da)+A+B
            indexes.append((i,j))
            check=("BF"==self.__get_side(i,j))
        if position["LS"]:
            i = int((-x+c/2)/dc)+B; j = int((-z-b/2)/db)+A
            indexes.append((i,j))
            check=("LS"==self.__get_side(i,j))
        if position["RS"]:
            i = int((-x+c/2)/dc)+B; j = int((z+b/2)/db)+(2*A+B)
            indexes.append((i,j))
            check=("RS"==self.__get_side(i,j))                          

        assert check, "Error in calculating x,y,z to i,j for array mask. check == False"
        assert len(indexes) <4, "Error in calculating x,y,z to i,j for array mask. Point cannot be on 4 faces (max 3)"
        
          


    def __make_segment(self, i, j):
        a = self.a; b = self.b; c = self.c
        A = self.A; B = self.B; C = self.C

        da = a/A; db = b/B; dc = c/C

        rtn = None

        position = self.__get_side(i,j)
        
        match position:
            case "NC":
                x0 = c/2
                rtn = np.array([[x0, a/2 - da*j, -b/2 + db*i],
                            [x0, a/2 -da*(j+1),-b/2 + db*i],
                            [x0, a/2 -da*(j+1),-b/2 +db*(i+1)],
                            [x0, a/2 -da*j,-b/2 +db*(i+1)]])
            
            case "FC":
                x0 = -c/2
                i -= (B+C)
                rtn = np.array([[x0, a/2 - da*j, b/2 - db*i],
                            [x0, a/2 -da*(j+1),b/2 - db*i],
                            [x0, a/2 -da*(j+1),b/2 - db*(i+1)],
                            [x0, a/2 -da*j,b/2 - db*(i+1)]])
            
            case "TF":
                z0 = b/2
                i -=B
                rtn = np.array([[c/2-dc*i, a/2 - da*j, z0],
                            [c/2-dc*(i+1), a/2 - da*j, z0],
                            [c/2-dc*(i+1), a/2 - da*(j+1), z0],
                            [c/2-dc*i, a/2 - da*(j+1), z0]])
            
            case "BF":
                z0 = -b/2
                i -=B
                j -= A+B
                rtn = np.array([[c/2-dc*i, -a/2 + da*j, z0],
                            [c/2-dc*(i+1), -a/2 + da*j, z0],
                            [c/2-dc*(i+1), -a/2 + da*(j+1), z0],
                            [c/2-dc*i, -a/2 + da*(j+1), z0]])

            case "LS":
                y0 = -a/2
                i -= B
                j -= A
                rtn = np.array([[c/2-dc*i, y0, b/2 - db*j],
                            [c/2-dc*(i+1), y0, b/2 - db*j],
                            [c/2-dc*(i+1), y0, b/2 - db*(j+1)],
                            [c/2-dc*i, y0, b/2 - db*(j+1)]])

            case "RS":
                y0 = a/2
                i -= B
                j -= (2*A+B)
                rtn = np.array([[c/2-dc*i, y0, -b/2 + db*j],
                            [c/2-dc*(i+1), y0, -b/2 + db*j],
                            [c/2-dc*(i+1), y0, -b/2 + db*(j+1)],
                            [c/2-dc*i, y0, -b/2 + db*(j+1)]]) 
            case _:
                rtn = None

        return rtn

                    

    def __get_side(self, i,j):
        rtn = ""
        A = self.A; B = self.B; C = self.C
        if 0<=j<=A-1:
            if 0<=i<=B-1:
                rtn = "NC"
            elif B<=i<=B+C-1:
                rtn = "TF"
            elif B+C<=i<=2*B+C-1:
                rtn = "FC"
            else:
                rtn = "X"
        elif A<=j<=A+B-1:
            if B<=i<=B+C-1:
                rtn = "LS"
            else:
                rtn = "X"
        elif A+B<=j<=2*A+B-1:
            if B<=i<=B+C-1:
                rtn = "BF"
            else:
                rtn = "X"
        elif 2*A+B<=j<=2*A+2*B-1:
            if B<=i<=B+C-1:
                rtn = "RS"
            else:
                rtn = "X"                     
        else:
            rtn = "X"
        return rtn
    
    def __get_neighbors(self,i,j):
        A = self.A; B = self.B; C = self.C
        
        position = self.__get_side(i,j)
        rtn = {"UP": np.array([i+1,j]), 
            "DOWN": np.array([i-1,j]), 
            "LEFT": np.array([i,j-1]), 
            "RIGHT": np.array([i,j+1])}

        match position:
            case "NC":
                if i == 0:
                    rtn["DOWN"]=np.array([B,-j+2*A+B-1])
                if j == 0:
                    rtn["LEFT"] = np.array([B,i+2*A+B])
                if j == A-1:
                    rtn["RIGHT"] = np.array([B,-i+A+B-1])        
            case "TF":
                if j==0:
                    rtn["LEFT"] = np.array([i,2*A+2*B-1])
            case "FC":
                if i == 2*B+C-1:
                    rtn["UP"]=np.array([B+C-1,-j+2*A+B-1])
                if j==0:
                    rtn["LEFT"] = np.array([B+C-1,-i+2*A+3*B+C-1])
                if j== A-1:
                    rtn["RIGHT"] = np.array([B+C-1,i+A-B-C])                
            case "LS":
                if i == B:
                    rtn["DOWN"]=np.array([-j+A+B-1,A-1])
                if i==B+C-1:
                    rtn["UP"] = np.array([j-A+B+C,A-1])
            case "BF":
                if i==B:
                    rtn["DOWN"]=np.array([0,-j+2*A+B-1])
                if i == B+C-1:
                    rtn["UP"]=np.array([2*B+C-1,-j+2*A+B-1])   
            case "RS":
                if i == B:
                    rtn["DOWN"]=np.array([j-2*A-B,0])
                if i == B+C-1:
                    rtn["UP"]=np.array([-j+2*A-B+C+1,0])
                if j == 2*A+2*B-1:
                    rtn["RIGHT"]=np.array([i,0])
            case _:
                rtn["DOWN"]=None; rtn["UP"]=None; rtn["LEFT"]=None; rtn["RIGHT"]=None
        return rtn 

def child(parrent1 = 1, parrent2=1, alpha_range = np.array([0.0,1.0])):
    assert isinstance(parrent1,WGS_Antenna), f"when using _child(), parrent1 must be of type WGS_Antenna. type(parrent1) = {type(parrent1)}"
    assert isinstance(parrent1,WGS_Antenna), f"when using _child(), parrent1 must be of type WGS_Antenna. type(parrent2) = {type(parrent2)}"

    ok_bool = True
    if len(parrent1.map_mask) == len(parrent2.map_mask):
        for i in range(len(parrent1.map_mask)):
            if len(parrent1.map_mask[i])!=len(parrent2.map_mask[i]):
                ok_bool = False
    
    assert ok_bool, "parrent1 and parrent2 must be the same dimensions"

    child_particle1 = copy.copy(parrent1)
    child_particle2 = copy.copy(parrent2)
    for i in range(len(parrent1.map_mask)):
        for j in range(len(parrent1.map_mask[i])):
            if parrent1.map_mask[i][j]:
                alpha = float(random.randrange(int(alpha_range[0]*100),int(alpha_range[1]*100)))/100
                if random.random()>alpha:
                    child_particle1.conductor_grid[i][j] = parrent2.conductor_grid[i][j]
                    child_particle2.conductor_grid[i][j] = parrent1.conductor_grid[i][j]

    return child_particle1, child_particle2

def mutate(particle, alpha = 0.02):
    assert isinstance(particle, WGS_Antenna), f"When using mutate(), particle must be of type WGS_Antenna. type(particle) = {type(particle)}"

    for i in range(len(particle.map_mask)):
        for j in range(len(particle.map_mask[i])):
            if random.random() > alpha and particle.map_mask[i][j]:
                particle.conductor_grid[i][j] = not particle.conductor_grid[i][j]

    return particle
