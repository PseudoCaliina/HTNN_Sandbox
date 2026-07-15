# -*- coding: utf-8 -*-
"""
- This python file provide users to access more easily 
  to simulate groundwater flow using 'VSAFT2 SOLVER'

-  The version now only support "2D" and "3D" "Steady" state 
   groundwater flow simulation in " Rectangular Boundary "

Author: PseudoCaliina
Date: 2026-07-01
"""
import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE, STDOUT

# ===== Functions =====
def element_cor_2d(start_coord, element_spacing):  
    """ 
    Generate coordinate for each element
    
    for 1-D example:
    
        │--10--│----20----│------30------│   (element spacing) 
        0      10        30              60  (node coordinate)
        ↓    
        0--5--10----20---30------45------60  (element coordinate)
    
    To generate 2d coordinate alike in grid_mateiral.dat
    start_coord: [x_0, y_0]
    element_spacing: [dx, dy] dx = [dx1, dx2, ..., dxn], n = x_num
    """
    dx, dy= element_spacing

    x_e = len(dx)
    y_e = len(dy)
    ele_num = x_e*y_e
    x_cor = np.insert(  (start_coord[0] + np.cumsum(dx)),0, start_coord[0]  )
    y_cor = np.insert(  (start_coord[1] + np.cumsum(dy)),0, start_coord[1]  )  

    element_cor_x = (x_cor[:len(x_cor)-1] + x_cor[1:len(x_cor)])/2 
    element_cor_y = (y_cor[:len(y_cor)-1] + y_cor[1:len(y_cor)])/2 


    col_ele_idx = np.arange(1, ele_num + 1).astype(int)
    col_x_gird = np.tile(element_cor_x, y_e )
    col_y_gird = np.repeat(element_cor_y, x_e)

    df_node = pd.DataFrame(np.column_stack( (col_ele_idx,col_x_gird, col_y_gird)))
    df_node.columns = ['column_idx','x', 'y']
    
    return df_node

def element_cor_3d(start_coord, element_spacing):  
    """ 
    Generate each 3D element center coordinate
    Args:
        start_coord: [x_0, y_0, z_0]
        element_spacing: [dx, dy, dz]
    Return:
        df_node: DataFrame contain x,y,z coordinate
    """
    dx, dy, dz = element_spacing

    x_e, y_e, z_e = len(dx), len(dy), len(dz)
    ele_num = x_e * y_e * z_e 

    x_cor = np.insert(start_coord[0] + np.cumsum(dx), 0, start_coord[0])
    y_cor = np.insert(start_coord[1] + np.cumsum(dy), 0, start_coord[1])
    z_cor = np.insert(start_coord[2] + np.cumsum(dz), 0, start_coord[2])

    element_cor_x = (x_cor[:-1] + x_cor[1:]) / 2
    element_cor_y = (y_cor[:-1] + y_cor[1:]) / 2
    element_cor_z = (z_cor[:-1] + z_cor[1:]) / 2

    col_ele_idx = np.arange(1, ele_num + 1).astype(int)
    col_x_grid = np.tile(np.tile(element_cor_x, y_e), z_e)  
    col_y_grid = np.tile(np.repeat(element_cor_y, x_e), z_e)  
    col_z_grid = np.repeat(element_cor_z, x_e * y_e) 
    
    df_node = pd.DataFrame(np.column_stack((col_ele_idx, col_x_grid, col_y_grid, col_z_grid)))
    df_node.columns = ['column_idx', 'x', 'y', 'z']
    return df_node

def node_2d_df(node_num, start_coord, element_spacing, init_h, init_flux):
    """ 
    To generate 2d df in function node
    node_num: total node number
    start_coord: [x_0, y_0]
    element_spacing: [dx, dy] dx = [dx1, dx2, ..., dxn], n = x_num
    init_h: inital head for simulation
    """
    if len(start_coord) != 2:
        raise ValueError("start_coord must contain 2 element for 2D")
    
    dx, dy = element_spacing
    
    x_e = len(dx)
    y_e = len(dy)
    
    x_cor = np.concatenate([[start_coord[0]], start_coord[0] + np.cumsum(dx)])
    y_cor =  np.concatenate([[start_coord[1]], start_coord[1] + np.cumsum(dy)])

    col_node_idx = np.arange(1, node_num + 1).astype(int)
    col_x_node = np.tile(x_cor, (y_e + 1))  # repeat array x_cor (y_e + 1) times
    col_y_node = np.repeat(y_cor, (x_e + 1))         # repeat each element in  y_cor for (x_e + 1) times

    col_init_h = np.ones(node_num)*init_h  # inital head 
    col_init_c = np.zeros(node_num)        # concentration  
    col_init_c_flux = np.zeros(node_num)*init_flux    # concentration flux
    
    df_node = pd.DataFrame(np.column_stack(
        (col_node_idx, col_x_node, col_y_node, col_init_h, col_init_c, col_init_c_flux)))

    return df_node 

def node_3d_df( node_num, start_coord, element_spacing, init_h, init_flux):
    """ 
    To generate 3d df in function node
    node_num: total node number
    start_coord: [x_0, y_0, z_0]
    element_spacing: [dx, dy, dx] dx = [dx1, dx2, ..., dxn], n = x_num
    init_h: inital head for simulation
    """

    dx, dy, dz = element_spacing
    x_e = len(dx)
    y_e = len(dy)
    z_e = len(dz)
    
    x_cor = np.concatenate([[start_coord[0]], start_coord[0] + np.cumsum(dx)])
    y_cor = np.concatenate([[start_coord[1]], start_coord[1] + np.cumsum(dy)])
    z_cor = np.concatenate([[start_coord[2]], start_coord[2] + np.cumsum(dz)])

    col_node_idx = np.arange(1, node_num + 1)
    col_x_node = np.tile(x_cor, (y_e + 1)*(z_e + 1))
    col_y_node = np.tile( np.repeat(y_cor, (x_e + 1)) , (z_e + 1))
    col_z_node = np.repeat(z_cor,  (x_e + 1)*(y_e + 1)) 

    col_init_h = np.ones(node_num)*init_h  # inital head 
    col_init_c = np.zeros(node_num)        # concentration  
    col_init_c_flux = np.zeros(node_num)*init_flux      # concentration flux   

    df_node = pd.DataFrame(np.column_stack(
        (col_node_idx, col_x_node, col_y_node, col_z_node, col_init_h, col_init_c, col_init_c_flux)))

    return df_node

def element_2d_df(x_e, y_e):
    """ 
    x_e: element number in x direction
    y_e: element number in y direction
    """
    num_rows = x_e*y_e
    col_ele_idx = np.arange(1, num_rows + 1)
    col_mal_idx = np.arange(1, num_rows + 1)
    col_ang_idx = np.zeros(num_rows, dtype = int)
    col_zone_idx = np.ones(num_rows, dtype = int) # only have zone one

    dx = np.repeat(np.arange(1, y_e+1) - 1, x_e) 
    ld_idx = col_ele_idx + dx  # 1
    rd_idx = ld_idx + 1        # 2
    lu_idx = ld_idx + (x_e + 1) # 4 (x_e +1): 上下各差一個X方向元素數量+1
    ru_idx = lu_idx + 1         # 3

    df = pd.DataFrame(np.column_stack((col_ele_idx, 
                                    ld_idx,
                                    rd_idx,
                                    ru_idx,
                                    lu_idx,
                                    col_mal_idx,
                                    col_ang_idx,
                                    col_zone_idx)))
    return df

def element_3d_df(x_e, y_e, z_e):
    """ 
    x_e: element number in x direction
    y_e: element number in y direction
    z_e: element number in z direction
    """
    num_rows = x_e*y_e*z_e
    col_ele_idx = np.arange(1, num_rows + 1)
    col_mal_idx = np.arange(1, num_rows + 1)
    col_ang_idx = np.zeros(num_rows, dtype = int)
    col_zone_idx = np.ones(num_rows, dtype = int) # only have zone one

    re_array_x =  np.arange(1, (y_e*z_e) +1) - 1
    x_inc = np.repeat(re_array_x, x_e)
    re_array_z = (np.arange(1, z_e +1)-1)*(x_e + 1) 
    z_inc = np.repeat(re_array_z,  x_e*y_e) # 


    ld_1_idx = col_ele_idx + x_inc + z_inc     # col1 : a
    rd_1_idx = ld_1_idx + 1                    # col2 : a+1
    lu_1_idx = ld_1_idx + (x_e + 1)*(y_e + 1)  # col4 : b [a+ (x_e+1)* (y_e+1)]
    ru_1_idx = lu_1_idx + 1                    # col3 : b+1

    ld_2_idx = ld_1_idx  + (x_e + 1)           # col5 : e [ a + (x_e+1)]
    rd_2_idx = ld_2_idx  + 1                   # col6 : e+1
    lu_2_idx =  lu_1_idx + (x_e + 1)           # col8 : f [b + (x_e+1)]
    ru_2_idx =  lu_2_idx + 1                   # col7 : f+1

    df = pd.DataFrame(np.column_stack((col_ele_idx, 
                                        ld_1_idx,
                                        rd_1_idx,
                                        ru_1_idx,
                                        lu_1_idx,
                                        ld_2_idx,
                                        rd_2_idx,
                                        ru_2_idx,
                                        lu_2_idx,
                                     col_mal_idx,
                                     col_ang_idx,
                                     col_zone_idx)))
    return df

def create_material_format_2d(K, Ss, n, ele_num, Kh_paras, Se_paras, Trans_paras):
    """ 
    Generate 2D material.dat format
    
    Arguments:
    -------------
    K: Saturated hydraulic conductivity
    Ss: Specific storage
    n: porosity
    ele_num: int (total element in this case) 
    
    *Kh_paras: default parameters
    *Se_paras: default parameters
    *Trans_paras: default parameters
    
    """
    
    Sat_paras = [K, K, n, Ss]  # Kx, Ky, n, Ss
    df_ele = pd.DataFrame(np.tile([ele_num, " "], (1, 1)))
    df_Sat = pd.DataFrame(np.tile(Sat_paras, (ele_num, 1)) )
    df_Kh  = pd.DataFrame(np.tile(Kh_paras, (ele_num, 1)) )
    df_Se  = pd.DataFrame(np.tile(Se_paras, (ele_num, 1)) )
    df_Tr  = pd.DataFrame(np.tile(Trans_paras, (ele_num, 1)) )
    df_floor = pd.DataFrame(np.tile([0, " "], (5, 1)))
    
    

    df = pd.concat([df_ele,
                    df_Sat,
                    df_Kh,
                    df_Se,
                    df_Tr,
                    df_floor,
                    ], ignore_index=True)
    
    return df
    
def create_material_format_3d(K, Ss, n, ele_num, Kh_paras, Se_paras, Trans_paras):
    """ 
    Generate 3D material.dat format
    
    Arguments:
    -------------
    K: Saturated hydraulic conductivity
    Ss: Specific storage
    n: porosity
    ele_num: int (total element in this case) 
    
    *Kh_paras: default parameters (add z direction)
    *Se_paras: default parameters
    *Trans_paras: default parameters
    
    """
    
    Sat_paras = [K, K, K, n, Ss]  # Kx, Ky, n, Ss
    df_ele = pd.DataFrame(np.tile([ele_num, " "], (1, 1)))
    df_Sat = pd.DataFrame(np.tile(Sat_paras, (ele_num, 1)) )
    df_Kh  = pd.DataFrame(np.tile(Kh_paras, (ele_num, 1)) )
    df_Se  = pd.DataFrame(np.tile(Se_paras, (ele_num, 1)) )
    df_Tr  = pd.DataFrame(np.tile(Trans_paras, (ele_num, 1)) )
    df_floor = pd.DataFrame(np.tile([0, " "], (5, 1)))
    
    

    df = pd.concat([df_ele,
                    df_Sat,
                    df_Kh,
                    df_Se,
                    df_Tr,
                    df_floor,
                    ], ignore_index=True)
    
    return df
   
def rf_generation():
    """ 
    Allow user to generate random field based on the geometry and geostatistics
    
    """
    
    pass

def plot_domain_2D():
    """ 
    Plot and check the geometry of the 2D domain
    
    """
    pass

def plot_domain_3D():
    """  
    Plot and check  geometry of the 3D domain
    
    """
    
    pass

def plot_wells():
    """ 
    Plot the wells (include Inj_w and Obs_w)
    
    
    """
    
    pass

def plot_simulation_head():
    """ 
    Plot the head simulation result (separate by transient and steady)
    
    """
    
    pass

# ===== Class =====
class sle_io:
    """
    --- Vsaft2 Solver Forward Simulation Input Files ---
    grid.dat     
    element.dat  
    node.dat     
    material.dat  
    problem.dat    
    function.dat   
    simulation.dat 
    sources.dat   
    time.dat               (nc if steady?)
    obwell.dat    
    boundary.dat  
    bc.dat       
    time_vary_h_bc.dat     (nc)
    time_vary_f_bc.dat     (nc)
    validation_control.dat (nc)
    
    --- sle_io basic functions---
    Create all .dat files in working folder
    Execute "SLE.exe"
    Visualize io data
    Save output as csv
    
    ! All .dat files and SLE.exe must be located in the same folder; 
    ! Otherwise, the simulation will not run.
                 
    """
    
    ###### Simulation control ######
    DIMENSION_MAP = {
        "2D": 2,
        "3D": 3,
    }

    PROBLEM_MAP = {
        "steady": 1,
        "transient": 2,
    }

    AQUIFER_MAP = {
        "confined": 0,
        "unconfined": 1,
    }
      
    ###### DEFAULT PARAMETERS ######
    KH_PARAS_2D = (0.001, 0.1, 0.001, 0.1, 2)
    #              ax     bx    ay     by   model

    KH_PARAS_3D = (0.001, 0.1, 0.001, 0.1, 0.001, 0.1, 2)
    #              ax     bx    ay     by    az     bz   model

    # Soil-water retention parameters
    SE_PARAS =    (1, 0.1, 0.4, 0.01)
    #             a_m  b_m theta_s theta_r

    # Solute transport parameters
    TRANS_PARAS = (20,  20,  2,  0.0014)
    #              D_l  D_m  D_t   r_b
    
    # simulation
    SIMULATION_DEFAULT = {
        "iteration_scheme": 2,          # 2: Newton-Raphson iteration scheme
        "matrix_type": 2,               # 2: Diagonal lumped matrix
        "interpolation": 0,             # 0: Element interpolation method
        "max_iteration": 100,           # Maximum nonlinear iterations
        "flow_weight": 0.5,             # Time weighting factor for flow
        "transport_weight": 0.5,        # Time weighting factor for transport
        "head_tolerance": 1e-3,         # Pressure head convergence tolerance
        "flow_tolerance": 1e-9,         # Flow convergence tolerance
        "velocity_tolerance": 1e-9,     # Velocity convergence tolerance
        "transport_tolerance": 1e-9,    # Transport convergence tolerance
    }
    
    # function
    FUNCTION_DEFAULT = {
        "parallel": 0,
        "solver_mode": [1, 2],
        "regularization": [1, 0, 0, 0, 0, 0, 0, 0],
        "update_method": 2,
    }

    # problem
    PROBLEM_DEFAULT = {
        "simulation_type": 1,      # 1: Forward
        "output_flag": 1,          # Generate output.dat
        "coordinate_system": 1,    # Vertical axisymmetric
        "head_type": 1,            # Total head
    }

    ###### CHOICE FOR BOUNDARY ######
    SURFACE_MAP = {
        "LEFT":    ("x", "min"),
        "RIGHT":   ("x", "max"),
        "DOWN":    ("y", "min"),
        "UP":      ("y", "max"),
        "BOTTOM":  ("z", "min"),
        "TOP":     ("z", "max"),
    }
    
    ###### BC TYPE #####
    BC_MAP = {
        'flux': 0,
        'head': 1
    }
    
    ###### WELL TYPE ######
    WELL_TYPE = { 
                 "observation", 
                  "injection",
                  "pumping"
                }
    
    ###### OUTPUT FLAG ######
    OUTPUT_MAP = {
        'integer_time':  1 ,
        'each_time_step': 0,
    }
        
    def __init__(self, project_name):
        self.project_name = project_name

        # empty dict for project
        self.events = {}
        print(f'Project: {self.project_name}')

    # Tools
    def set_parameters(self, simulation_control):
        """
        Set simulation parameters
        
        Parameters
        ----------
        simulation_control : tuple [str, str, str, int]
        (
            dimension,
            problem_type,
            aquifer_type,
            stress_number
        )
        dimension: "2D" or "3D"
        problem_type: "steady" or "transient"
        aquifer_type: "confined" or "unconfined"
        stress_number: total pumping/injection event 
        
        """ 
        if len(simulation_control) != 4:
            raise ValueError("Error: Tuple simulation control must contain 4 element")        

        # Unpack tuple
        dimension, problem, aquifer, stress_number= simulation_control
        
        # Dimension
        try:
            self.dimension = self.DIMENSION_MAP[dimension]
        except KeyError:
            raise ValueError(
                f"Invalid dimension: '{dimension}'. "
                "Only '2D' or '3D' are allowed."
            )

        # Problem type
        try:
            self.problem_type = self.PROBLEM_MAP[problem]
        except KeyError:
            raise ValueError(
                f"Invalid problem type: '{problem}'. "
                "Only 'steady' or 'transient' are allowed."
            )

        # Aquifer type
        try:
            self.aquifer_type = self.AQUIFER_MAP[aquifer]
        except KeyError:
            raise ValueError(
                f"Invalid aquifer type: '{aquifer}'. "
                "Only 'confined' or 'unconfined' are allowed."
            )
        
        # Stress number
        self.stress_number = stress_number
        
        print(f"Simulation in '{dimension}' case, '{aquifer}' aquifer under '{problem}' state with {stress_number} events")
        
        return self
   
    def add_stress(self, stress_idx, time, obs, src):
        """ 
        Add sources, obwell, time by each stress
        
        Parameters (for each added stress)
        -----------------------------
        stress_idx: stress idex
        
        time: (dt, dt_max, dt_mul, t_max, t_red, t_ops)
        
        obs: (obs_1, obs_2, ...)
            obs_1 = ((x,y), "name", "well_type", "{}")
        src: (src_1, src_2, ...) 

        
        e.g. for adding stress contain 2 injection wells, 8 observation wells

            time = (10, 10, 1, 600, 'each_time_step', 0)
            obs =  (
                     ((188.5, 71,  130),  "Pie_1",  "observation", {}),
                     ((88.5,  131, 200),  "Pie_2",  "observation", {}),
                     ((88.5,  251, 180),  "Pie_3",  "observation", {}),
                     ((188.5, 311, 150),  "Pie_4",  "observation", {}),
                     ((388.5, 311, 150),  "Pie_5",  "observation", {}),
                     ((488.5, 251, 180),  "Pie_6",  "observation", {}),
                     ((488.5, 131, 200),  "Pie_7",  "observation", {}),
                     ((388.5, 71,  130),  "Pie_8",  "observation", {}),
            )
            src = (
                    ((288.5, 71,  120), 'inj_1', 'injection', {'rate':const_rate, 'time': inj_time})
                    ((88.5,  71,  110), 'inj_2', 'injection', {'rate':const_rate, 'time': inj_time})
            )

        """
        
        self.events[stress_idx] = {
            
            'time':      time,
            'observation': obs,
            'sources':     src,            
            
        }
        
        return self

    def select_node(self, by, value):
        """
        Select nodes by surface or coordinates.

        Parameters
        ----------
        by : {"surface", "coordinate"}
            Node selection method.

        value :
            If by == "surface":
                {"LEFT", "RIGHT", "UP", "DOWN", "TOP", "BOTTOM"}

            If by == "coordinate":
                Tuple of coordinates.

                2D:
                    ((x1, y1), (x2, y2), ...)

                3D:
                    ((x1, y1, z1), (x2, y2, z2), ...)

        Returns
        -------
        pandas.DataFrame
            Selected node information.
        """

        by = by.lower()

        # ---------------------------------------------------------
        # Select by surface
        # ---------------------------------------------------------
        if by == "surface":

            surface = value.upper()

            if surface not in self.SURFACE_MAP:
                raise ValueError(
                    f"Unknown surface '{surface}'. "
                    f"Available surfaces: {tuple(self.SURFACE_MAP.keys())}"
                )

            axis, side = self.SURFACE_MAP[surface]

            coord = getattr(self.df_node_coordinate[axis], side)()

            df = self.df_node_coordinate[
                self.df_node_coordinate[axis] == coord
            ].copy()

        # ---------------------------------------------------------
        # Select by coordinates
        # ---------------------------------------------------------
        elif by == "coordinate":

            coords = value

            if not isinstance(coords, tuple):
                raise ValueError(
                    "Coordinates must be provided as a tuple.\n"
                    "Example (2D): ((0, 0), (10, 0))\n"
                    "Example (3D): ((0, 0, 0), (10, 0, 5))"
                )
            

            selected = []

            for coord in coords:

                if len(coord) == 2:

                    x, y = coord

                    df = self.df_node_coordinate[
                        (self.df_node_coordinate["x"] == x) &
                        (self.df_node_coordinate["y"] == y)
                    ]

                elif len(coord) == 3:

                    x, y, z = coord

                    df = self.df_node_coordinate[
                        (self.df_node_coordinate["x"] == x) &
                        (self.df_node_coordinate["y"] == y) &
                        (self.df_node_coordinate["z"] == z)
                    ]

                else:

                    raise ValueError(
                        "Each coordinate must be either (x, y) or (x, y, z)."
                    )

                selected.append(df)

            df = pd.concat(selected, ignore_index=True)

        # ---------------------------------------------------------
        # Invalid method
        # ---------------------------------------------------------
        else:

            raise ValueError(
                "Selection method must be either 'surface' or 'coordinate'."
            )

        # ---------------------------------------------------------
        # Check result
        # ---------------------------------------------------------
        if df.empty:
            raise ValueError("No matching nodes were found.")

        return df

    def add_wells(self, *wells):
        """
        Add wells.

        Parameters
        ----------
        w1 :
            (
                stress,
                coordinate,
                name,
                type,
                parameter(dict)
            )
        wells = (w1, w2,...)    
        
        e.g.:
        inj_1 = (   1,          
                    ((288.5, 71,  120), )
                    'inj_1', 
                    'injection', 
                    {'rate':const_rate, 'time': inj_time}
                )
                
        mdoel.add_wells(inj_1, inj_2, ...)

        """
        
        for w in wells:

            stress, coordinate, name, well_type, parameter = w

            if well_type not in self.WELL_TYPE:
                raise ValueError(
                    f"Well type must be {self.WELL_TYPE[:]} for {name}"
                )                   
                     
            df_node = self.select_node(
                by="coordinate",
                value = coordinate
            )

            if df_node.empty:
                raise ValueError(
                    f"Cannot find node at {coordinate}"
                )

            node_id = int(df_node["node_id"].values)

            self.wells.append({
                "stress": stress,
                "node_id": node_id,
                "coordinate": coordinate,
                "name": name,
                "type": well_type.lower(),
                "parameter": parameter
            })

        return self
     
    # Dataframe creators         
    def create_geometry(self, start_coord, element_num, element_spacing):
        """
        Generate grid and element .dat files for groundwater simulation.

        This function constructs computational grid geometry based on
        starting coordinates, element configuration, and grid spacing.

        Parameters
        ----------
        start_coord : tuple [float, float, float]
            Starting coordinate of the grid (x_0, y_0, z_0).

        element_num : tuple [int]
            Grid resolution definition:
            - 2D case: [e_x, e_y]
            - 3D case: [e_x, e_y, e_z]

        element_spacing : tuple [float, ...]
            Grid spacing:
            - 2D: (dx, dy)
            - 3D: (dx, dy, dz)

            Notes
            -----
            dx, dy, dz can be either:
            - scalar (uniform spacing), or
            - np.ndarray (variable spacing per cell)

        Returns
        -------
        None
            Writes .dat files to work folder.
        """       
        # Check input length
        if len(start_coord) != self.dimension:
            raise ValueError(
                f"start_coord should contain {self.dimension} values to match the dimension."
            )

        if len(element_num) != self.dimension:
            raise ValueError(
                f"element_num should contain {self.dimension} values to match the dimension."
            )

        if len(element_spacing) != self.dimension:
            raise ValueError(
                f"element_spacing should contain {self.dimension} values to match the dimension."
            )

        # Check element number match element spacing number in each direction
        for num, spacing in zip(element_num, element_spacing):
            
            if num != len(spacing):
                raise ValueError(
                    "The length of each spacing array must match the "
                    "corresponding number of elements."
                )


        # define total element and node number
        
        total_element_num = math.prod(element_num)
        total_node_num = math.prod([i + 1 for i in element_num])
                
        if self.dimension == 2:
            x_e, y_e = element_num
            df_element = element_2d_df(x_e, y_e)            
            df_coordinate = element_cor_2d(start_coord, element_spacing)
            
        if self.dimension == 3:
            x_e, y_e, z_e = element_num
            df_element = element_3d_df(x_e, y_e, z_e)
            df_coordinate = element_cor_3d(start_coord, element_spacing)
            
        
        self.df_element = df_element
        self.df_coordinate = df_coordinate
        self.df_grid = pd.DataFrame((total_element_num, total_node_num))
        
        self.total_element_num = total_element_num
        self.total_node_num = total_node_num
        self.start_coord = start_coord
        self.element_num = element_num
        self.element_spacing = element_spacing
        
        return self
    
    def create_initial(self, init_paras):
        """
        Generate node, material file for groundwater simulation.

        Parameters
        ----------
        init_paras : tuple [init_h, init_K, init_Ss, porosity]
        
        Returns
        -------
        """       
        # check init_para contains enough data
        if len(init_paras) != 5:
            raise ValueError("Error: missing initial parameters, check init_para tuple")        

        
        init_h, init_flux, init_K, init_Ss, porosity = init_paras
        self.init_paras = init_paras 
        
        # node and material & df_node_coordinate (for selecting node_id)
        if self.dimension == 2:
                self.df_node = node_2d_df(self.total_node_num, self.start_coord, self.element_spacing, init_h, init_flux)    
                self.df_material = create_material_format_2d(
                        init_K, init_Ss, porosity, self.total_element_num,
                        self.KH_PARAS_2D, self.SE_PARAS, self.TRANS_PARAS
                    )
                
                self.df_node_coordinate = self.df_node.iloc[:, :self.dimension+1].copy()
                self.df_node_coordinate.columns = ['node_id', 'x', 'y']
                
        if self.dimension == 3:
                self.df_node = node_3d_df(self.total_node_num, self.start_coord, self.element_spacing, init_h, init_flux) 
                self.df_material = create_material_format_3d(
                        init_K, init_Ss, porosity, self.total_element_num,
                        self.KH_PARAS_3D, self.SE_PARAS, self.TRANS_PARAS
                    )
                
                self.df_node_coordinate = self.df_node.iloc[:, :self.dimension+1].copy()
                self.df_node_coordinate.columns = ['node_id', 'x', 'y', 'z']
                

        return self

    def create_boundary(self, boundary):
        """
        Generate boundary and bc file for groundwater simulation.

        Parameters
        ----------
        boundary : tuple
            (method, value)

            method:
                "surface" -> "LEFT", "RIGHT", ...

                "coordinate" -> ((x1,y1), (x2,y2), ...)

        Returns
        -------
        self
        """
            
        method, value, bc_type = boundary
        
        if bc_type not in self.BC_MAP:
            
            raise ValueError(f"boundary should be {self.BC_MAP}")
        
        df_node_id_select = self.select_node(method, value)
        df_bc_ele = pd.DataFrame([[len(df_node_id_select['node_id']), " "]])

        # for boundary
        df_boundary = pd.DataFrame({
            'node_id':  df_node_id_select['node_id'].values,
            'hydraulic condition': self.BC_MAP[bc_type],
            'sc': 0,
            'sp': 0,
            'ep': 0,  
            'bc_node_for_angle':0
        })
        
        self.df_boundary = pd.concat(
            [df_bc_ele, df_boundary], ignore_index = True
        )
        
        # for bc
        df_bc = pd.DataFrame({
            'node_id':  df_node_id_select['node_id'].values,
            'init_h': self.init_paras[0],
            'flux':   self.init_paras[1],
            'sc': 0,  
            'sflux': 0
        })
    
            
        self.df_bc = pd.concat(
            [df_bc_ele, df_bc], ignore_index = True
        )
                
        return self

    def create_observation(self):

        obs_list = []
        for stress_idx, event in sorted(self.events.items()):
            
            rows = []
            obs = event["observation"]
            df_between = pd.DataFrame([[stress_idx], [len(obs)]])
            
            for w in obs:
                coor, name = w
                
                
                df_node = self.select_node(
                    by="coordinate",
                    value = (coor,)
                )

                if df_node.empty:
                    raise ValueError(
                        f"Cannot find node at {coor}"
                    )

                node_id = int(df_node["node_id"].values)   
                             
                rows.append([
                    node_id,
                    name
                ])

            df_ = pd.DataFrame(rows)
            obs_list.append(pd.concat([df_between, df_]))    

        self.df_obwell = pd.concat(obs_list)
        
        return self

    def create_source(self):
        
        
        df_up = pd.DataFrame([[len(sorted(self.events))], [0]])
        src_list = [df_up]
        
        for _, event in sorted(self.events.items()): 
            
            rows = []
            srcs = event['sources']
            df_between = pd.DataFrame([len(srcs)])
            
            for src in srcs:
                
                coor, name, paras = src
                
                df_node = self.select_node(
                    by="coordinate",
                    value = (coor,)
                )

                if df_node.empty:
                    raise ValueError(
                        f"Cannot find node at {coor}"
                    )

                node_id = int(df_node["node_id"].values)
                    
                rows.append([
                    node_id,              # node index
                    paras["rate"],        # sink/source for flow
                    0,                    # sink/source for concentration
                    paras["time"][0],     # start time
                    paras["time"][1],     # end time
                    0,                    # start time for concentration
                    0,                    # end time for concentration
                    name,                 # well name
                ])

            df_ = pd.DataFrame(rows)
            src_list.append(pd.concat([df_between, df_]))    
        
        self.df_source = pd.concat(src_list)
            
    
        
        return self

    def create_times(self):
        
        time_list = []
        
        for _, event in sorted(self.events.items()):
            
            t = event['time']
            dt, dt_max, dt_mul, t_max, max_red, flag = t
            
            try:
                self.OUTPUT_MAP[flag]
                
            except KeyError:
                raise ValueError(
                    f"Invalid output flag: '{flag}'. "
                    "Only 'integer_time' or 'each_time_step' two output options."
                )
            
            
            df_ = pd.DataFrame([dt, dt_max, dt_mul, t_max, max_red])  
              
            time_list.append(df_)

        
        self.df_time = pd.concat(time_list, axis=1)
        
        return self

    def create_function(self, **kwargs):
        """
        Create function.dat.

        Any keyword argument overrides the default value.
        
        function.dat format:
        --------------------------
        2                dimension (2:2D, 3:3D)
        0                confined/unconfined (0,1)
        0                nctimes parallel/solver parallel  (0,1)
        1 2              0: Pert. 1: Adj.
        1 0 0 0 0 0 0 0  [rp,rq1,rq2,rq3,rq,rw,rc,rcq]
        2                1: Reduced order by SVD.  2: Full matrix.   3: No update
        --------------------------
        """

        d = self.FUNCTION_DEFAULT.copy()
        d.update(kwargs)

        self.df_function =  pd.DataFrame([
            [self.dimension],
            [self.aquifer_type],
            [d["parallel"]],
            d["solver_mode"],
            d["regularization"],
            [d["update_method"]],
        ])
        
        return self

    def create_simulation(self, **kwargs):
        """
        Create simulation.dat.

        Any keyword argument overrides the default value.
        """

        d = self.SIMULATION_DEFAULT.copy()
        d.update(kwargs)

        self.df_simulation =  pd.DataFrame([[
                    d["iteration_scheme"],
                    d["matrix_type"],
                    d["interpolation"],
                    d["max_iteration"],
                    d["flow_weight"],
                    d["transport_weight"],
                    d["head_tolerance"],
                    d["flow_tolerance"],
                    d["velocity_tolerance"],
                    d["transport_tolerance"],
        ]]).T
        
        return self

    def create_problem(self, **kwargs):
        """
        Create problem.dat.

        Any keyword argument overrides the default value.
        """

        d = self.PROBLEM_DEFAULT.copy()
        d.update(kwargs)

        self.df_problem =  pd.DataFrame([[
                d["simulation_type"],
                d["output_flag"],
                self.problem_type,
                d["coordinate_system"],
                d["head_type"],
        ]]).T

        return self
        
    def write_forward_input(self, path):
        """
        Write all required input files for forward simulation.

        Parameters
        ----------
        path : str or Path
            Output folder for SLE input files.
        """

        # All required DataFrames
        required_files = {
            "grid.dat":     self.df_grid,
            "node.dat":     self.df_node,
            "element.dat":  self.df_element,
            "bc.dat":       self.df_bc,
            "boundary.dat": self.df_boundary,
            "sources.dat":  self.df_source,
            "material.dat": self.df_material,
            "obwell.dat":   self.df_obwell,
            "function.dat": self.df_function,
            "time.dat":     self.df_time,
            "problem.dat":  self.df_problem,
            "simulation.dat": self.df_simulation,
        }

        # Check missing DataFrames
        missing = [
            name for name, df in required_files.items()
            if df is None
        ]

        if missing:
            raise ValueError(
                "The following input files have not been created:\n"
                + "\n".join(f"  - {name}" for name in missing)
            )

        # Create output folder if it does not exist
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = output_dir.resolve()
        # Write files
        for filename, df in required_files.items():

            header = filename == "problem.dat"

            df.to_csv(
                output_dir / filename,
                sep="\t",
                index=False,
                header=header
            )

        if (output_dir / "SLE.exe").exists():
            print("Ready for forward simulation!")
        else:
            print("SLE.exe not found. Please copy SLE.exe into this folder.")

        
    # Run simulation
    def run_forward(self, K = None, Ss = None, print_output = False):
        """
        Run forward simulation and allow to change initial parameters (K, Ss)
        Parameters
        --------------------
        K: hydraulic conductivity list in material.dat form
        Ss: Specific storage list in material.dat from
        """
        # the forward .dat files must exist before execute SLE.exe
        
        folder = self.output_dir
        forward_required_files = ["grid.dat", 
                                  "material.dat", 
                                  "node.dat",
                                  "element.dat",
                                  "boundary.dat",
                                  "bc.dat",
                                  "obwell.dat",
                                  "problem.dat",
                                  "function.dat",
                                  "simulation.dat",
                                  "time.dat",
                                  "sources.dat",
                                  "SLE.exe"]
        
        missing_files = [f for f in forward_required_files if not os.path.exists( self.output_dir)]
        if missing_files:
            raise FileNotFoundError(f"missing following files\n" +
                                    "\n".join(f"  - {file}" for file in missing_files))

        if K is None: 
            K = self.init_paras[0]
            
        if Ss is None:
            Ss = self.init_paras[1]
        
        if self.dimension == 2:
            self.df_material.columns = ['Kx','Ky','n','Ss','none1','none2']
            self.df_material.loc[1:self.total_element_num, 'Kx'] = K
            self.df_material.loc[1:self.total_element_num, 'Ky'] = K
            self.df_material.loc[1:self.total_element_num, 'Ss'] = Ss
        
        elif self.dimension == 3:
            self.df_material.columns = ['Kx','Ky','Kz','n','Ss','none1','none2']
            self.df_material.loc[1:self.total_element_num, 'Kx'] = K
            self.df_material.loc[1:self.total_element_num, 'Ky'] = K
            self.df_material.loc[1:self.total_element_num, 'Kz'] = K
            self.df_material.loc[1:self.total_element_num, 'Ss'] = Ss            
            
        self.df_material.to_csv(folder/'material.dat', sep="\t", header=False, index=False)
        
        # Execute SLE.exe
        exe = folder / "SLE.exe"
        p = Popen([exe], 
                  cwd = folder,
                  stdout=PIPE, 
                  stdin=PIPE, 
                  stderr=PIPE)
        
        stdout_data, _ = p.communicate(input='\n'.encode())       
        
        if print_output:
            print(stdout_data.decode("utf-8"))
            print('-'*40)
            print('Finish running SLE.exe')
            
    # Result
    def read_stress_result(self, path, ):
        
        return

    def plot_head_result():
        """ 
        Plot observation result from each stress
        
        
        """
        



