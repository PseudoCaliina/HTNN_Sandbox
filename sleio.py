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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== Functions =====
def element_cor_2d(start_coord, element_spacing):  # 建立元素中心座標
    """ 
    To generate 2d df in grid_mateiral
    grid_num: total grid number, otain by grid.dat
    start_coord: [x_0, y_0]
    element_spacing: [dx, dy] dx = [dx1, dx2, ..., dxn], n = x_num
    init_h: inital head for simulation
    """
    dx = element_spacing[0]
    dy = element_spacing[1]

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
    產生 3D 元素中心座標
    Args:
        start_coord: [x_0, y_0, z_0] -> 起始座標
        element_spacing: [dx, dy, dz] -> 每個方向的元素間距
    Return:
        df_node: DataFrame，包含元素索引及 x, y, z 座標
    """
    dx, dy, dz = element_spacing

    # 計算各方向的元素數量
    x_e, y_e, z_e = len(dx), len(dy), len(dz)
    ele_num = x_e * y_e * z_e  # 總元素數量

    # 計算 X, Y, Z 節點座標
    x_cor = np.insert(start_coord[0] + np.cumsum(dx), 0, start_coord[0])
    y_cor = np.insert(start_coord[1] + np.cumsum(dy), 0, start_coord[1])
    z_cor = np.insert(start_coord[2] + np.cumsum(dz), 0, start_coord[2])

    # 計算 X, Y, Z 元素中心
    element_cor_x = (x_cor[:-1] + x_cor[1:]) / 2
    element_cor_y = (y_cor[:-1] + y_cor[1:]) / 2
    element_cor_z = (z_cor[:-1] + z_cor[1:]) / 2

    # 展開為 3D 座標
    col_ele_idx = np.arange(1, ele_num + 1).astype(int)
    col_x_grid = np.tile(np.tile(element_cor_x, y_e), z_e)  # X 軸重複填充
    col_y_grid = np.tile(np.repeat(element_cor_y, x_e), z_e)  # Y 軸重複填充
    col_z_grid = np.repeat(element_cor_z, x_e * y_e)  # Z 軸重複填充

    # 建立 DataFrame
    df_node = pd.DataFrame(np.column_stack((col_ele_idx, col_x_grid, col_y_grid, col_z_grid)))
    df_node.columns = ['column_idx', 'x', 'y', 'z']
    return df_node

def node_2d_df(grid_num, start_coord, element_spacing, init_h):
    """ 
    To generate 2d df in function node
    grid_num: total grid number, otain by grid.dat
    start_coord: [x_0, y_0]
    element_spacing: [dx, dy] dx = [dx1, dx2, ..., dxn], n = x_num
    init_h: inital head for simulation
    """
    if len(start_coord) != 2:
        raise ValueError("start_coord must contain 2 element for 2D")
    
    dx = element_spacing[0]
    dy = element_spacing[1]

    x_e = len(dx)
    y_e = len(dy)
    
    x_cor = np.concatenate([[start_coord[0]], start_coord[0] + np.cumsum(dx)])
    y_cor =  np.concatenate([[start_coord[1]], start_coord[1] + np.cumsum(dy)])

    col_grid_idx = np.arange(1, grid_num + 1).astype(int)
    col_x_grid = np.tile(x_cor, (y_e + 1))  # repeat array x_cor (y_e + 1) times
    col_y_grid = np.repeat(y_cor, (x_e + 1))         # repeat each element in  y_cor for (x_e + 1) times

    col_init_h = np.ones(grid_num)*init_h  # inital head 
    col_init_c = np.zeros(grid_num)        # concentration  
    col_init_c_flux = np.zeros(grid_num)    # concentration flux
    
    df_node = pd.DataFrame(np.column_stack(
        (col_grid_idx, col_x_grid, col_y_grid, col_init_h, col_init_c, col_init_c_flux)))

    return df_node 

def node_3d_df( grid_num, start_coord, element_spacing, init_h):
    """ 
    To generate 3d df in function node
    grid_num: total grid number, otain by grid.dat
    start_coord: [x_0, y_0, z_0]
    element_spacing: [dx, dy, dx] dx = [dx1, dx2, ..., dxn], n = x_num
    init_h: inital head for simulation
    """
    dx = element_spacing[0]
    dy = element_spacing[1]
    dz = element_spacing[2]

    x_e = len(dx)
    y_e = len(dy)
    z_e = len(dz)
    
    x_cor = np.concatenate([[start_coord[0]], start_coord[0] + np.cumsum(dx)])
    y_cor = np.concatenate([[start_coord[1]], start_coord[1] + np.cumsum(dy)])
    z_cor = np.concatenate([[start_coord[2]], start_coord[2] + np.cumsum(dz)])

    col_grid_idx = np.arange(1, grid_num + 1)
    col_x_grid = np.tile(x_cor, (y_e + 1)*(z_e + 1))
    col_y_grid = np.tile( np.repeat(y_cor, (x_e + 1)) , (z_e + 1))
    col_z_grid = np.repeat(z_cor,  (x_e + 1)*(y_e + 1)) 

    col_init_h = np.ones(grid_num)*init_h  # inital head 
    col_init_c = np.zeros(grid_num)        # concentration  
    col_init_c_flux = np.zeros(grid_num)      # concentration flux   

    df_node = pd.DataFrame(np.column_stack(
        (col_grid_idx, col_x_grid, col_y_grid, col_z_grid, col_init_h, col_init_c, col_init_c_flux)))

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

def read_FwdObs(file_path, simulation_state, init_h):
    header = ['loca', 'well', 'time', 'wlevel','flux_x','flux_y','w_content','solute_concentation','solute_flux', '?']
    df = pd.read_csv(file_path, header=None,sep=r'\s+', names = header, engine='python')
    df.drop([0], inplace=True)
    df_groupby = df.groupby('well')

    df_head = pd.DataFrame()
    well_list = sorted(list(df_groupby.groups), key=len)

    if simulation_state == "transient":

        for well_name in well_list:     
            df_well = df_groupby.get_group(well_name)['wlevel']
            df_well.name = well_name
            df_well.reset_index(inplace=True, drop=True)
            
            df_head[well_name] = df_well
        df_head = df_head.subtract(df_head.iloc[0], axis = 1 )

    elif simulation_state == "steady":

        for well_name in well_list:   
            df_well = df_groupby.get_group(well_name)['wlevel']
            df_well.name = well_name
            df_well.reset_index(inplace=True, drop=True)
            
            df_head[well_name] = df_well
        df_head = df_head.subtract(init_h, axis = 1 )            
                
    return df_head

# ===== Class =====
class sle_io:
    """
    --- Vsaft2 Solver Forward Simulation Input Files ---
    grid.dat
    node.dat
    element.dat
    
    problem.dat
    simulation.dat
    sources.dat
    time.dat               (nc if steady?)
    material.dat
    obwell.dat
    boundary.dat
    bc.dat
    time_vary_h_bc.dat     (nc)
    time_vary_f_bc.dat     (nc)
    validation_control.dat (nc)
    
    --- sle_io basic fucntions---
    Create all .dat files in working folder
    Execute "SLE.exe"
    Visualize io data
    Save output as csv
    
    ! All .dat files and SLE.exe must be located in the same folder; 
    ! Otherwise, the simulation will not run.
                 

    """
    
    # CLASS ATTRIBUTE
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
    
    def __init__(self, project_name):
        self.project_name = project_name
        print(f'Project: {self.project_name}')

    def Set_parameters(self, simulation_control):
        """
        Set simualtion parameters
        
        Parameters
        ----------
        simulation_control : tuple[str, str, str]
        (
            dimension,
            problem_type,
            aquifer_type,
        )
        dimension: "2D" or "3D"
        problem_type: "steady" or "transient"
        aquifer_type: "confined" or "unconfined"
        
        """ 
        if len(simulation_control) != 3:
            raise ValueError("Error: Tuple simulation control must contain 3 element")        

        # Unpack tuple
        dimension, problem, aquifer = simulation_control
        
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
        
        print(f"Simulation with '{dimension}' case, '{aquifer}' aquifer under '{problem}' state")
        
        return self

    def Add_geometry(self, start_coord, element_num, element_spacing):
        """
        Generate grid, node and element .dat files for groundwater simulation.

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
            Writes .dat files to disk.
        """       

        # 
        if self.dimension == 2 and len(element_num) != 2:
            raise ValueError(
                "2D simulation requires element_num = [x_num, y_num]."
            )

        if self.dimension == 3 and len(element_num) != 3:
            raise ValueError(
                "3D simulation requires element_num = [x_num, y_num, z_num]."
            )

        x_0, y_0, *z_0 = start_coord
        x_e, y_e, *z_e = element_num
        dx,  dy,  *dz  = element_spacing
        
        if self.dimension == 2:
            total_element_num = x_e*y_e
            total_node_num = (x_e + 1)*(y_e + 1)
            df_element = element_2d_df(x_e, y_e)            
            df_coordinate = element_cor_2d(start_coord, element_spacing)
            
        if self.dimension == 3:
            total_element_num = x_e*y_e*z_e
            total_node_num = (x_e + 1)*(y_e + 1)*(z_e + 1)        
            df_element = element_3d_df(x_e, y_e, z_e)
            df_coordinate = element_cor_3d(start_coord, element_spacing)
            
        
        self.df_element = df_element
        self.df_coordinate = df_coordinate
        self.df_grid = pd.DataFrame((total_element_num, total_node_num))
        
        
        return self
    
    def Add_element():
        """
        Args:

        Return:

        """    
        
        return
    
    def Add_node():
        """
        Args:

        Return:

        """    
        
        return    
    
    def Add_boundary():
        """
        Args:

        Return:

        """    
        
        return

    def Add_observation():
        """
        Args:

        Return:

        """    
        
        return

    def Add_source():
        """
        Args:

        Return:

        """    
        
        return

    def Add_function():
        """
        Args:

        Return:

        """    
        
        return

    def Add_simulation():
        """
        Args:

        Return:

        """    
        
        return

    def WriteToFile():
        """
        Args:

        Return:

        """    
        
        return
    
    def Run_Forward():
        pass


