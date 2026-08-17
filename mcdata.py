# -*- coding: utf-8 -*-
"""
- This python code allow user to perfrom 3D SSDS process
- Include variogram estimation, conditional point selection and random field generation

Author: PseudoCaliina
Date: 2026-07-24
"""
import os
import numpy as np
import pandas as pd
import gstools as gs
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from gstools.random import MasterRNG

# ===== Functions =====

def parse_estimation(file_name, dim):
    """
    Used for parsing SLE inversion result, prepared for random field generation step
    Args:
        file_name: file name(direct path) for O-kestimate.dat or O-sestimate.dat 
        dim: 2d or 3d
    Returns:
        df_K: SLE estimated K mean (mm/s)
        df_ln_var: SLE estimated variance (Var. ln(K))
    """
    
    file_name = Path(file_name)  
    base_dir = file_name.parent
    
    with open(file_name, 'r') as file:
        lines = file.readlines()

    df = pd.read_csv( base_dir /'grid.dat', header=None,sep='\s+', engine='python')    
    ele_num = df.iloc[0,0]
    node_num = df.iloc[1,0]

    idx_list = []
    col_name = []
    iteration = 1
    for idx, line in enumerate(lines):
        if 'zone' in line:
            col_name.append('iter_' + str(iteration))
            idx_list.append(idx)
            iteration += 1    

    column_data = []  
    column_mean = []
    column_ln_var = []

    for i in range(len(idx_list)):
        str_idx = idx_list[i] + 1

        # estimation mean
        if i == 0:
            var_total_length = (node_num*dim + ele_num*2)
            mean_total_length = (node_num*dim + ele_num*3) 
        else:
            var_total_length = ele_num*2
            mean_total_length = ele_num*3

        mean_section = lines[str_idx: str_idx + mean_total_length]
        var_section = lines[str_idx: str_idx + var_total_length]

        mean_data = mean_section[-ele_num:]
        ln_var_data = var_section[-ele_num:]

        mean_data = [float(item.strip()) for item in mean_data]
        ln_var_data = [float(item.strip()) for item in ln_var_data]
        column_mean.append(mean_data)
        column_ln_var.append(ln_var_data)

    # sandbox no need to change to log scale 
    column_mean = np.array(column_mean).T
    column_ln_var = np.array(column_ln_var).T 
    column_data = np.array(column_data).T 
    df_result_mean = pd.DataFrame(columns=col_name, data =column_mean)
    df_result_var = pd.DataFrame(columns=col_name, data =column_ln_var)

    df_gird = pd.read_csv( base_dir /'grid-material.dat', sep='\s+', names = ['idx', 'x', 'y', 'z']) # ['idx', 'x', 'y', 'z]
    df_gird = df_gird.drop(columns='idx')
    df_K = pd.concat([df_gird,df_result_mean],axis=1)
    df_ln_var = pd.concat([df_gird,df_result_var],axis=1)
    
    return df_K, df_ln_var    

def interpolate_3d(df, value_col, grid_size=50, method="linear"):
    """
    Interpolate scattered 3D data (x, y, z, value) onto a regular 3D grid.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['x', 'y', 'z', value_col]
    value_col : str
        Column to interpolate (e.g. 'value')
    grid_size : int or tuple
        Number of grid points in each dimension (int or (nx, ny, nz))
    method : str
        Interpolation method: 'linear', 'nearest', 'cubic'

    Returns
    -------
    X, Y, Z : 3D numpy arrays (grid coordinates)
    V       : 3D numpy array (interpolated values)
    """
    # Points and values
    points = df[['x', 'y', 'z']].values
    values = df[value_col].values

    # Grid definition
    if isinstance(grid_size, int):
        nx = ny = nz = grid_size
    else:
        nx, ny, nz = grid_size

    xi = np.linspace(df['x'].min(), df['x'].max(), nx)
    yi = np.linspace(df['y'].min(), df['y'].max(), ny)
    zi = np.linspace(df['z'].min(), df['z'].max(), nz)
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing="ij")

    # Interpolate
    from scipy.interpolate import griddata
    V = griddata(points, values, (X, Y, Z), method=method)

    return X, Y, Z, V

def variogram_test(coordinate, value, bins, direction, max_dis):
    """ 
    Compute and fit variogram models for given spatial data.
    Args:
        coordinate: Tuple of numpy arrays (x_cor, y_cor) or (x_cor, y_cor, z_cor)
        value: Data values at given coordinates
        bins: Number of bins for variogram estimation
        direction: 'X', 'Y', or 'Z' (for 3D only)
        max_dis: Maximum distance for model fitting
    """
    # switch column and row direction (X<-->Y)
    dim_flag = len(coordinate)
    if dim_flag == 2:
        coordinate = (coordinate[1], coordinate[0])  # (y, x)
        angle = 0
    elif dim_flag == 3:
        coordinate = (coordinate[1], coordinate[0], coordinate[2])  # (y, x, z)
        angle = [0,0,0]
    else:
        raise ValueError("Only supports 2D or 3D spatial data.")
    
    # direction index
    direction_map = {'X': 1, 'Y': 0, 'Z': 2}
    if direction not in direction_map:
        raise ValueError("Direction must be 'X', 'Y', or 'Z' (for 3D only).")
    
    dir_index = direction_map[direction]
    if dim_flag == 2 and direction == 'Z':
        raise ValueError("Z direction is not valid for 2D data.")
    
    # experimental vaiorgram 
    bin_center, dir_vario, counts = gs.vario_estimate(
        *((coordinate, value, bins)),
        sampling_size=500,
        sampling_seed=8964,
        direction=gs.rotated_main_axes(dim=dim_flag, angles = angle),
        bandwidth=30,
        return_counts=True,
    )  
    
    # theorical model
    models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Stable": gs.Stable,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
    }
    scores = {}  
    
    # plot fitting result
    plt.figure(figsize=(8, 4))
    plt.scatter(bin_center, dir_vario[dir_index], color="k", label="Data")
    ax = plt.gca()
    print(f'-------------- Variogram Fitting for {direction} Direction --------------')
    for model_name, model_class in models.items():
        fit_model = model_class(dim=dim_flag)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, dir_vario[dir_index], return_r2=True)
        fit_model.plot(x_max=max_dis, ax=ax)
        scores[model_name] = r2
        print(f'Model: {model_name};  Pseudo r2 = {r2:.3f}')
        print(fit_model)
    
    # ax.set_title(f'{direction} Direction')
    ax.set_title(f'{direction} Direction', fontsize=18)
    ax.set_xlabel("Lag Distance", fontsize=14)
    ax.set_ylabel("Semivariance", fontsize=14)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # legend with fontsize
    ax.legend(fontsize=12)
    plt.show()

def d2_weight(points, wells, sigma):
    """ 
    points: (node* 3)
    wells:  (well* 3)
    
    """
    # points = coordinate[['x', 'y', 'z']].to_numpy()       # (node, 3)
    # wells = np.array(
    #         [coordinate for coordinate, _, _ in injs],    # (well, 3)
    #         dtype=float
    # )
    

    dist = np.linalg.norm(
        points[:, None, :] - wells[None, :, :],  # Broadcasting (node, 1, 3) : (1, well, 3)
        axis=2
    )

    weight = 1/(dist**2).sum(axis = 1)*sigma   
    
    return weight

def sampler(df, crf_params):

    voxel_x = crf_params['correlation_scale']['x']*crf_params['c_scale_mul']
    voxel_y = crf_params['correlation_scale']['y']*crf_params['c_scale_mul']
    voxel_z = crf_params['correlation_scale']['z']*crf_params['c_scale_mul']

    df['x_section'] = (df['x'] // voxel_x).astype(int)
    df['y_section'] = (df['y'] // voxel_y).astype(int)
    df['z_section'] = (df['z'] // voxel_z).astype(int)

    # sample conditional point (each section 1 data point)
    # --- by voxel
    df_ = df.groupby(['x_section', 'y_section', 'z_section']).sample(n = crf_params['section_sample_num']).reset_index(drop=True)
    
    # --- by distance from inj_A (*fraction_rate)
    df_spatial_sample = df_.sample(frac = crf_params['fraction_rate'], weights = df_['weight']).reset_index(drop=True)
    
    # sampling conditional value (lnK)
    df_spatial_sample['con_value'] = df_spatial_sample.apply(
        
        # uniform sampler
        lambda r:np.random.uniform(r['ln_mean']*crf_params['fraction_rate'], r['ln_mean'], 1)[0], axis = 1
        
            )
    return df_spatial_sample[['x', 'y', 'z', 'con_value']].copy()

def crf_generator(coordinate, df_con, crf_params, seed, idx):
        
    # model
    model = gs.Gaussian(
        dim = len(crf_params['correlation_scale']),
        var = crf_params['variance'],
        len_scale=[ crf_params['correlation_scale']['y']*crf_params['c_scale_mul'], 
                    crf_params['correlation_scale']['y']*crf_params['c_scale_mul'], 
                    crf_params['correlation_scale']['y']*crf_params['c_scale_mul']],
        angles=[0, 0, 0]
    )

    # conditional info (points & values)
    cond_pos = (
        df_con["y"].values,
        df_con["x"].values,
        df_con["z"].values
    )
    cond_val = df_con["con_value"].values
    
    
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)
    cond_srf = gs.CondSRF(krige)

    # grid
    cond_srf.set_pos((
        coordinate["y"].values,
        coordinate["x"].values,
        coordinate["z"].values
    ))    
    
    # generate realization
    crf = cond_srf(seed = seed, store=f"K_field{idx}")

    
    return crf

def plot_crf_isosurfaces(df_cor, col_name, df_con, iso_num, iso_range, title):
    """
    Parameters:
    -----------
    df_cor : pandas.DataFrame
        A DataFrame containing the 3D coordinates and the corresponding scalar values for plotting.
        Expected columns: ['x', 'y', 'z', 'value']

    df_con: pandas.DataFrame
        A DataFrame containing the 3D coordinates of the conditional point
        Expected columns: ['x_con', 'y_con', 'z_con', 'con_value']

    col_name : str
        The name of the column in `df_cor` to be used for isosurface plotting.

    iso_num : int
        The number of isosurfaces to be plotted.

    iso_range : list or tuple of float
        The value range for the isosurfaces. Should be specified as [min_value, max_value].

    title : str
        The title of the figure to be displayed.

    Returns:
    --------
    A 3D Plotly figure object with the specified number of isosurfaces plotted within the defined range.
    """
    
    value = df_cor[col_name]
    # print(iso_range)
    iso_min, iso_max = iso_range

    fig = go.Figure()
    
    ## Add isosurface plot
    fig.add_trace(go.Isosurface(
        x = df_cor['x'].values,
        y = df_cor['y'].values,
        z = df_cor['z'].values,
        value = value,
        isomin = iso_min,
        isomax = iso_max,
        surface_count = iso_num,
        colorscale = 'rdylbu',  #'rdylbu'
        showscale=True,
        opacity = 0.5,
        caps=dict(x_show=False, y_show=False, z_show=False),
        
        colorbar=dict(
            title=dict(
                text='K',
                font=dict(
                    size=18,
                    family='Times New Roman',
                    color='black'
                )
            ),
            thickness=30,
            x=0.9,
            len=1.1,
            tickfont=dict(
                size=16,
                family='Times New Roman',
                color='black'
            )
        )
    ))

    ## Add scatter plot for conditional point
    if not df_con.empty:
        fig.add_trace(go.Scatter3d(
            x=df_con['x'],
            y=df_con['y'],
            z=df_con['z'],
            mode='markers',
            marker=dict(
                size=5,
                color='black',      # or use a column for color: df_points['value']
                opacity=0.7,
                symbol='circle',
            ),
            name='Conditional Points'  # name for legend
        ))

    
    # styling
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'font': {'size': 36, 'family': 'Times New Roman'}
        },

        legend=dict(
            x=0.6,
            y=0.1,
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor='black',
            borderwidth=1,
            font=dict(
                family="Times New Roman",
                size=16
        )),

        scene=dict(

            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),

            xaxis=dict(
                gridcolor='lightgray',
                showbackground=True,
                showline=True,
                linecolor='black',
                linewidth=2,
                showticklabels=False,
            ),
            yaxis=dict(
                gridcolor='lightgray',
                showbackground=True,
                showline=True,
                linecolor='black',
                linewidth=2,
                showticklabels=False,
            ),
            zaxis=dict(
                gridcolor='lightgray',
                showbackground=True,
                showline=True,
                linecolor='black',
                linewidth=2,
                showticklabels=False,
            ),
            aspectmode='data'
        ),

        autosize = True,
        showlegend=True,
        width = 750,
        height = 600,
        margin=dict(l=10, r=10, t=50, b=40),
        font=dict(family="Times New Roman", size=20),
    )
    
    return fig

# ====== FORWARD ======= 

def col_to_tensor(col, ele_num_x, ele_num_y, ele_num_z):
    
    layer_list = []
    plane_num = ele_num_y* ele_num_x
    for j in range(0, plane_num*ele_num_z, plane_num):
        layer = col[j:j+plane_num].reshape(ele_num_y, ele_num_x)
        layer_list.append(layer.reshape((-1,) + layer.shape)) # layer K
    tensor = np.concatenate(layer_list, axis=0) # concate at z direction
    
    return tensor
    
def tensor_to_col(tensor, ele_num_z):
    
    layer_K_list = []
    for i in range(ele_num_z):
        layer_K = tensor[i, :, :].flatten()
        layer_K_list.append(layer_K)
    col = np.concatenate(layer_K_list, axis = 0)
    
    return col

def save_to_tensor(field_idx, ele_num_x, ele_num_y, ele_num_z):

    project_dir = Path.cwd()
    rf_dir = project_dir/"material"/f"material_{field_idx}.dat"
    
    df_rf = pd.read_csv( rf_dir , sep="\t", header = 0 )
    k_col = df_rf['K'].values
    
    k_tensor = col_to_tensor(k_col, ele_num_x, ele_num_y, ele_num_z)    
    
    
    return k_tensor

def read_one_stress_FwdObs(event_dir, init_h = None):
    
    header = ['loca', 'well', 'time', 'wlevel','flux_x','flux_y','w_content','solute_concentation','solute_flux', '?']
    df = pd.read_csv(event_dir/"O-FwdObs.dat", header=None,sep=r'\s+', names = header, engine='python')
    df.drop([0], inplace=True)
    df_groupby = df.groupby('well')
        
    df_head = pd.DataFrame()
    well_list = sorted(list(df_groupby.groups), key=len)

    result = []

    for well_name in well_list:

        df_well = df_groupby.get_group(
            well_name
        )

        head = df_well['wlevel'].iloc[0]
        result.append(head)

    result = np.array(result)

    if init_h is not None:
        result = result - np.asarray(init_h)

    return result     

def run_single_case(event_name, field_idx, init_h = None, print_output = None):
    
    print(
    f"[START] "
    f"PID={os.getpid()} | "
    f"Event={event_name} | "
    f"Field={field_idx}"
    )
    
    project_dir = Path.cwd()
    rf_dir = project_dir/"material"/f"material_{field_idx}.dat"
    event_dir = project_dir/"forward"/f"{event_name}" 
    
    
    # All required DataFrames
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

    # Check missing DataFrames
    missing_files = [f for f in forward_required_files if not os.path.exists( event_dir )]
    if missing_files:
        raise FileNotFoundError(f"missing following files\n" +
                                "\n".join(f"  - {file}" for file in missing_files))
    
    # Read random field
    df_rf = pd.read_csv( rf_dir , sep="\t", header = 0 )

    
    # Read material format
    df_material = pd.read_csv( event_dir/'material.dat', sep="\t", header=None )
    
    # Read grid
    total_element_num = np.loadtxt(os.path.join('test_forward', 'grid.dat'))[0].astype(int)

    # Update material
    df_material.columns = ['Kx','Ky','Kz','n','Ss','none1','none2']
    df_material.loc[1:total_element_num, 'Kx'] = df_rf['K'].values
    df_material.loc[1:total_element_num, 'Ky'] = df_rf['K'].values
    df_material.loc[1:total_element_num, 'Kz'] = df_rf['K'].values
    df_material.loc[1:total_element_num, 'Ss']
    df_material.to_csv(event_dir/'material.dat', sep="\t", header=False, index=False)
                
    # Run SLE
    p = Popen( [str(event_dir/'SLE.exe')], cwd = event_dir, stdout=PIPE, stdin=PIPE, stderr=PIPE )

    stdout_data, stderr_data = p.communicate(
        input=b"\n"
    )

    if print_output:
        print(
            stdout_data.decode(
                errors="ignore"
            )
        )
    df_head = read_one_stress_FwdObs(event_dir, init_h)
    
    return df_head


# ====== CRF class ======

class crf_process:
    """ 
    What this class need?
    First, we need the prior information (In VSAFT3 format or in dataframe)
    I think this part we need to stick to a dataframe for all the following process!! 
    
    df with col_name:        ['x', 'y', 'z', 'ln_mean' 'ln_var']
        df[['x', 'y', 'z']]: The coordinate for element center used in VSAFT3 system
        df[['ln_mean']]:     Estimated mean value by SLE inversion (log scale)
        df[['ln_var']]:      The uncertainty estimated during the SLE inversion (log scale)
    
    Next, we conduct the geostatistics analysis (Use gstools)
    Then, we sample the conditional random field and conditional values
    Check the result, and conclude all the crf parameters (correlation scale, variance, ...)
    Finally, we generate multiple random fields to provide VSAFT3 to run forward simulation

    """
    
    def __init__(self, project_name, path):
        
        self.project_name = project_name
        self.work_dir = Path(path).resolve()

        # empty dict for storing parameters
        self.crf_list = []
        self.parameters = {}
        print(f'Project: {self.project_name}')
        print(f'Work on folder:{self.work_dir}')
 
    def get_prior(self, inverse_path = None, iter_idx = None, df_prior = None):
        
        """ 
        If data is in k-esimtated form, use parse fucntion to get the df
        Or if df_prior is exist, input directly
        
        df_prior with col_name: ['x', 'y', 'z', 'ln_mean' 'ln_var']
        
        """
        df_mean, df_var = parse_estimation(inverse_path, dim =3)
        
        df_ = pd.DataFrame({
            'x': df_mean['x'],
            'y': df_mean['y'],
            'z': df_mean['z'],
            f'ln_mean': np.log(df_mean[f'iter_{iter_idx}'].values),
            f'ln_var':  df_var[f'iter_{iter_idx}'],
        })
        
        if df_prior:
            self.df_prior = df_prior
        self.df_prior = df_
        self.df_coordinate = self.df_prior[['x', 'y', 'z']].copy()
        
        return self

    def get_core_section(self, x_range, y_range, z_range, resample = None, grid_size = None):
        """ 
        Parameters
        ----------
        x_range: range for core section in x direction
        y_range: range for core section in y direction
        z_range: range for core section in z direction
        resample: If Ture, 
                    parameter will interpolation by new grid system: grid_size
        
        grid_size: (x_ele, y_ele, z_ele)
        """
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        
        df_prior_core = self.df_prior[
            (self.df_prior['x'] >= x_min)  & (self.df_prior['x'] <= x_max) &
            (self.df_prior['y'] >= y_min)  & (self.df_prior['y'] <= y_max) &
            (self.df_prior['z'] >= z_min)  & (self.df_prior['z'] <= z_max)
        ]
        
        if resample:
            xg, yg, zg, inter_lnK = interpolate_3d(
                                                df_prior_core, 
                                                value_col = 'ln_mean', 
                                                grid_size = grid_size, 
                                                method="linear"
                                                )
            
            self.xg = xg
            self.yg = yg
            self.zg = zg
        
            self.df_geo = pd.DataFrame({    'x': xg.flatten(),
                                            'y': yg.flatten(),
                                            'z': zg.flatten(),
                                    'ln_mean': inter_lnK.flatten()})
          
        else:
            # Use original coordinates directly 
            self.xg = df_prior_core['x'].to_numpy()
            self.yg = df_prior_core['y'].to_numpy()
            self.zg = df_prior_core['z'].to_numpy() 
            
            # Keep original data 
            self.df_geo = df_prior_core[ ['x', 'y', 'z', 'ln_mean'] ].copy()
        
        self.df_prior_core = df_prior_core
          
        return self

    def variogram_analysis(self, x_paras, y_paras, z_paras):
        """ 
        x_paras: (max_estimate_dist, number of bins)
        """
        
        bins_x = np.arange(0, x_paras[0], x_paras[1])
        bins_y = np.arange(0, y_paras[0], y_paras[1])
        bins_z = np.arange(0, z_paras[0], z_paras[1])
        
        variogram_test((self.xg, self.yg, self.zg), self.df_geo['ln_mean'], bins_x, 'X', x_paras[0])
        variogram_test((self.xg, self.yg, self.zg), self.df_geo['ln_mean'], bins_y, 'Y', y_paras[0])
        variogram_test((self.xg, self.yg, self.zg), self.df_geo['ln_mean'], bins_z, 'Z', z_paras[0])
        
        return self

    def set_parameters(self, injs, **kwargs):
        """ 
        Add all parameters for crf 
        
        Parameters
        --------------------
        injs = (coordinate, inj_name, paras)
        
        """

        self.injs = injs
        
        # allow to change parameters
        self.parameters = {
            "ens_num": 10,
            "seed": 20260629,
            "model": "Gaussian",
            "variance": 0.09,
            "correlation_scale": {"x": 50, "y": 60, "z": 47},
            "c_scale_mul": 1.5,
            "section_sample_num": 1,
            "fraction_rate": 0.25,
            "resample_time": 2,
        }
        
        self.parameters.update(kwargs)
        self.parameters["inj_A_num"] = len(injs)
        
        return self 

    def run_generate_crf(self, check = None, save = None):
        
        # Log crf parameters
        log_path = self.work_dir / "crf_parameters.txt"
        
        with open( log_path, "w", encoding="utf-8") as f:
        
            f.write("-" * 50 + "\n")
            f.write(f"Case: {self.project_name}\n")
            f.write("CRF Parameters\n")
            f.write("-" * 50 + "\n\n")

            for key, value in self.parameters.items():
                f.write(  f"{key} = {value}\n" )            
            
        df_est = self.df_prior.copy()
        df_est_core = self.df_prior_core.copy()
        
        rng = MasterRNG(self.parameters["seed"])
        
        w_coor = np.array(
            [coordinate for coordinate, _, _ in self.injs],
            dtype=float
        )
        ele_coor = self.df_prior[['x', 'y', 'z']].to_numpy()
        
        df_est['weight'] = d2_weight(ele_coor, w_coor, 10**6)

        for i in range(self.parameters['resample_time']):
            
            sub_num = self.parameters['ens_num']//self.parameters['resample_time']
            print(f'process resample group {i+1}/{self.parameters["resample_time"]}')
            print("-"*30)
            
            df_con = sampler(df_est, self.parameters)
            
            for j in range(sub_num):
                seed = int(rng())
                process_index_ =  sub_num*i + j
                crf = crf_generator(self.df_coordinate, df_con, self.parameters, seed, j+1) 
                print(f"Generating field {process_index_+1}   K mean:{np.mean(np.exp(crf)):.3f}")    
                
                self.crf_list.append(crf)
                if save:
                    
                    save_path = self.work_dir / f"material_{process_index_ + 1}.dat"
                    # swtich back to normal scale
                    df_material = pd.DataFrame({'K': np.exp(crf)})
                    df_material.to_csv(save_path, header=True, index=False)
                                    
                if check:
                    df_crf = pd.DataFrame({
                        'x': self.df_coordinate['x'].values,
                        'y': self.df_coordinate['y'].values,
                        'z': self.df_coordinate['z'].values,
                        'crf_value': crf,
                    })

                    x_min, x_max = self.x_range
                    y_min, y_max = self.y_range
                    z_min, z_max = self.z_range    
                    
                    df_crf_core = df_crf[
                        (df_crf['x'] >= x_min) & (df_crf['x']  <= x_max) &
                        (df_crf['y'] >= y_min) & (df_crf['y']  <= y_max) &
                        (df_est['z'] >= z_min) &  (df_est['z'] <= z_max)
                    ]
                    
                    lower_ = df_est_core['ln_mean'].values.max()
                    upper_ = df_est_core['ln_mean'].values.min()
                    print(f" R2 score: { r2_score(df_est_core['ln_mean'].values, df_crf_core['crf_value']):.3f}")
                    fig = plot_crf_isosurfaces(df_crf, 'crf_value', df_con, 8, [lower_, upper_], f'Crf_{j+1}')
                    fig.show()
                
        return self
    
    def col_to_tensor(self, col, ele_num_x, ele_num_y, ele_num_z):
        
        layer_list = []
        plane_num = ele_num_y* ele_num_x
        for j in range(0, plane_num*ele_num_z, plane_num):
            layer = col[j:j+plane_num].reshape(ele_num_y, ele_num_x)
            layer_list.append(layer.reshape((-1,) + layer.shape)) # layer K
        tensor = np.concatenate(layer_list, axis=0) # concate at z direction
        
        return tensor
        
    def tensor_to_col(self, tensor, ele_num_z):
        
        layer_K_list = []
        for i in range(ele_num_z):
            layer_K = tensor[i, :, :].flatten()
            layer_K_list.append(layer_K)
        col = np.concatenate(layer_K_list, axis = 0)
        
        return col

       
# if __name__ == "__main__":  
    
#     crf = crf_process('test', Path.cwd()/'material')
#     crf.get_prior('inverse/O-kestimate_prior.dat', 16)
#     crf.get_core_section((132, 445), 
#                          (96,  286), 
#                          (95,  195), 
#                          resample = True, 
#                          grid_size = (150, 60, 80))
    
#     crf.variogram_analysis((200, 10), (150, 10), (120, 10))
    
    