# -*- coding: utf-8 -*-
"""
- This python code allow user to perfrom 3D SSDS process
- Include variogram estimation, conditional point selection and random field generation

Author: PseudoCaliina
Date: 2026-07-24
"""
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
    df_ = df.groupby(['x_section', 'y_section', 'z_section']).sample(n = 1).reset_index(drop=True)
    
    # --- by distance from inj_A (*fraction_rate)
    df_spatial_sample = df_.sample(frac = crf_params['fraction_rate'], weights = df_['weight']).reset_index(drop=True)
    
    # sampling conditional value (lnK)
    df_spatial_sample['con_value'] = df_spatial_sample.apply(
        
        # uniform sampler
        
        lambda r:np.random.uniform(r['mean']*crf_params['fraction_rate'], r['mean'], 1)[0], axis = 1
        
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
    print(iso_range)
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

class crf_process:
    """ 
    What this class need?
    First, we need the prior information (In VSAFT3 format or in dataframe)
    I think this part we need to stick to a dataframe for all the following process!! 
    
    df with col_name: ['x', 'y', 'z', 'mean' 'variance']
        df[['x', 'y', 'z']]: The coordinate for element center used in VSAFT3 system
        df[['mean']]:        Estimated mean value by SLE inversion
        df[['variance']]:    The uncertainty estimated during the SLE inversion 
    
    Next, we conduct the geostatistics analysis (Use gstool)
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
        
        df_prior with col_name: ['x', 'y', 'z', 'mean' 'variance']
        
        """
        df_mean, df_var = parse_estimation(inverse_path, dim =3)
        
        df_ = pd.DataFrame({
            'x': df_mean['x'],
            'y': df_mean['y'],
            'z': df_mean['z'],
            f'mean': df_mean[f'iter_{iter_idx}'],
            f'var': df_var[f'iter_{iter_idx}'],
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
        
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        df_prior_core = self.df_prior[
            (self.df_prior['x'] >= x_min)  & (self.df_prior['x'] <= x_max) &
            (self.df_prior['y'] >= y_min)  & (self.df_prior['y'] <= y_max) &
            (self.df_prior['z'] >= z_min)  & (self.df_prior['z'] <= z_max)
        ]
        
        if resample:
            xg, yg, zg, inter_inK = interpolate_3d(
                                                df_prior_core, 
                                                value_col = 'mean', 
                                                grid_size = grid_size, 
                                                method="linear"
                                                )
            
            self.xg = xg
            self.yg = yg
            self.zg = zg
        
            self.df_geo = pd.DataFrame({    'x': xg.flatten(),
                                            'y': yg.flatten(),
                                            'z': zg.flatten(),
                                    'mean': inter_inK.flatten()})
          
          
        self.df_prior_core = df_prior_core
          
        return self

    def variogram_analysis(self, x_paras, y_paras, z_paras):
        """ 
        x_paras: (max_estimate_dist, number of bins)
        """
        
        bins_x = np.arange(0, x_paras[0], x_paras[1])
        bins_y = np.arange(0, y_paras[0], y_paras[1])
        bins_z = np.arange(0, z_paras[0], z_paras[1])
        
        
        plt.figure(figsize = (10, 6))
        variogram_test((self.xg, self.yg, self.zg), self.df_geo['mean'], bins_x, 'X', x_paras[0])
        plt.show()
        variogram_test((self.xg, self.yg, self.zg), self.df_geo['mean'], bins_y, 'Y', x_paras[0])
        plt.show()
        variogram_test((self.xg, self.yg, self.zg), self.df_geo['mean'], bins_z, 'Z', x_paras[0])
        plt.show()
        
        
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
            "contain_rate": 0.5
        }
        
        self.parameters.update(kwargs)
        self.parameters["inj_A_num"] = len(injs)
        
        return self 

    def run_generate_crf(self, check = None):
        
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
            # print(len(df_con))
            
            for j in range(sub_num):
                seed = int(rng())
                process_index_ =  sub_num*i + j
                crf = crf_generator(self.df_coordinate, df_con, self.parameters, seed, j+1) 
                # plt.scatter(df_est['K'].values, crf)
                print(f"Generating field {process_index_+1}   lnK mean:{np.mean(crf):.3f}")    
                
                self.crf_list.append(crf)
                
                
                if check:
                    df_crf = pd.DataFrame({
                        'x': self.df_coordinate['x'].values,
                        'y': self.df_coordinate['y'].values,
                        'z': self.df_coordinate['z'].values,
                        'value': crf,
                    })

                    df_crf_core = df_crf[
                        (df_crf['x'] >= 132) & (df_crf['x'] <= 445) &
                        (df_crf['y'] >= 96) & (df_crf['y'] <= 286) &
                        (df_est['z'] >= 95) &  (df_est['z'] <= 195)
                    ]
                    print(f"{ r2_score(df_est_core['K'].values, df_crf_core['value']):.5f}")
                    fig = plot_crf_isosurfaces(df_crf, 'value', df_con, 8, [1, 2], f'Crf_{j+1}')
                    fig.show()
               
                
        return self

if __name__ == "__main__":  
    
    crf = crf_process('test', Path.cwd()/'material')
    crf.get_prior('inverse/O-kestimate_prior.dat', 16)
    crf.get_core_section((132, 445), 
                         (96,  286), 
                         (95,  195), 
                         resample = True, 
                         grid_size = (150, 60, 80))
    
    crf.variogram_analysis((200, 10), (150, 10), (120, 10))
    
    