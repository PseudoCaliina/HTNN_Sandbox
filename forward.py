import os
import math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from subprocess import Popen, PIPE
from multiprocessing import Pool

ens_num = 5
stress_num = 8
event_names = [f"inj_{i+1}" for i in range(stress_num )]

# fucntion
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


import time
import numpy as np
from multiprocessing import Pool


if __name__ == "__main__":

    n_processes = 8
    field_indices = [1, 2]
    head_matrices = []

    total_start_time = time.perf_counter()

    for field_idx in field_indices:

        print("=" * 60)

        print(
            f"Start Field {field_idx}"
        )

        print("=" * 60)


        # --------------------------------------
        # 建立這個 RF 的 8 個 Jobs
        # --------------------------------------

        jobs = [
            (
                event_name,
                field_idx,
                None,
                False
            )
            for event_name in event_names
        ]


        # --------------------------------------
        # 計算 Field 開始時間
        # --------------------------------------

        field_start_time = time.perf_counter()


        # --------------------------------------
        # 平行執行 8 個 Injection Events
        # --------------------------------------

        with Pool(
            processes=n_processes
        ) as pool:

            results = pool.starmap(
                run_single_case,
                jobs
            )


        # --------------------------------------
        # 組合成 8×8 Matrix
        # --------------------------------------

        head_matrix = np.vstack(
            results
        )


        # --------------------------------------
        # 計算 Field 執行時間
        # --------------------------------------

        field_elapsed_time = (
            time.perf_counter()
            - field_start_time
        )


        # --------------------------------------
        # 儲存結果
        # --------------------------------------

        head_matrices.append(
            head_matrix
        )


        # --------------------------------------
        # 顯示結果
        # --------------------------------------

        print(
            f"Field {field_idx} completed."
        )

        print(
            f"Matrix shape: "
            f"{head_matrix.shape}"
        )

        print(
            f"Field {field_idx} elapsed time: "
            f"{field_elapsed_time:.2f} seconds"
        )

        print(
            f"Field {field_idx} elapsed time: "
            f"{field_elapsed_time / 60:.2f} minutes"
        )

        print(
            head_matrix
        )


    # ==========================================
    # 組合所有 RF
    # ==========================================

    head_matrices = np.array(
        head_matrices
    )


    # ==========================================
    # 計算總時間
    # ==========================================

    total_elapsed_time = (
        time.perf_counter()
        - total_start_time
    )


    # ==========================================
    # 最終結果
    # ==========================================

    print("=" * 60)

    print(
        "All simulations completed."
    )

    print(
        "Final result shape:",
        head_matrices.shape
    )

    print(
        f"Total elapsed time: "
        f"{total_elapsed_time:.2f} seconds"
    )

    print(
        f"Total elapsed time: "
        f"{total_elapsed_time / 60:.2f} minutes"
    )

    print("=" * 60)
    
    
# Field_i (8*8 matrix)
# │
# ├── inj_1 → [w1, w2, w3, w4, w5, w6, w7, w8]
# ├── inj_2 → [w1, w2, w3, w4, w5, w6, w7, w8]
# ├── inj_3 → [w1, w2, w3, w4, w5, w6, w7, w8]
# ├── inj_4 → [w1, w2, w3, w4, w5, w6, w7, w8]
# ├── inj_5 → [w1, w2, w3, w4, w5, w6, w7, w8]
# ├── inj_6 → [w1, w2, w3, w4, w5, w6, w7, w8]
# ├── inj_7 → [w1, w2, w3, w4, w5, w6, w7, w8]
# └── inj_8 → [w1, w2, w3, w4, w5, w6, w7, w8]