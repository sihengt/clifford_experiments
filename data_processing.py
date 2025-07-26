import numpy as np
import pandas as pd
import os
import glob
import sys
import matplotlib.pyplot as plt

def main(folder_path):
    corrective = True
    if folder_path == "results_full":
        corrective = False
        
    # Cross track error.
    l_xdot_sim = sorted(glob.glob(os.path.join(folder_path, "numpy/xdot_sim_track*.npy")))
    l_xdot_kbm = sorted(glob.glob(os.path.join(folder_path, "numpy/xdot_kbm_track*.npy")))
    l_xdot_res = sorted(glob.glob(os.path.join(folder_path, "numpy/xdot_res_track*.npy")))
    l_sim = [np.load(file) for file in l_xdot_sim]
    l_kbm = [np.load(file) for file in l_xdot_kbm]
    l_res = [np.load(file) for file in l_xdot_res]
    
    rmse_sim_kbm_results = np.zeros((9, 4))
    rmse_sim_res_results = np.zeros((9, 4))
    for i in range(len(l_sim)):
        np_sim = l_sim[i]
        np_kbm = l_kbm[i]
        np_res = l_res[i]

        rmse_sim_kbm = np.sqrt(np.mean((np_sim - np_kbm) ** 2, axis=0))
        
        if corrective:
            rmse_sim_res = np.sqrt(np.mean((np_sim - (np_kbm + np_res)) ** 2, axis=0))
        else:
            rmse_sim_res = np.sqrt(np.mean((np_sim - np_res) ** 2, axis=0))

        rmse_sim_kbm_results[i, :] = rmse_sim_kbm
        rmse_sim_res_results[i, :] = rmse_sim_res

    df_kbm = pd.DataFrame(rmse_sim_kbm_results, columns=["xdot", "ydot", "vdot", "psidot"])
    df_kbm.to_csv(os.path.join(folder_path, f"results/df_kbm_{folder_path}.csv"))
    df_crn = pd.DataFrame(rmse_sim_res_results, columns=["xdot", "ydot", "vdot", "psidot"])
    df_crn.to_csv(os.path.join(folder_path, f"results/df_crn_{folder_path}.csv"))

    # 1) Bar Plot (Average RMSE per State Variable)
    state_names = ["$\dot{x}$", "$\dot{y}$", "a", r"$\dot{\psi}$"]
    df_cols = ["xdot", "ydot", "vdot", "psidot"]

    mean_rmse_crn = df_crn.mean().values
    mean_rmse_kbm = df_kbm.mean().values

    x = np.arange(len(state_names))
    width = 0.35

    plt.figure(figsize=(8, 6), dpi=300)
    plt.bar(x - width/2, mean_rmse_crn, width, label='CRN')
    plt.bar(x + width/2, mean_rmse_kbm, width, label='KBM')

    plt.ylabel('Average RMSE')
    plt.xticks(x, state_names)
    plt.title('Average RMSE per State Variable (CRN vs KBM)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, "results/AverageRMSEperState.pdf"), bbox_inches="tight", dpi=300)

    plt.figure(figsize=(8, 6), dpi=300)
    rmse_diff = df_kbm[["xdot", "ydot", "vdot", "psidot"]] - df_crn[["xdot", "ydot", "vdot", "psidot"]]
    rmse_diff["track"] = df_crn.index
    rename_dict = dict(zip(df_cols, state_names))
    rmse_diff = rmse_diff.rename(columns=rename_dict)

    rmse_diff.plot(
        x="track",
        y=state_names,
        kind="bar", figsize=(10, 4),
        title="RMSE Reduction (KBM - CRN)")
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("RMSE Difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "results/RMSEReduction.pdf"), bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    print(sys.argv)
    assert len(sys.argv) == 2
    fp = sys.argv[1]
    assert(os.path.isdir(fp))
    main(fp)