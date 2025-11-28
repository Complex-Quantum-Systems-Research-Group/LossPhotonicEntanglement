from compute_entanglement import von_neumann_entropy, mutual_information
import numpy as np
# ============================================================================
# PARAMETERS
# ============================================================================
n_max = 4
N_spins = 6
omega1 = 1.0
omega2 = 1.0
g1 = 2.0
g2 = 2.0
J = -1.0
delta = 0.5
tau_1 = 0.05
tau_2 = 0.05
final_evolution_time = 2.0
temperature_list = np.arange(0.5,2.0,0.2)
delta_t_list = np.arange(0.5,2.05,0.2)
von_neumann_list = []
mutual_information_list = []

for Temp_spin in temperature_list:
    for delta_t in delta_t_list:
        filename_von_neumann = f"von_neumann_entropy_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        filename_mutual_info = f"mutual_information_Nspins={N_spins}_nmax={n_max}_omega1={omega1:1.2f}_omega2={omega2:1.2f}_J{J:1.2f}_delta={delta:1.2f}_T={Temp_spin:1.2f}_tau1={tau_1:1.2f}_tau2={tau_2:1.2f}_delta_t={delta_t:1.2f}_final_t={final_evolution_time:1.2f}.npy"
        von_neumann_entropy=np.load(filename_von_neumann, allow_pickle=True)
        mutual_information=np.load(filename_mutual_info, allow_pickle=True)
        print(type(von_neumann_list))
        von_neumann_list.append((Temp_spin, delta_t, von_neumann_entropy))
        mutual_information_list.append((Temp_spin, delta_t, mutual_information))

von_neumann_list = np.array(von_neumann_list)
mutual_information_list = np.array(mutual_information_list)
np.save("von_neumann_entropy_summary.npy", von_neumann_list)
np.save("mutual_information_summary.npy", mutual_information_list)

#plot heatmaps using matplotlib of von_neumann and mutual_information as a function of Temp_spin and delta_t
import matplotlib.pyplot as plt
import seaborn as sns
# Reshape data for heatmap
von_neumann_heatmap = von_neumann_list[:,2].reshape(len(temperature_list), len(delta_t_list))
mutual_information_heatmap = mutual_information_list[:,2].reshape(len(temperature_list), len(delta_t_list))
# Plot Von Neumann Entropy Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(von_neumann_heatmap, xticklabels=np.round(delta_t_list,2), yticklabels=np.round(temperature_list,2), cmap="YlGnBu")
plt.title("Von Neumann Entropy Heatmap")
plt.xlabel("Delta t")
plt.ylabel("Temperature Spin")
plt.savefig("von_neumann_entropy_heatmap.png")
plt.close()
# Plot Mutual Information Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(mutual_information_heatmap, xticklabels=np.round(delta_t_list,2), yticklabels=np.round(temperature_list,2), cmap="YlGnBu")
plt.title("Mutual Information Heatmap")
plt.xlabel("Delta t")
plt.ylabel("Temperature Spin")
plt.savefig("mutual_information_heatmap.png")
plt.close()
