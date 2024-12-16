'''
Skript zum Aufruf von Plots, um das main-notebook übersichtlicher zu machen
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''
Plots für importierte Daten
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#


def plot_data(data_dict, figsize=(18, 12), title=""):
    fig, axs = plt.subplots(3, 3, figsize=figsize)

    for i in range(len(data_dict)):
        for row in range(3):
            for col in range(3):
                # Stress-Strain plot
                axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['P'][row, col], color='black', s=10)
                
                # Strain energy density - Strain plot
                #axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['W'], color='green', s=10)
                
                # Labels nur einmal hinzufügen
                if i == 0:
                    axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['P'][row, col], color='black', s=10, label='Stress')
                    #axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['W'], color='green', s=10, label='Strain energy density ')

    for row in range(3):
        for col in range(3):
            # Titel für jeden Subplot
            axs[row, col].set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")

            # X-Achsenbeschriftungen für jeden Subplot variabel anpassen
            axs[row, col].set_xlabel(f"F{row+1}{col+1}")  # z.B. F11, F12, F13...

            # Y-Achsenbeschriftungen für jeden Subplot variabel anpassen
            axs[row, col].set_ylabel(f"P{row+1}{col+1}")  # z.B. P11, P12, P13...

            # Legende hinzufügen, falls sie noch nicht existiert
            axs[row, col].legend()
    
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.grid()
    plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''
Vergleich zwischen P aus gradientem von W und P gegeben 
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_dWdF(data_dict, P, load, figsize=(18, 12), title=""):
    fig, axs = plt.subplots(3, 3, figsize=figsize) 


    for i in range(len(data_dict)):
        for row in range(3):
            for col in range(3):
                # Stress-Strain plot
                axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['P'][row, col], color='black', s=10)
                axs[row, col].scatter(data_dict[i]['F'][row, col], np.array(P[load][i][row, col]), color='red', s=5)
                    

                # Strain energy density - Strain plot
                #axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['W'], color='green', s=10)
                    
                # Labels nur einmal hinzufügen
                if i == 0:
                    axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['P'][row, col], color='black', s=10, label='Stress')
                    axs[row, col].scatter(data_dict[i]['F'][row, col], np.array(P[load][i][row, col]), color='red', s=5, label='Gradient of W')
                    #axs[row, col].scatter(data_dict[i]['F'][row, col], data_dict[i]['W'], color='green', s=10, label='Strain energy density ')

    for row in range(3):
        for col in range(3):
            # Titel für jeden Subplot
            axs[row, col].set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")

            # X-Achsenbeschriftungen für jeden Subplot variabel anpassen
            axs[row, col].set_xlabel(f"F{row+1}{col+1}")  # z.B. F11, F12, F13...

            # Y-Achsenbeschriftungen für jeden Subplot variabel anpassen
            axs[row, col].set_ylabel(f"P{row+1}{col+1}")  # z.B. P11, P12, P13...

            # Legende hinzufügen, falls sie noch nicht existiert
            axs[row, col].legend()
    
    fig.suptitle(title, fontsize=16)


    
    plt.tight_layout()
    plt.grid()
    plt.show()


'''
Task2 
'''

def plot_task2_final(data_dict, prediction_list, step, figsize=(18, 12), title="", plot_over_F=None, dpi=500):
    

    if plot_over_F is None:
        plot_over_F = [(0, 0), (1, 1), (2, 2)]  # Default

    fig, axs = plt.subplots(3, 3, figsize=figsize)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
            # Daten vorbereiten
            P_values = [data_dict[i]['P'][row, col] for i in range(len(data_dict))]
            P_predictions = [prediction_list[i][row, col] for i in range(len(data_dict))]

            if (row, col) in plot_over_F:  # Plot über F
                F_values = [data_dict[i]['F'][row, col] for i in range(len(data_dict))]

                # Scatter: Nur jeden step-ten Punkt
                axs[row, col].scatter(F_values[::step], P_values[::step], color='black', s=40, marker='<', label='Stress - true')

                # Linienplot: Alle Punkte
                axs[row, col].plot(F_values, P_predictions, color='black', label='Stress - prediction', linewidth=3)

                # Achsenbeschriftungen
                axs[row, col].set_xlabel(f"F{row+1}{col+1}", fontsize=16)
            else:  # Plot über Index
                indices = list(range(len(data_dict)))

                # Scatter: Nur jeden step-ten Punkt
                axs[row, col].scatter(indices[::step], P_values[::step], color='black', s=40, marker='<', label='Stress - true')

                # Linienplot: Alle Punkte
                axs[row, col].plot(indices, P_predictions, color='black', label='Stress - prediction', linewidth=3)

                # Achsenbeschriftungen
                axs[row, col].set_xlabel("Index", fontsize=18)

            # Subplot-Titel und allgemeine Achsenbeschriftungen
            axs[row, col].set_title(f"P{row+1}{col+1}", fontsize=18)
            axs[row, col].set_ylabel(f"P{row+1}{col+1}", fontsize=18)

            axs[row, col].tick_params(axis='both', labelsize=18)



            axs[row, col].legend()

    # Gesamttitel und Layout anpassen
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
  
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''
Task 3 
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_task3_final(data_dict, prediction_list, W_list_true, W_list_pred, step, figsize=(18, 12), title="", plot_over_F=None):
    import matplotlib.pyplot as plt

    if plot_over_F is None:
        plot_over_F = [(0, 0), (1, 1), (2, 2)]  # Standardmäßig Plots über F

    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=500)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
            # Daten vorbereiten
            P_values = [data_dict[i]['P'][row, col] for i in range(len(data_dict))]
            P_predictions = [prediction_list[i][row, col] for i in range(len(data_dict))]
            W_true_values = [W_list_true[i] for i in range(len(W_list_true))]
            W_pred_values = [W_list_pred[i] for i in range(len(W_list_pred))]

            if (row, col) in plot_over_F:  # Plot über F
                F_values = [data_dict[i]['F'][row, col] for i in range(len(data_dict))]

                # Primäre y-Achse für P
                ax1 = axs[row, col]
                ax1.scatter(F_values[::step], P_values[::step], color='black', s=40, marker='<', label='Stress - true')
                ax1.plot(F_values, P_predictions, color='black', label='Stress - prediction', linewidth=3)

                ax1.set_xlabel(f"F{row+1}{col+1}", fontsize=16)
            else:  # Plot über Index
                indices = list(range(len(data_dict)))

                ax1 = axs[row, col]
                ax1.scatter(indices[::step], P_values[::step], color='black', s=40, marker='<', label='Stress - true')
                ax1.plot(indices, P_predictions, color='black', label='Stress - prediction', linewidth=3)

                ax1.set_xlabel("Index", fontsize=16)

            ax2 = ax1.twinx()
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.spines['right'].set_color('red')

            if (row, col) in plot_over_F:
                ax2.scatter(F_values[::step], W_true_values[::step], color='red', marker='o', s=40, label='Energy - true')
                ax2.plot(F_values, W_pred_values, color='red', label='Energy - prediction', linewidth=3)
            else:
                ax2.scatter(indices[::step], W_true_values[::step], color='red', marker='o', s=40, label='Energy - true')
                ax2.plot(indices, W_pred_values, color='red', label='Energy - prediction', linewidth=3)

            ax1.set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}", fontsize=16)
            ax1.set_ylabel(f"P{row+1}{col+1}", fontsize=16)
            ax2.set_ylabel(f"W{row+1}{col+1}", fontsize=16, color='red')

            ax1.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0, 1))
            ax2.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0, 0.85))

    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()



#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
'''
Task 4 
'''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_task4(F_true, P_true, P_pred, step, figsize=(18, 12), title=""):
    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
       
            F_values = [F_true[i][row, col] for i in range(len(F_true))]
            P_values = [P_true[i][row, col] for i in range(len(P_true))]
            P_predictions = [P_pred[i][row, col] for i in range(len(P_pred))]

            axs[row, col].scatter(F_values[::step], P_values[::step], color='black', s=40, marker='o', label='Stress - true')

            axs[row, col].plot(F_values, P_predictions, color='black', label='Stress - prediction', linewidth=3)
            
            axs[row, col].set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")
            axs[row, col].set_xlabel(f"F{row+1}{col+1}")
            axs[row, col].set_ylabel(f"P{row+1}{col+1}")

            axs[row, col].legend()


    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def loadpaths(F_true, P_true, step, loadpath_length=50, figsize=(18, 12), title=""):
 
    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)

    num_loadpaths = len(F_true) // loadpath_length // 6 # Darstellung von 10 Lastpfaden

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
            for lp_idx in range(num_loadpaths):

                start_idx = lp_idx * loadpath_length
                end_idx = start_idx + loadpath_length

                F_values = [F_true[i][row, col] for i in range(start_idx, end_idx)]
                P_values = [P_true[i][row, col] for i in range(start_idx, end_idx)]

                axs[row, col].scatter(
                    F_values[::step], 
                    P_values[::step], 
                    label=f'Loadpath {lp_idx + 1}', 
                    s=40, marker='o')

            axs[row, col].set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")
            axs[row, col].set_xlabel(f"F{row+1}{col+1}")
            axs[row, col].set_ylabel(f"P{row+1}{col+1}")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()



def plot_task4_PANN(F_true, P_true, P_pred, W_true, W_pred, step, figsize=(18, 12), title=""):
    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
           
            F_values = [F_true[i][row, col] for i in range(len(F_true))]
            P_values = [P_true[i][row, col] for i in range(len(P_true))]
            W_values = [W_true[i] for i in range(len(W_true))]

            P_predictions = [P_pred[i][row, col] for i in range(len(P_pred))]
            W_predictions = [W_pred[i] for i in range(len(W_pred))]

            ax1 = axs[row, col]

            ax2 = ax1.twinx()

            ax1.scatter(F_values[::step], P_values[::step], color='black', s=40, marker='o', label='Stress - true')
            ax1.plot(F_values, P_predictions, color='black', label='Stress - prediction', linewidth=3)
            ax1.set_xlabel(f"F{row+1}{col+1}")
            ax1.set_ylabel(f"P{row+1}{col+1}", color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            ax2.scatter(F_values[::step], W_values[::step], color='red', s=40, marker='x', label='Energy - true')
            ax2.plot(F_values, W_predictions, color='red', label='Energy - prediction', linewidth=3)
            ax2.set_ylabel(f"W{row+1}{col+1}", color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax1.set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()




'''
Task 5
'''

def Task5_1_train(dict_bcc_uniaxial_c, dict_bcc_biaxial_c, dict_bcc_shear_c, dict_bcc_planar_c, dict_bcc_volumetric_c, step, title='', figsize=(18, 12)):
    # Definieren der Subplot-Größe
    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)
  
    data_dict = {
        'uniaxial': dict_bcc_uniaxial_c,
        'biaxial': dict_bcc_biaxial_c,
        'shear': dict_bcc_shear_c,
        'planar': dict_bcc_planar_c,
        'volumetric': dict_bcc_volumetric_c
    }
    
    # Iterieren durch alle Subplot-Positionen
    for row in range(3):
        for col in range(3):
            # Listen zum Speichern der F- und P-Werte
            F_values = []
            P_values = []

            for case, data_dict_case in data_dict.items():
                F = data_dict_case['F']  
                P = data_dict_case['P'] 
               
                F_values.append(F[::step, row, col]) 
                P_values.append(P[::step, row, col])  

            F_values = [f.numpy() if hasattr(f, 'numpy') else f for f in F_values]
            P_values = [p.numpy() if hasattr(p, 'numpy') else p for p in P_values]

            for i, case in enumerate(data_dict.keys()):
                axs[row, col].scatter(F_values[i], P_values[i], label=case, alpha=1, s=15)

            axs[row, col].set_xlabel(f'F{row+1}{col+1}')
            axs[row, col].set_ylabel(f'P{row+1}{col+1}')
            axs[row, col].grid(True, linestyle='--', alpha=0.7)

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.suptitle(title, fontsize=16)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True, fontsize=14)
    
    plt.show()

def Task5_1_test(dict_bcc_test1, dict_bcc_test2, dict_bcc_test3, step, title='', figsize=(18, 12)):
    # Definieren der Subplot-Größe
    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)
    
    # Erstellen einer Liste der verschiedenen Belastungsarten und ihrer zugehörigen Daten
    data_dict = {
        'Test 1': dict_bcc_test1,
        'Test 2': dict_bcc_test2,
        'Test 3': dict_bcc_test3,
    }
    
    # Iterieren durch alle Subplot-Positionen
    for row in range(3):
        for col in range(3):
      
            F_values = []
            P_values = []

            for case, data_dict_case in data_dict.items():
                F = data_dict_case['F'] 
                P = data_dict_case['P'] 
                
                F_values.append(F[::step, row, col])  
                P_values.append(P[::step, row, col])  

            F_values = [f.numpy() if hasattr(f, 'numpy') else f for f in F_values]
            P_values = [p.numpy() if hasattr(p, 'numpy') else p for p in P_values]

            for i, case in enumerate(data_dict.keys()):
                axs[row, col].scatter(F_values[i], P_values[i], label=case, alpha=1, s=15)

            axs[row, col].set_xlabel(f'F{row+1}{col+1}')
            axs[row, col].set_ylabel(f'P{row+1}{col+1}')
            axs[row, col].grid(True, linestyle='--', alpha=0.7)

    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.suptitle(title, fontsize=16)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=True, fontsize=14)
    
    plt.show()            


def task5_2_final(F_true, P_true, P_pred, W_true, W_pred, step, figsize=(18, 12), title="", plot_over_F=None):

    if plot_over_F is None:
        plot_over_F = [(0, 0), (1, 1), (2, 2)]  # Standardmäßig Plots über F

    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=500)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
            # Daten vorbereiten
            F_values = [F_true[i][row, col] for i in range(len(F_true))]
            P_values = [P_true[i][row, col] for i in range(len(P_true))]
            P_predictions = [P_pred[i][row, col] for i in range(len(P_pred))]
            W_true_values = [W_true[i] for i in range(len(W_true))]
            W_pred_values = [W_pred[i] for i in range(len(W_pred))]

            ax1 = axs[row, col]
            ax2 = ax1.twinx()

            if (row, col) in plot_over_F:  # Plot über F
                ax1.scatter(F_values[::step], P_values[::step], color='black', s=40, marker='o', label='Stress - true')
                ax1.plot(F_values, P_predictions, color='black', linewidth=3, label='Stress - prediction')

                ax2.scatter(F_values[::step], W_true_values[::step], color='red', s=40, marker='x', label='Energy - true')
                ax2.plot(F_values, W_pred_values, color='red', linewidth=3, label='Energy - prediction')

                ax1.set_xlabel(f"F{row+1}{col+1}", fontsize=18)
            else:  # Plot über Index
                indices = list(range(len(F_true)))

                ax1.scatter(indices[::step], P_values[::step], color='black', s=40, marker='o', label='Stress - true')
                ax1.plot(indices, P_predictions, color='black', linewidth=3, label='Stress - prediction')

                ax2.scatter(indices[::step], W_true_values[::step], color='red', s=40, marker='x', label='Energy - true')
                ax2.plot(indices, W_pred_values, color='red', linewidth=3, label='Energy - prediction')

                ax1.set_xlabel("Index", fontsize=18)

            ax1.set_ylabel(f"P{row+1}{col+1}", color='black', fontsize=18)
            ax1.tick_params(axis='y', labelcolor='black')

            ax2.set_ylabel(f"W{row+1}{col+1}", color='red', fontsize=18)
            ax2.tick_params(axis='y', labelcolor='red')

            ax1.set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def checkobjectivity_stress(F_true, P_true, P_pred, P_pred_rot, step, figsize=(18, 12), title="", plot_over_F=None):
    if plot_over_F is None:
        plot_over_F = [(0, 0), (1, 1), (2, 2)]  # Standardmäßig Plots über F

    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
            # Daten vorbereiten
            try:
                F_values = [F_true[i][row, col] for i in range(len(F_true))]
                P_values = [P_true[i][row, col] for i in range(len(P_true))]
                P_predictions = [P_pred[i][row, col] for i in range(len(P_pred))]
                P_predictions_rot = [P_pred_rot[i][row, col] for i in range(len(P_pred_rot))]
            except IndexError:
                print(f"IndexError bei (row={row}, col={col}). Daten unvollständig.")
                continue

            ax1 = axs[row, col]

            if (row, col) in plot_over_F:  # Plot über F
                if len(F_values) != len(P_values) or len(F_values) != len(P_predictions):
                    print(f"Datenlänge stimmt nicht überein bei (row={row}, col={col})")
                    continue
                ax1.scatter(F_values[::step], P_values[::step], color='black', s=40, marker='o', label='Stress - true')
                ax1.plot(F_values, P_predictions, color='black', linewidth=3, label='Stress - prediction')
                ax1.plot(F_values, P_predictions_rot, color='grey', linewidth=3, linestyle='--', label='Stress - prediction rotate')
                ax1.set_xlabel(f"F{row+1}{col+1}", fontsize=18)
            else:  # Plot über Index
                indices = list(range(len(F_true)))
                if len(indices) != len(P_values) or len(indices) != len(P_predictions):
                    print(f"Datenlänge stimmt nicht überein bei (row={row}, col={col})")
                    continue
                ax1.scatter(indices[::step], P_values[::step], color='black', s=40, marker='o', label='Stress - true')
                ax1.plot(indices, P_predictions, color='black', linewidth=3, label='Stress - prediction')
                ax1.plot(indices, P_predictions_rot, color='grey', linewidth=3, linestyle='--', label='Stress - prediction rotate')
                ax1.set_xlabel("Index", fontsize=18)

            ax1.set_ylabel(f"P{row+1}{col+1}", color='black', fontsize=18)
            ax1.tick_params(axis='y', labelcolor='black')

            ax1.set_title(f"F{row+1}{col+1} vs. P{row+1}{col+1}")
            ax1.legend(loc='upper left')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Platz für den Suptitel lassen
    plt.show()





# Hier für uniaxial und shear trainingsdaten
def task5_2_paper_plot_train(F_true_uniax, P_true_uniax, P_pred_uniax, F_true_shear, P_true_shear, P_pred_shear, step, figsize=(11, 6), title=''):


    '''
    
    Farbe #FFD700: r'$P_{11}$' (Gold)
    Farbe #00008B: r'$P_{12}$' (dark blue)
    Farbe #929591: r'$P_{13}$' (hellgrau))
    Farbe #069AF3: r'$P_{21}$' (hellblau)
    Farbe #F97306: r'$P_{22}$' (Orangerot)
    Farbe #929591: r'$P_{23}$' (hellgrau)
    Farbe #929591: r'$P_{31}$' (hellgrau)
    Farbe #929591: r'$P_{32}$' (hellgrau)
    Farbe #FF0000F: r'$P_{33}$' (rot)
    '''
    colors = ['#FFD700', '#00008B', '#929591', '#069AF3', '#F97306', '#929591', '#929591', '#929591', '#FF0000']
    labels = [r'$P_{11}$', r'$P_{12}$', r'$P_{13}$',
              r'$P_{21}$', r'$P_{22}$', r'$P_{23}$',
              r'$P_{31}$', r'$P_{32}$', r'$P_{33}$']

  
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=1000)

    # Linker Plot (uniaxial: S_ij  vs. F11)
    idx = 0 
    for i in range(3): 
        for j in range(3): 
            axes[0].scatter(F_true_uniax[::step, 0, 0], P_true_uniax[::step, i, j], color=colors[idx], label=labels[idx], alpha=1)
            axes[0].plot(F_true_uniax[::step, 0, 0], P_pred_uniax[::step, i, j], color=colors[idx])
            idx += 1

    axes[0].set_xlabel(r"$F_{11}$", fontsize=18)
    axes[0].set_ylabel(r"Uniaxial - $P_{ij}$", fontsize=18)
    #axes[0].legend(fontsize=12, loc="upper left", ncol=3)
  
    axes[0].legend(fontsize=12, loc="upper left", ncol=3, columnspacing=1)


    # Rechter Plot (shear: S_ij  vs. F12)
    idx = 0  
    for i in range(3): 
        for j in range(3):  
            axes[1].scatter(F_true_shear[::step, 0, 1], P_true_shear[::step, i, j], color=colors[idx], alpha=1)
            axes[1].plot(F_true_shear[::step, 0, 1], P_pred_shear[::step, i, j], color=colors[idx])
            idx += 1

    axes[1].set_xlabel(r"$F_{12}$", fontsize=18)
    axes[1].set_ylabel(r"Shear - $P_{ij}$", fontsize=18)


    for ax in axes:
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

 
    if title:
        fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()

# Hier für Test daten
def task5_2_paper_plot_test(F_test1, P_test1_true, P_test1_pred, F_test3, P_test3_true, P_test3_pred, step, figsize=(11, 6), title=''):

    '''
    
    Farbe #FFD700: r'$P_{11}$' (Gold)
    Farbe #00008B: r'$P_{12}$' (dark blue)
    Farbe #929591: r'$P_{13}$' (hellgrau))
    Farbe #069AF3: r'$P_{21}$' (hellblau)
    Farbe #F97306: r'$P_{22}$' (Orangerot)
    Farbe #929591: r'$P_{23}$' (hellgrau)
    Farbe #929591: r'$P_{31}$' (hellgrau)
    Farbe #929591: r'$P_{32}$' (hellgrau)
    Farbe #FF0000: r'$P_{33}$' (rot)
    '''
    colors = ['#FFD700', '#00008B', '#929591', '#069AF3', '#F97306', '#929591', '#929591', '#929591', '#FF0000']
    labels = [r'$P_{11}$', r'$P_{12}$', r'$P_{13}$',
              r'$P_{21}$', r'$P_{22}$', r'$P_{23}$',
              r'$P_{31}$', r'$P_{32}$', r'$P_{33}$']

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=1000)

    # Linker Plot (uniaxial: S_ij  vs. F11)
    idx = 0  # Für Farb- und Label-Zuweisung
    for i in range(3):  # Iteriere über Zeilen
        for j in range(3):  # Iteriere über Spalten
            axes[0].scatter(F_test1[::step, 0, 0],P_test1_true[::step, i, j], color=colors[idx], label=labels[idx], alpha=1)
            axes[0].plot(F_test1[::step, 0, 0], P_test1_pred[::step, i, j], color=colors[idx])
            idx += 1

    axes[0].set_xlabel(r"$F_{11}$", fontsize=18)
    axes[0].set_ylabel(r"biaxial test - $P_{ij}$", fontsize=18)
    #axes[0].legend(fontsize=12, loc="upper left", ncol=3)

    axes[0].legend(fontsize=12, loc="upper left", ncol=3, columnspacing=1)


    # Rechter Plot (shear: S_ij  vs. F12)
    idx = 0 
    for i in range(3): 
        for j in range(3):  
            axes[1].scatter(F_test3[::step, 0, 0], P_test3_true[::step, i, j], color=colors[idx], alpha=1)
            axes[1].plot(F_test3[::step, 0, 0], P_test3_pred[::step, i, j], color=colors[idx])
            idx += 1

    axes[1].set_xlabel(r"$F_{12}$", fontsize=18)
    axes[1].set_ylabel(r"mixed test - $P_{ij}$", fontsize=18)

    for ax in axes:
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Gesamten Titel hinzufügen (optional)
    if title:
        fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()

def task5_2_test_data(F_test1, P_test1_true,step, figsize=(11, 6), title=''):

  
    '''
    
    Farbe #FFD700: r'$P_{11}$' (Gold)
    Farbe #00008B: r'$P_{12}$' (dark blue)
    Farbe #929591: r'$P_{13}$' (hellgrau))
    Farbe #069AF3: r'$P_{21}$' (hellblau)
    Farbe #F97306: r'$P_{22}$' (Orangerot)
    Farbe #929591: r'$P_{23}$' (hellgrau)
    Farbe #929591: r'$P_{31}$' (hellgrau)
    Farbe #929591: r'$P_{32}$' (hellgrau)
    Farbe #FF0000: r'$P_{33}$' (rot)
    '''
    colors = ['#FFD700', '#00008B', '#929591', '#069AF3', '#F97306', '#929591', '#929591', '#929591', '#FF0000']
    labels = [r'$P_{11}$', r'$P_{12}$', r'$P_{13}$',
              r'$P_{21}$', r'$P_{22}$', r'$P_{23}$',
              r'$P_{31}$', r'$P_{32}$', r'$P_{33}$']

    # Initialisiere Subplots
    fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=1000)

    # Linker Plot (uniaxial: S_ij  vs. F11)
    idx = 0  # Für Farb- und Label-Zuweisung
    for i in range(3):  # Iteriere über Zeilen
        for j in range(3):  # Iteriere über Spalten
            axes.scatter(F_test1[::step, 0, 0],P_test1_true[::step, i, j], color=colors[idx], label=labels[idx], alpha=1)
            idx += 1

    axes.set_xlabel(r"$F_{11}$", fontsize=18)
    axes.set_ylabel(r"Shear-tension test case - $P_{ij}$", fontsize=18)
    #axes[0].legend(fontsize=12, loc="upper left", ncol=3)
    # Legende transponieren: Drei Spalten, eine Reihe für jede Komponente
    axes.legend(fontsize=12, loc="upper left", ncol=3, columnspacing=1)

    if title:
        fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    plt.show()




def check_objectivity_final(F_true, W_true, W_pred, W_pred_QF, step, figsize=(18, 12), title="", plot_over_F=None):
    import matplotlib.pyplot as plt

    if plot_over_F is None:
        plot_over_F = [(0, 0), (1, 1), (2, 2)]  # Standardmäßig Plots über F

    fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=100)

    # Schleife über alle Subplots
    for row in range(3):
        for col in range(3):
            # Daten vorbereiten
            F_values = [F_true[i][row, col] for i in range(len(F_true))]
            W_true_values = [W_true[i] for i in range(len(W_true))]
            W_pred_values = [W_pred[i] for i in range(len(W_pred))]
            W_pred_values_QF = [W_pred_QF[i] for i in range(len(W_pred_QF))]

            ax1 = axs[row, col]

            if (row, col) in plot_over_F:  # Plot über F
                ax1.scatter(F_values[::step], W_true_values[::step], color='red', s=40, marker='x', label='Energy - true')
                ax1.plot(F_values, W_pred_values, color='red', linewidth=4, label='W(F)')
                ax1.plot(F_values, W_pred_values_QF, color='blue', linewidth=4, linestyle='--', label='W(QF)')
            else:  # Plot über Index
                indices = list(range(len(F_true)))
                ax1.scatter(indices[::step], W_true_values[::step], color='red', s=40, marker='x', label='Energy - true')
                ax1.plot(indices, W_pred_values, color='red', linewidth=4, label='W(Index)')
                ax1.plot(indices, W_pred_values_QF, color='blue', linewidth=4, linestyle='--', label='W(QF Index)')

           
            ax1.set_ylabel(f"W", color='red', fontsize=18)
            ax1.tick_params(axis='y', labelcolor='red')

            ax1.set_title(f"F{row+1}{col+1} vs. W" if (row, col) in plot_over_F else f"Index vs. W", fontsize=18)
            ax1.legend(loc='upper left')


    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.show()






def OB(F_true_shear, P_pred_shear, P_shear_true, step=10, title=''):
   
    colors = ['#FFD700', '#00008B', '#929591',  # P_11, P_12, P_13
              '#069AF3', '#F97306', '#929591',  # P_21, P_22, P_23
              '#929591', '#929591', '#FF0000']  # P_31, P_32, P_33
    labels = [r'$P_{11}$', r'$P_{12}$', r'$P_{13}$',
              r'$P_{21}$', r'$P_{22}$', r'$P_{23}$',
              r'$P_{31}$', r'$P_{32}$', r'$P_{33}$']

    plt.figure(figsize=(12, 8), dpi=500)

    for i in range(3):
        for j in range(3):
            component_index = i * 3 + j  
            color = colors[component_index]
            label = labels[component_index]

            P_component = P_pred_shear[:, :, i, j]
           
            variances = np.var(P_component, axis=1)
            idx_low, idx_high = np.argsort(variances)[[0, -1]]

        
            F_values = F_true_shear[::step, 0, 1]  
            P_shear_true_values = P_shear_true[::step, i, j] 
            P_low = P_component[idx_low, ::step]
            P_high = P_component[idx_high, ::step]
            
            plt.plot(F_values, P_low, color=color, linewidth=2, label=f"{label}")
            plt.scatter(F_values, P_shear_true_values, color=color, s=20)
            plt.plot(F_values, P_high, color=color, linewidth=2)
            plt.fill_between(F_values, P_low, P_high, color=color, alpha=0.2)

    
    plt.xlabel(r"$F_{12}$")
    plt.ylabel(r"$P_{ij}$")
    plt.ylim(-10,20)
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.show()



