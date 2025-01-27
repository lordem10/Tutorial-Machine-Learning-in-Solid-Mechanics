import numpy as np
import tensorflow as tf
'''
Task 1, 2, 3
'''

# --------------------Calibration data----------------------------------------------------------------------------------
biaxial_c  = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/calibration/biaxial.txt")
uniaxial_c = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/calibration/uniaxial.txt")
shear_c    = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/calibration/pure_shear.txt")
# --------------------Test data-----------------------------------------------------------------------------------------
biaxial_test = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/test/biax_test.txt")
mixed_test   = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/test/mixed_test.txt")
#-----------------------------------------------------------------------------------------------------------------------


def data_tf(data):
    F_list = [] 
    P_list = [] 
    W_list = []
    
    for i in range(len(data)):
        F_i = np.array(data[i][0:9]).reshape((3,3))
        P_i = np.array(data[i][9:18]).reshape((3,3))
        W_i = data[i][18]
        
        # Füge die Daten zur Liste hinzu
        F_list.append(F_i)
        P_list.append(P_i)
        W_list.append(W_i)
    
    # Konvertiere die Listen in Tensoren
    F_tensor = tf.convert_to_tensor(F_list, dtype=tf.float32) # shape (199, 3, 3)
    P_tensor = tf.convert_to_tensor(P_list, dtype=tf.float32)
    W_tensor = tf.convert_to_tensor(W_list, dtype=tf.float32)
    
    return {'F': F_tensor, 'P': P_tensor, 'W': W_tensor}

def dict_tf():
    
    # Erstelle dicts
    D_dict_biaxial_c_tf = data_tf(biaxial_c)
    D_dict_uniaxial_c_tf= data_tf(uniaxial_c)
    D_dict_shear_c_tf = data_tf(shear_c)
    D_dict_biaxial_test_tf = data_tf(biaxial_test)
    D_dict_mixed_test_tf = data_tf(mixed_test)
    
    return D_dict_uniaxial_c_tf, D_dict_biaxial_c_tf, D_dict_shear_c_tf, D_dict_biaxial_test_tf, D_dict_mixed_test_tf

'''
Task 5
'''
path = "/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/BCC_data"
# --------------------Calibration data----------------------------------------------------------------------------------
bcc_uniax_c = np.loadtxt(path + '/BCC_uniaxial.txt')
bcc_biax_c  = np.loadtxt(path + '/BCC_biaxial.txt')
bcc_shear_c    = np.loadtxt(path + '/BCC_shear.txt')
bcc_planar_c    = np.loadtxt(path + '/BCC_planar.txt')
bcc_volumetric_c    = np.loadtxt(path + '/BCC_volumetric.txt')
# --------------------Test data-----------------------------------------------------------------------------------------
bcc_test1 = np.loadtxt(path + '/BCC_test1.txt')
bcc_test2 = np.loadtxt(path + '/BCC_test2.txt')
bcc_test3 = np.loadtxt(path + '/BCC_test3.txt')
#-----------------------------------------------------------------------------------------------------------------------

def creat_dict(data):
    F_list = [] 
    P_list = [] 
    W_list = []
    error_list = []
    
    for i in range(len(data)):
        F_i = np.array(data[i][0:9]).reshape((3,3))
        P_i = np.array(data[i][9:18]).reshape((3,3))
        W_i = data[i][18]
        error_i = data[i][19]
        
        # Füge die Daten zur Liste hinzu
        F_list.append(F_i)
        P_list.append(P_i)
        W_list.append(W_i)
        error_list.append(error_i)
    
    # Konvertiere die Listen in Tensoren
    F_tensor = tf.convert_to_tensor(F_list, dtype=tf.float32) # shape (199, 3, 3)
    P_tensor = tf.convert_to_tensor(P_list, dtype=tf.float32)
    W_tensor = tf.convert_to_tensor(W_list, dtype=tf.float32)
    error_tensor = tf.convert_to_tensor(error_list, dtype=tf.float32)
    
    return {'F': F_tensor, 'P': P_tensor, 'W': W_tensor, 'error': error_tensor}

def dicts():
    
    # Erstelle dicts
    dict_bcc_uniaxial_c = creat_dict(bcc_uniax_c)
    dict_bcc_biaxial_c = creat_dict(bcc_biax_c)
    dict_bcc_shear_c = creat_dict(bcc_shear_c)
    dict_bcc_planar_c = creat_dict(bcc_planar_c)
    dict_bcc_volumetric_c = creat_dict(bcc_volumetric_c)

    dict_bcc_test1 = creat_dict(bcc_test1)
    dict_bcc_test2 = creat_dict(bcc_test2)
    dict_bcc_test3 = creat_dict(bcc_test3)
   
    return dict_bcc_uniaxial_c, dict_bcc_biaxial_c,  dict_bcc_shear_c, dict_bcc_planar_c, dict_bcc_volumetric_c, dict_bcc_test1, dict_bcc_test2, dict_bcc_test3 






def cauchyGreen(C_list):
# Als input nutzen wir den right cauchy green tensor "C". 
# Aufgrund von Symmetrie, reicht es aber aus nur die 6 voneinander unabhängigen Einträge (C11, C22, C33, C12, C13, C23) zu verwenden.
# Im Folgenden werden die Daten vorbereitet

# Listen für uniaxiale und biaxiale Tensoren
    C11_uniax = []
    C22_uniax = []
    C33_uniax = []
    C12_uniax = []
    C13_uniax = []
    C23_uniax = []

    C11_biax = []
    C22_biax = []
    C33_biax = []
    C12_biax = []
    C13_biax = []
    C23_biax = []

    C11_shear = []
    C22_shear = []
    C33_shear = []
    C12_shear = []
    C13_shear = []
    C23_shear = []

    C11_biax_test = []
    C22_biax_test = []
    C33_biax_test = []
    C12_biax_test = []
    C13_biax_test = []
    C23_biax_test = []

    C11_mix_test = []
    C22_mix_test = []
    C33_mix_test = []
    C12_mix_test = []
    C13_mix_test = []
    C23_mix_test = []



    for i in range(5):
        if i == 0:  # Uniaxiale Belastung
            C11_uniax.append(C_list[i][:, 0, 0])
            C22_uniax.append(C_list[i][:, 1, 1])
            C33_uniax.append(C_list[i][:, 2, 2])
            C12_uniax.append(C_list[i][:, 0, 1])
            C13_uniax.append(C_list[i][:, 0, 2])
            C23_uniax.append(C_list[i][:, 1, 2])

            # Stapeln und Transponieren für uniaxiale Belastung
            C_uniax = tf.squeeze(tf.stack([C11_uniax, C22_uniax, C33_uniax, C12_uniax, C13_uniax, C23_uniax], axis=1), axis=0)
            C_uniax = tf.transpose(C_uniax)  # Ergebnis: (199, 6)

        elif i == 1:  # Biaxiale Belastung
            C11_biax.append(C_list[i][:, 0, 0])
            C22_biax.append(C_list[i][:, 1, 1])
            C33_biax.append(C_list[i][:, 2, 2])
            C12_biax.append(C_list[i][:, 0, 1])
            C13_biax.append(C_list[i][:, 0, 2])
            C23_biax.append(C_list[i][:, 1, 2])

            # Stapeln und Transponieren für biaxiale Belastung
            C_biax = tf.squeeze(tf.stack([C11_biax, C22_biax, C33_biax, C12_biax, C13_biax, C23_biax], axis=1), axis=0)
            C_biax = tf.transpose(C_biax)  # Ergebnis: (199, 6)

        elif i == 2:  # Scherbelastung
            C11_shear.append(C_list[i][:, 0, 0])
            C22_shear.append(C_list[i][:, 1, 1])
            C33_shear.append(C_list[i][:, 2, 2])
            C12_shear.append(C_list[i][:, 0, 1])
            C13_shear.append(C_list[i][:, 0, 2])
            C23_shear.append(C_list[i][:, 1, 2])

            # Stapeln und Transponieren für Scherbelastung
            C_shear = tf.squeeze(tf.stack([C11_shear, C22_shear, C33_shear, C12_shear, C13_shear, C23_shear], axis=1), axis=0)
            C_shear = tf.transpose(C_shear)  # Ergebnis: (250, 6)

        elif i == 3:  # Testdaten für Biaxiale
            C11_biax_test.append(C_list[i][:, 0, 0])
            C22_biax_test.append(C_list[i][:, 1, 1])
            C33_biax_test.append(C_list[i][:, 2, 2])
            C12_biax_test.append(C_list[i][:, 0, 1])
            C13_biax_test.append(C_list[i][:, 0, 2])
            C23_biax_test.append(C_list[i][:, 1, 2])

            # Stapeln und Transponieren für Scherbelastung
            C_biax_t = tf.squeeze(tf.stack([C11_biax_test, C22_biax_test, C33_biax_test, C12_biax_test, C13_biax_test, C23_biax_test], axis=1), axis=0)
            C_biax_test = tf.transpose(C_biax_t)  # Ergebnis: (99, 6)
        else:
            C11_mix_test.append(C_list[i][:, 0, 0])
            C22_mix_test.append(C_list[i][:, 1, 1])
            C33_mix_test.append(C_list[i][:, 2, 2])
            C12_mix_test.append(C_list[i][:, 0, 1])
            C13_mix_test.append(C_list[i][:, 0, 2])
            C23_mix_test.append(C_list[i][:, 1, 2])

            # Stapeln und Transponieren für Scherbelastung
            C_mix_t = tf.squeeze(tf.stack([C11_mix_test, C22_mix_test, C33_mix_test, C12_mix_test, C13_mix_test, C23_mix_test], axis=1), axis=0)
            C_mix_test = tf.transpose(C_mix_t)  # Ergebnis: (100, 6)

    return C_uniax, C_biax, C_shear, C_biax_test, C_mix_test