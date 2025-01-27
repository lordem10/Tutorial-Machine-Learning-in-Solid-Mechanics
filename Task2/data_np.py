import numpy as np

# --------------------Calibration data----------------------------------------------------------------------------------
biaxial_c  = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/calibration/biaxial.txt")
uniaxial_c = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/calibration/uniaxial.txt")
shear_c    = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/calibration/pure_shear.txt")
# --------------------Test data-----------------------------------------------------------------------------------------
biaxial_test = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/test/biax_test.txt")
mixed_test   = np.loadtxt("/Users/ldemuth/Documents/Tutorial_ML_SolidMechanics/Task2/Task_11/test/mixed_test.txt")
#-----------------------------------------------------------------------------------------------------------------------

def load_data_file(data):
    D_dict = {}
    for i in range(len(data)):
        F_i = np.array(data[i][0:9]).reshape((3,3))
        P_i = np.array(data[i][9:18]).reshape((3,3))
        W_i = data[i][18]
        D_dict[i] = {'F': F_i, 'P': P_i, 'W': W_i}
    return D_dict

def loaddata():
    D_dict_biaxial_c = load_data_file(biaxial_c)
    D_dict_uniaxial_c = load_data_file(uniaxial_c)
    D_dict_shear_c = load_data_file(shear_c)
    D_dict_biaxial_test = load_data_file(biaxial_test)
    D_dict_mixed_test = load_data_file(mixed_test)
    
    return D_dict_biaxial_c, D_dict_uniaxial_c, D_dict_shear_c, D_dict_biaxial_test, D_dict_mixed_test
