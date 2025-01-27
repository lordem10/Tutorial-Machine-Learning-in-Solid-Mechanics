"""
Tutorial Machine Learning in Solid Mechanics (WiSe 23/24)
Task 3: Viscoelasticity
==================
Authors: Loris Demuth
         
01/2025
"""


# %%   
"""
Task 1: Simple RNN
"""


import tensorflow as tf
from tensorflow.keras import layers, constraints

    
class MLP(layers.Layer):
    def __init__(self, units, activation_funcs):
        super().__init__()
        self.ls = [layers.Dense(units[i], activation=activation_funcs[i]) for i in range(len(units))]

    def call(self, x):
        for layer in self.ls:
            x = layer(x)
        return x
    

class RNNCell(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1

        self.MLP = MLP(units=[16, 2], activation_funcs=['softplus', 'softplus'])
        
    def call(self, inputs, states):
        
        #   states are the internal variables
        #   n: current time step, N: old time step
                
        eps_n = inputs[:, 0:1]
        hs = inputs[:, 1:2]

        #eps_n = inputs[0]
        #hs = inputs[1]
        
        #   gamma: history variable
        gamma_N = states[0]
        
        #   x contains the current strain, the current time step size, and the 
        #   history variable from the previous time step
        x = tf.concat([eps_n, hs, gamma_N], axis = 1)
                
        #   x gets passed to a FFNN which yields the current stress and history
        #   variable
    
        for l in self.ls:
            x = l(x)
         
        sig_n = x[:, 0:1]
        gamma_n = x[:, 1:2]
            
                
        return sig_n, [gamma_n]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        #   define initial values of the internal variables      
        return [tf.zeros([batch_size, 1])]


def main(**kwargs):
    # define inputs
    eps = tf.keras.Input(shape=[None, 1],name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')

    # concatenate inputs
    concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([eps, hs])

    # define RNN cell
    cell = RNNCell()
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False) # return_state=True => hier wird sig_n zurückgegeben, return_state=False => hier wird gamma nicht zurückgeben, aber für nächste Berechnung verwendet
    sigs = layer1(concatenated_inputs)

    model = tf.keras.Model([eps, hs], [sigs])
    model.compile('adam', 'mse')
    return model


#---------------------------------------------------------------------------------------#

"""
Task 2.1: Maxwell model (analytical solution)
"""

class MaxwellModelAnalytic(layers.Layer):
    def __init__(self, constants, **kwargs):
        super(MaxwellModelAnalytic, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1

        # Konstanten
        self.eta = constants['eta']
        self.E = constants['E']
        self.E_inf = constants['E_inf']

    def call(self, inputs, states):
        eps_N = inputs[:, 0:1]
        hs_N = inputs[:, 1:2]
        gamma_N = states[0]

        # Berechnung der Spannung nach Gl. 6
        sigma_N = self.E_inf * eps_N + self.E * (eps_N - gamma_N)

        # Berechnung der internen Variablen nach Gl. 15 mit dem expliziten Euler Verfahren
        gamma_n = gamma_N + hs_N * (self.E / self.eta) * (eps_N - gamma_N)

        return sigma_N, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #   define initial values of the internal variables      
        return [tf.zeros([batch_size, 1])]

def main_MaxwellModel(constants, **kwargs):
    # define inputs
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')

    # concatenate inputs
    concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([eps, hs])

    # define RNN cell
    cell = MaxwellModelAnalytic(constants)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False) 
    sigs = layer1(concatenated_inputs)
    model_maxwell = tf.keras.Model([eps, hs], [sigs])
    model_maxwell.compile('adam', 'mse')
    return model_maxwell


'''
Task 2.2: Maxwell model with FFNN for general evolution equation, stress = gradient of energy
'''
'''
class MaxwellModellFFNN(layers.Layer):
    def __init__(self, constants, **kwargs):
        super(MaxwellModellFFNN, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1

        self.E = constants['E']
        self.eta = constants['eta']
        self.E_inf = constants['E_inf']

        self.ls = [layers.Dense(32, activation = 'softplus')]
        self.ls += [layers.Dense(1)]

    def call(self, inputs, states):
        eps_N = inputs[:, 0:1]
        hs_N = inputs[:, 1:2]
        gamma_N = states[0]

        # Berechnung der Spannung nach Gl. 6
        sigma_N = self.E_inf * eps_N + self.E * (eps_N - gamma_N)

        x = tf.concat([eps_N, gamma_N], axis = 1)
        
        # Die Funktion f(eps_N, gamma_N) wird durch ein FFNN approximiert, siehe Gl. 29
        for l in self.ls:
            x = l(x)
        
        gamma_n = gamma_N + hs_N * x * (eps_N - gamma_N)

        return sigma_N, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, 1])]
    
def main_MaxwellModel_FFNN(constants, **kwargs):
    # Inputs definieren
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')

    concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([eps, hs])

    cell = MaxwellModellFFNN(constants)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1(concatenated_inputs)
    model_maxwellFFNN = tf.keras.Model([eps, hs], [sigs])
    model_maxwellFFNN.compile('adam', 'mse')

    return model_maxwellFFNN
'''

'''
Task 2.2: Maxwell model with FFNN for general evolution equation, stress = gradient of energy with respect to strain
'''

class MaxwellModellFFNN(layers.Layer):
    def __init__(self, constants, **kwargs):
        super(MaxwellModellFFNN, self).__init__(**kwargs)
        self.state_size = 1
        self.output_size = 1

        self.E = constants['E']
        self.eta = constants['eta']
        self.E_inf = constants['E_inf']

        self.MLP = MLP(units=[16, 1], activation_funcs=['softplus', 'softplus']) # Softplus in der output schicht, damit Dissipation ungleichung größer gleich 0 ist

    def call(self, inputs, states):
        eps_N = inputs[:, 0:1]
        hs_N = inputs[:, 1:2]
        gamma_N = states[0]

        with tf.GradientTape() as g:
            g.watch(eps_N)
            energy = 0.5 * self.E_inf * eps_N**2 + 0.5 * self.E * (eps_N - gamma_N)**2

        sig_N = g.gradient(energy, eps_N)
        
    
        x = tf.concat([eps_N, gamma_N], axis = 1)

        f = self.MLP(x)

        gamma_n = gamma_N + hs_N * f * (eps_N - gamma_N)
 
        return sig_N, [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, 1])]
    

def main_MaxwellModel_FFNN(constants, **kwargs):
    # Inputs definieren
    eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    hs = tf.keras.Input(shape=[None, 1], name='input_hs')

    #eps = tf.keras.Input(shape=(None, 1), name='input_eps')
    #hs = tf.keras.Input(shape=(None, 1), name='input_hs')

    concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([eps, hs])

    cell = MaxwellModellFFNN(constants)
    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1(concatenated_inputs)
    model_maxwellFFNN = tf.keras.Model([eps, hs], [sigs])
    model_maxwellFFNN.compile('adam', 'mse')

    return model_maxwellFFNN





#---------------------------------------------------------------------------------------#
'''
Task 3 : Generalized standard materials (GSM)
'''
class GSMModell(layers.Layer):
    def __init__(self, constant, **kwargs):
        super(GSMModell, self).__init__(*kwargs)
        self.state_size = 1
        self.output_size = 1
        self.eta = constant['eta']

        self.MLP = MLP(units=[16, 1], activation_funcs=['softplus', None])     

    def call(self, inputs, states):
        eps_N = inputs[:, 0:1]
        hs_N = inputs[:, 1:2]
        gamma_N = states[0]

        # Bei dem GSM Modell soll die Energie mithilfe eines FFNN approximiert werden.
        # e(eps, gamma)
        x = tf.concat([eps_N, gamma_N], axis = 1)

        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            e = self.MLP(x) # Vorhersage des FFNN ist die Energy e(eps, gamma)

        de_deps_de_dgamma = g.gradient(e, x)
        del g

        #sig_N = de_deps_de_dgamma[:, 0, tf.newaxis]
        sig_N = de_deps_de_dgamma[:, 0:1]
        #gamma_n = gamma_N - hs_N * (1/self.eta) * de_deps_de_dgamma[:, 1, tf.newaxis]
        gamma_n = gamma_N - hs_N * self.eta**(-1) * de_deps_de_dgamma[:, 1:2]
        #gamma_n = gamma_N - hs_N * (1/self.eta) * de_dgamma
                
        return sig_N , [gamma_n]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([batch_size, 1])]
    
def main_GSM(constant, **kwargs):
    # Inputs definieren
    #eps = tf.keras.Input(shape=[None, 1], name='input_eps')
    #hs = tf.keras.Input(shape=[None, 1], name='input_hs')

    eps = tf.keras.Input(shape=(None, 1))
    hs = tf.keras.Input(shape=(None, 1))

    #concatenated_inputs = tf.keras.layers.Concatenate(axis=-1)([eps, hs])
    concatenated_inputs = tf.keras.layers.Concatenate(axis=2)([eps, hs])
    cell = GSMModell(constant, **kwargs)

    layer1 = layers.RNN(cell, return_sequences=True, return_state=False)
    sigs = layer1(concatenated_inputs)
    model_GSM = tf.keras.Model([eps, hs], [sigs])
    model_GSM.compile('adam', 'mse')

    return model_GSM



# %%
