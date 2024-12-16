"""
Tutorial Machine Learning in Solid Mechanics (WiSe 24/25)
Task 2: Hyperelasticity

==================

Author: Loris Demuth
         
11/2024
"""

"""
import modules
"""
import tensorflow as tf
from tensorflow.keras import layers, constraints
import datetime

now = datetime.datetime.now

""" 
2.1) Neural network model - A naive approach
"""


class M_s(layers.Layer):
    def __init__(self):
        super(M_s, self).__init__()
        # defin hidden layers
        self.ls = [layers.Dense(64, activation="softplus")]
        self.ls += [layers.Dense(64, activation="softplus")]
        self.ls += [layers.Dense(64, activation="softplus")]
        # Wir möchten den gesamten Stress-Tensor (9 Einträge) vorhersagen, deshalb 9 outputs
        self.ls += [layers.Dense(9, activation="linear")]

    def call(self, x):
        for l in self.ls:
            x = l(x)
        return x


""" 
2.1) Build
"""

def main(**kwargs):
    # define input shape
    c = tf.keras.Input(shape=(6,))
    # define which (custom) layer the model should use
    p = M_s(**kwargs)(c)
    # connect input with output
    model = tf.keras.Model(inputs=[c], outputs=[p])
    # define optimizer and loss
    model.compile(optimizer="adam", loss="mse")
    return model
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
""" 
Calculation of the invariants for task 3
"""
def calc_invariants(F):
    G_ti = tf.constant([[4, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    C = tf.matmul(tf.transpose(F, perm=[0, 2, 1]), F)
    #C_inv    = tf.linalg.inv(C)
    C_inv = tf.linalg.inv(C + tf.eye(3) * 1e-6)  # Kleine Diagonalstörung hinzufügen
    I_1 = tf.linalg.trace(C)
    J = tf.linalg.det(F)
    I_3 = tf.linalg.det(C)  # (199,)
    I_3_exp = tf.expand_dims(tf.expand_dims(I_3, axis=-1), axis=-1)
    cof_C = I_3_exp * C_inv
    I_4 = tf.linalg.trace(tf.matmul(C, G_ti))
    I_5 = tf.linalg.trace(tf.matmul(cof_C, G_ti))
    W = 8 * I_1 + 10 * J**2 - 56 * tf.math.log(J) + 0.2 * (I_4**2 + I_5**2) - 44
    return tf.stack([I_1, J, -J, I_4, I_5], axis=1)  # shape(648,5)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

'''
Calculation ov invariants for task 5.2
'''
def structural_tensor():
    e1 = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
    e2 = tf.constant([0.0, 1.0, 0.0], dtype=tf.float32)
    e3 = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)

    Gcub_e1 = tf.tensordot(e1, e1, axes=0)
    Gcub_e2 = tf.tensordot(e2, e2, axes=0)
    Gcub_e3 = tf.tensordot(e3, e3, axes=0)

    Gcub = Gcub_e1 + Gcub_e2 + Gcub_e3
    return tf.einsum('ij,kl->ijkl', Gcub, Gcub)

def calc_I_task5(F):
    C = tf.matmul(tf.transpose(F, perm=[0, 2, 1]), F)

    I_1 = tf.linalg.trace(C)
  
    # Achtung hier müssen die shapes angepasst werden! 
    Cof_C = tf.linalg.det(C)[..., tf.newaxis, tf.newaxis] * tf.linalg.inv(C)
    I_2 = tf.linalg.trace(Cof_C)

    J = tf.linalg.det(F)

    Gcub = structural_tensor()
    I_7 = tf.einsum('bij,ijkl,bkl->b', C, Gcub, C)
    I_11 = tf.einsum('bij,ijkl,bkl->b', Cof_C, Gcub, Cof_C)
    
    return tf.stack([I_1, I_2, J, -J, I_7, I_11], axis=1)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
""" 
Task 3: Physics-augmented neural network: Invariant-based PANN
"""
# Als input in das neuronale Netz werden invarianten von F verwendet, siehe Aufgabenstellung.
# Der output des NNs ist das Hyperelastische Potential "W"(nur ein output), welches anschließend durch automatische Differenzierung mit GradientTape
# zu der Spannung P führt. In diesem Fall benötigen wir eine benutzerdefinierte Trainingsschleife, um W dynamisch zu differenzieren. "Model.compile" funktioniert nur,
# wenn P und W direkte outputs des NNs sind. Das Problem bei dieser Variante ist allerdings, dass der Zusammenhang zw. P und W nicht gelernt wird. Fehlende bzw.
# unzureichende physikalische Konsistenz.

class W_I_task3(layers.Layer):
    def __init__(self):
        super(W_I_task3, self).__init__()
        self.ls = [layers.Dense(16, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(16, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(1, activation="linear", kernel_constraint=constraints.NonNeg())]

    def call(self, invariants):
        for l in self.ls:
            invariants = l(invariants)
        # Output von letzter Schicht ist W
        W = invariants
        return W


# Initialisierung des Modells und dem Optimierer
inputs = tf.keras.Input(shape=(5,))
output = W_I_task3()(inputs)
model_W_I_task3 = tf.keras.Model(inputs=inputs, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)


# Benutzerdefinierte Trainingsschleife mit GradientTape


@tf.function  # optimiert TensorFlow-Funktionen, indem sie in einen Berechnungsgraphen kovertiert werden
def train_step_task3(F, W_true, P_true, sample_weight=None, train_on="both"):
    F = tf.cast(F, tf.float32)
    W_true = tf.cast(W_true, tf.float32)
    P_true = tf.cast(P_true, tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(F)  # F wird beobachtet, um später P zu berechenen (Link zw. F und W aufbauen)
        I = calc_invariants(F)  # Berechnung der Invarianten auf Basis des Deformationstensors
        W_pred = model_W_I_task3(I)  # Modellvorhersage für das hyperelastische Potential W

        P_pred = tape.gradient(W_pred, F)  # Gradient von W_pred bzgl. F => themodynamische Konsistenz

        # Umformung von W_true, um sicherzustellen, dass es die richtige Form hat
        W_true = tf.reshape(W_true, [-1, 1])

        # Falls sample_weight None ist, setze es auf Einsen
        if sample_weight is None:
            sample_weight = tf.ones_like(W_true, dtype=tf.float32)
        else:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            sample_weight = tf.broadcast_to(sample_weight, tf.shape(W_true))

        # sample_weight für P_pred anpassen
        sample_weight_P = tf.reshape(sample_weight, [-1, 1, 1])  # Auf [batch_size, 1, 1] erweitern
        sample_weight_P = tf.broadcast_to(sample_weight_P, tf.shape(P_pred))  # Auf [batch_size, 3, 3] erweitern

        # Kostenfunktionen
        loss_W = tf.reduce_mean(tf.square(W_pred - W_true) * sample_weight)
        loss_P = tf.reduce_mean(tf.square(P_pred - P_true) * sample_weight_P)

        # Definition der Kostenfunktion in abhängigkeit von "train_on"
        if train_on == "W":
            total_loss = loss_W
        elif train_on == "P":
            total_loss = loss_P
        elif train_on == "both":
            total_loss = loss_W + loss_P

        # Optimierung der Kostenfunktion
        grad_loss = tape.gradient(total_loss, model_W_I_task3.trainable_variables)
        optimizer.apply_gradients(zip(grad_loss, model_W_I_task3.trainable_variables))
        del tape
        return total_loss, W_pred, P_pred


# haupttrainingsschleife mit Batch-Verarbeitung
def train_model_W_I_task3(F_train, W_train, P_train, epochs, batch_size, sample_weight=None):

    # Anzahl der "batch-dimesion" entspricht der Anzahl der Messpunkte
    n_samples = F_train.shape[0]

    # Anzahl Batches (Ganzzahlig)
    n_batches = n_samples // batch_size

    history = []

    # Doppelte for schleife, um durch die trainingsdatan zu slicen mit batch_size und das für jede Epoch
    for epoch in range(epochs):
        # Initialiserung des Fehlers auf null
        epoch_loss = 0

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            batch_F = F_train[start_idx:end_idx]
            batch_W = W_train[start_idx:end_idx]
            batch_P = P_train[start_idx:end_idx]

            if sample_weight is not None:
                batch_sample_weight = sample_weight[start_idx:end_idx]
                batch_sample_weight = tf.convert_to_tensor(batch_sample_weight, dtype=tf.float32)
            else:
                batch_sample_weight = tf.ones_like(batch_W, dtype=tf.float32)

            loss, _, _ = train_step_task3(batch_F, batch_W, batch_P, batch_sample_weight)
            epoch_loss += loss

        epoch_loss = epoch_loss / n_batches
        history.append(epoch_loss)

        # Ausgabe des Fehlers z.B. alle 10 Epochen
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:4f}")

    return history, model_W_I_task3
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
""" 
Task 5.2
"""

class W_I_task5_2(layers.Layer):
    def __init__(self):
        super(W_I_task5_2, self).__init__()
        self.ls = [layers.Dense(32, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(32, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(32, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(1, activation="linear", kernel_constraint=constraints.NonNeg())]

    def call(self, invariants):
        for l in self.ls:
            invariants = l(invariants)
        # Output von letzter Schicht ist W
        W = invariants
        return W

# Initialisierung des Modells und dem Optimierer
inputs = tf.keras.Input(shape=(6,))
output = W_I_task5_2()(inputs)
model_W_I_task5_2 = tf.keras.Model(inputs=inputs, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)



@tf.function  # optimiert TensorFlow-Funktionen, indem sie in einen Berechnungsgraphen kovertiert werden
def train_step_task5_2(F, W_true, P_true, sample_weight=None, train_on="both"):
    F = tf.cast(F, tf.float32)
    W_true = tf.cast(W_true, tf.float32)
    P_true = tf.cast(P_true, tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(F)  # F wird beobachtet, um später P zu berechenen (Link zw. F und W aufbauen)
        I = calc_I_task5(F)  # Berechnung der Invarianten auf Basis des Deformationstensors
        W_pred = model_W_I_task5_2(I)  # Modellvorhersage für das hyperelastische Potential W

        P_pred = tape.gradient(W_pred, F)  # Gradient von W_pred bzgl. F => themodynamische Konsistenz

        # Umformung von W_true, um sicherzustellen, dass es die richtige Form hat
        W_true = tf.reshape(W_true, [-1, 1])

        # Falls sample_weight None ist, setze es auf Einsen
        if sample_weight is None:
            sample_weight = tf.ones_like(W_true, dtype=tf.float32)
        else:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            sample_weight = tf.broadcast_to(sample_weight, tf.shape(W_true))

        # sample_weight für P_pred anpassen
        sample_weight_P = tf.reshape(sample_weight, [-1, 1, 1])  # Auf [batch_size, 1, 1] erweitern
        sample_weight_P = tf.broadcast_to(sample_weight_P, tf.shape(P_pred))  # Auf [batch_size, 3, 3] erweitern

        # Kostenfunktion
        loss_W = tf.reduce_mean(tf.square(W_pred - W_true) * sample_weight)
        loss_P = tf.reduce_mean(tf.square(P_pred - P_true) * sample_weight_P)

        # Definition der Kostenfunktion in abhängigkeit von "train_on"
        if train_on == "W":
            total_loss = loss_W
        elif train_on == "P":
            total_loss = loss_P
        elif train_on == "both":
            total_loss = loss_W + loss_P

        # Optimierung der Kostenfunktion
        grad_loss = tape.gradient(total_loss, model_W_I_task5_2.trainable_variables)
        optimizer.apply_gradients(zip(grad_loss, model_W_I_task5_2.trainable_variables))
        del tape
        return total_loss, W_pred, P_pred


# haupttrainingsschleife mit Batch-Verarbeitung
def train_model_task5_2(F_train, W_train, P_train, epochs, batch_size, sample_weight=None):

    # Anzahl der "batch-dimesion" entspricht der Anzahl der Messpunkte
    n_samples = F_train.shape[0]

    # Anzahl Batches (Ganzzahlig)
    n_batches = n_samples // batch_size

    history = []

    # Doppelte for schleife, um durch die trainingsdatan zu slicen mit batch_size und das für jede Epoch
    for epoch in range(epochs):
        # Initialiserung des Fehlers auf null
        epoch_loss = 0

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            batch_F = F_train[start_idx:end_idx]
            batch_W = W_train[start_idx:end_idx]
            batch_P = P_train[start_idx:end_idx]

            if sample_weight is not None:
                batch_sample_weight = sample_weight[start_idx:end_idx]
                batch_sample_weight = tf.convert_to_tensor(batch_sample_weight, dtype=tf.float32)
            else:
                batch_sample_weight = tf.ones_like(batch_W, dtype=tf.float32)

            loss, _, _ = train_step_task5_2(batch_F, batch_W, batch_P, batch_sample_weight)
            epoch_loss += loss

        epoch_loss = epoch_loss / n_batches
        history.append(epoch_loss)

        # Ausgabe des Fehlers z.B. alle 10 Epochen
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:4f}")

    return history, model_W_I_task5_2
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#

""" 
Task 5.3: Deformation-gradient based neural network model
Um was gehts: Wir konstruieren nun ein NN, mit input (F, Cof_F, det F) und output W. Durch differenzieren erhalten P. Selber Ansatz wie Task 3. 
Allerdings wird OBJEKTIVÄT und MATERIALSYMMETRIE nicht mehr per Konstruktion erfüllt. 
Objektivität ist unabhängig von der Rotation: W(QF) = W(F) und P(QF) = QP(F) und mit diesem Modelansatz nicht erfüllt!!!
"""
def calc_input_W_F(F):
    det_F = tf.linalg.det(F)
    Cof_F = tf.linalg.det(F)[..., tf.newaxis, tf.newaxis] * tf.linalg.inv(F)

    # Flach umgeformte Matrizen (von 3x3 zu 9)
    F_flat = tf.reshape(F, (F.shape[0], 9))  # Shape: (batch_size, 9)
    det_F_expanded = tf.expand_dims(det_F, axis=1)
    Cof_F_flat = tf.reshape(Cof_F, (F.shape[0], 9))

    Input_W_F = tf.concat([F_flat, Cof_F_flat, det_F_expanded], axis=1)
    
    return Input_W_F



class W_F(layers.Layer):
    def __init__(self):
        super(W_F, self).__init__()
        self.ls = [layers.Dense(32, activation="softplus")]
        self.ls += [layers.Dense(32, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(32, activation="softplus", kernel_constraint=constraints.NonNeg())]
        self.ls += [layers.Dense(1, activation="linear", kernel_constraint=constraints.NonNeg())]

    def call(self, invariants):
        for l in self.ls:
            invariants = l(invariants)
        # Output von letzter Schicht ist W
        W = invariants
        return W
    

# Initialisierung des Modells und dem Optimierer
inputs = tf.keras.Input(shape=(19,))
output = W_F()(inputs)
model_W_F = tf.keras.Model(inputs=inputs, outputs=output)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

@tf.function  # optimiert TensorFlow-Funktionen, indem sie in einen Berechnungsgraphen kovertiert werden
def train_step_W_F(F_train, W_true, P_true, sample_weight=None, train_on="both"):
    F_train = tf.cast(F_train, tf.float32)
    W_true = tf.cast(W_true, tf.float32)
    P_true = tf.cast(P_true, tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(F_train)  # F wird beobachtet, um später P zu berechenen (Link zw. F und W aufbauen)
        Input_W_F = calc_input_W_F(F_train)  # Berechnung des inputs auf Basis des Deformationstensors mit Rotation
            
        W_pred = model_W_F(Input_W_F)  # Modellvorhersage für das hyperelastische Potential W

        P_pred = tape.gradient(W_pred, F_train)  

        # Umformung von W_true, um sicherzustellen, dass es die richtige Form hat
        W_true = tf.reshape(W_true, [-1, 1])

        # Falls sample_weight None ist, setze es auf Einsen
        if sample_weight is None:
            sample_weight = tf.ones_like(W_true, dtype=tf.float32)
        else:
            sample_weight = tf.reshape(sample_weight, [-1, 1])
            sample_weight = tf.broadcast_to(sample_weight, tf.shape(W_true))

        # sample_weight für P_pred anpassen
        sample_weight_P = tf.reshape(sample_weight, [-1, 1, 1])  # Auf [batch_size, 1, 1] erweitern
        sample_weight_P = tf.broadcast_to(sample_weight_P, tf.shape(P_pred))  # Auf [batch_size, 3, 3] erweitern

        # Kostenfunktion
        loss_W = tf.reduce_mean(tf.square(W_pred - W_true) * sample_weight)
        loss_P = tf.reduce_mean(tf.square(P_pred - P_true) * sample_weight_P)

        # Definition der Kostenfunktion in abhängigkeit von "train_on"
        if train_on == "W":
            total_loss = loss_W
        elif train_on == "P":
            total_loss = loss_P
        elif train_on == "both":
            total_loss = loss_W + loss_P

        # Optimierung der Kostenfunktion
        grad_loss = tape.gradient(total_loss, model_W_F.trainable_variables)
        optimizer.apply_gradients(zip(grad_loss, model_W_F.trainable_variables))
        del tape
        return total_loss, W_pred, P_pred
    

    # haupttrainingsschleife mit Batch-Verarbeitung
def train_model_W_F(F_train, W_train, P_train, epochs, batch_size, sample_weight=None):

    # Anzahl der "batch-dimesion" entspricht der Anzahl der Messpunkte
    n_samples = F_train.shape[0]

    # Anzahl Batches (Ganzzahlig)
    n_batches = n_samples // batch_size

    history = []

    # Doppelte for schleife, um durch die trainingsdatan zu slicen mit batch_size und das für jede Epoch
    for epoch in range(epochs):
        # Initialiserung des Fehlers auf null
        epoch_loss = 0

        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            batch_F = F_train[start_idx:end_idx]
            batch_W = W_train[start_idx:end_idx]
            batch_P = P_train[start_idx:end_idx]

            if sample_weight is not None:
                batch_sample_weight = sample_weight[start_idx:end_idx]
                batch_sample_weight = tf.convert_to_tensor(batch_sample_weight, dtype=tf.float32)
            else:
                batch_sample_weight = tf.ones_like(batch_W, dtype=tf.float32)

            loss, _, _ = train_step_W_F(batch_F, batch_W, batch_P,  batch_sample_weight)
            epoch_loss += loss

        epoch_loss = epoch_loss / n_batches
        history.append(epoch_loss)

        # Ausgabe des Fehlers z.B. alle 10 Epochen
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:4f}")

    return history, model_W_F
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------#