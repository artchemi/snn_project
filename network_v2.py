from neuron_models.IZHI import IZHI
from utils import *
import numpy as np


class SpikingNetwork:
    """
    3-слойная импульсная нейронная сеть
    
        Параметры
        ---------
        input : int
            Количество нейронов во входном слое
        hidden : int
            Количество нейронов в скрытом слое
        output : int
            Количество нейронов в выходном слое
        dt : any
            dT
        intra_connect : bool
            Настройка топологии в скрытом слое

        Методы
        ------
        feedforward(spike_input):
            Прямой проход спайков по сети (переделать в будущем)

    """

    def __init__(self, input, hidden, output, dt, intra_connect=False):
        self.m = input
        self.k = hidden
        self.n = output

        self.spike_times = []

        self.method = 0

        self.intra = intra_connect

        self.dt = dt

        self.layer1 = []
        self.layer2 = []
        self.layer3 = []

        for i in range(self.m):
            self.layer1.append(IZHI())

        for i in range(self.k):
            self.layer2.append(IZHI())

        for i in range(self.n):
            self.layer3.append(IZHI())

        # self.synapse1 = np.random.randint(1, 5, size=(self.k, self.m))
        # self.synapse2 = np.random.randint(1, 5, size=(self.n, self.k))

        self.synapse1 = np.random.uniform(1, 2, size=(self.k, self.m))
        self.synapse2 = np.random.uniform(1, 2, size=(self.n, self.k))

        # self.synapse1 = np.random.rand(self.k, self.m)
        # self.synapse2 = np.random.rand(self.n, self.k)

        self.sum_synapse = np.sum(self.synapse1) + np.sum(self.synapse2)  # Сумма весов должна быть const

    def feedforward(self, spike_input):
        """
        Прямой проход спайков по сети (переделать)

        """
        assert len(spike_input) == self.m, 'Размерность входного массива не совпадает с числом нейронов'

        time = len(spike_input[0])

        membrane_network = []

        # Первый слой
        out_l1 = []
        membrane_l1 = []
        spike_times_l1 = []

        for i in range(self.m):
            neuron = self.layer1[i]
            spike_ser = spike_input[i]
            spike_times_temp = []

            spike_out_temp = np.zeros(len(spike_ser))
            membrane_temp = np.zeros(len(spike_ser))

            for j in range(len(spike_ser)):
                neuron.step(self.dt, spike_ser[j], method=self.method)
                membrane_temp[j] = neuron.v

                if neuron.v >= neuron.thrs:
                    spike_out_temp[j] = 1
                    spike_times_temp.append(j)

            spike_times_l1.append(spike_times_temp)
            membrane_l1.append(membrane_temp)
            out_l1.append(spike_out_temp)

        membrane_network.append(membrane_l1)
        self.spike_times.append(spike_times_l1)

        # Второй слой
        membrane_l2 = []
        out_l2 = []
        spike_times_l2 = []

        for i in range(self.k):
            neuron = self.layer2[i]
            spike_times_temp = []

            spike_out_temp = np.zeros(time)
            membrane_temp = np.zeros(time)

            for j in range(time):
                spike = 0

                for k in range(len(out_l1)):
                    spike += out_l1[k][j] * self.synapse1[i, k]
                
                neuron.step(self.dt, spike, method=self.method)
                membrane_temp[j] = neuron.v

                if neuron.v >= neuron.thrs:
                    spike_out_temp[j] = 1
                    spike_times_temp.append(j)
            
            spike_times_l2.append(spike_times_temp)
            membrane_l2.append(membrane_temp)
            out_l2.append(spike_out_temp)

        membrane_network.append(membrane_l2)
        self.spike_times.append(spike_times_l2)

        # Третий слой
        membrane_l3 = []
        out_l3 = []
        spike_times_l3 = [] 

        for i in range(self.n):
            neuron = self.layer3[i]
            spike_times_temp = []

            spike_out_temp = np.zeros(time)
            membrane_temp = np.zeros(time)

            for j in range(time):
                spike = 0

                for k in range(len(out_l2)):
                    spike += out_l2[k][j] * self.synapse2[i, k]
                
                neuron.step(self.dt, spike, method=self.method)
                membrane_temp[j] = neuron.v

                if neuron.v >= neuron.thrs:
                    spike_out_temp[j] = 1
                    spike_times_temp.append(j)
            
            spike_times_l3.append(spike_times_temp)
            membrane_l3.append(membrane_temp)
            out_l3.append(spike_out_temp)

        membrane_network.append(membrane_l3)
        self.spike_times.append(spike_times_l3)

        return [out_l1, out_l2, out_l3], membrane_network
    
    def update_synapse(self, spt_layer_1, spt_layer_2):
        synapse_matrix = []
        for spike_time_pre in spt_layer_1:
            synapse_temp = []
            for spike_time_post in spt_layer_2:
                # Нужен первый спайк
                post_len = len(spike_time_post)
                delta_t_arr = np.array(spike_time_post) - np.array(spike_time_pre[0:post_len])
                delta_t_mean = np.mean(delta_t_arr)

                if delta_t_mean >= 0:
                    delta_w = 0.1 * np.exp(-delta_t_mean/20)

                else:
                    # Штраф
                    delta_w = -0.1 * np.exp(delta_t_mean/20)

                synapse_temp.append(delta_w)
            synapse_matrix.append(synapse_temp)
        
        return np.array(synapse_matrix)
