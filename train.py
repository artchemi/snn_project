import numpy as np

def stdp(delta_time: float, a_plus=10, a_minus=10, tau_pre=10, tau_post=10) -> float:
    """
    Возвращает изменение силы синапса dW

        Параметры:

            delta_time (float) : dT подсчитывается функцией delta_times_calc(lst_1: list, lst_2: list)

            a_plus (float) : параметр в уравнении, см. уравнение в блокноте

            a_minus (float) : см. уравнение в блокноте

            tau_pre (float) : 

            tau_post (float) :

        Возвращаемое значение:

            dw (float) : измение веса 
    """
    if delta_time > 0:
        dw = 1.0 * a_plus * np.exp(-1.0 * delta_time / tau_pre)

    elif delta_time < 0:
        dw = -1.0 * a_minus * np.exp(delta_time / tau_post)

    else:
        dw = 0

    return dw


def delta_times_calc(lst_1: list, lst_2: list) -> list:
    """
    Возвращает разницу во времени между двумя сериями спайков

        Параметры:

            lst_1 (list) : серия спайков постсинаптического нейрона
            
            lst_2 (list) : серия спайков пресинаптического нейрона

        Возвращаемое значение:

            delta_times (list) : массив dT для двух серий

    """
    delta_times = []
    for i in range(len(lst_1)):
        for j in range(len(lst_2)):
            delta_times.append(lst_1[i] - lst_2[j])

    return delta_times


def update_synapse(spike_times_1, spike_times_2, synapse):
    """
    Возвращает обновленную матрицу силы синапсов

        Параметры:

            spike_times_1 (list) : серия спайков постсинаптического нейрона

            spike_times_1 (list) : серия спайков пресинаптического нейрона

            synapse (np.array) : старая матрица весов

        Возвращаемое значение:

            synapse_new (np.array) : новая матрица весов
    """

    synapse_new = synapse

    for i in range(len(spike_times_1)):
        if spike_times_1[i] == []:
            continue
        
        for j in range(len(spike_times_2)):
            if spike_times_2[j] == []:
                continue
            
            delta_times = delta_times_calc(spike_times_1[i], spike_times_2[j])

            dw = sum(list(map(stdp, delta_times)))
            synapse_new[j, i] = synapse[j, i] + dw

    return synapse_new
