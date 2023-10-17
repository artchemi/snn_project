import numpy as np
import matplotlib.pyplot as plt
from snntorch import spikegen
import torch


def spike_gen(time, dt):
    steps = np.ceil(time / dt)
    I = [0 if 200 / dt <= i <= 300 / dt else 10 for i in np.arange(steps)]
    return np.array(I)


def spike_plot(membrane_lst, u_lst, spike_lst, thrs):
    plt.plot(np.arange(len(membrane_lst)), membrane_lst)
    plt.plot(np.arange(len(spike_lst)), u_lst)
    plt.plot(np.arange(len(spike_lst)), spike_lst, color='red')
    plt.plot([0, len(spike_lst)], [thrs, thrs], color='black', alpha=0.1)
    plt.xlabel('Time')
    plt.xlim(left=0, right=len(membrane_lst))
    plt.ylabel('Membrane potential, mV')
    plt.title('Membrane potential from a single neuron')
    # plt.axvspan(xmin=0, xmax=200, color='red', alpha=0.01)
    plt.show()


def image_to_spike(image, num_steps):
    """Конвертирует изображение в спайки, rate coding"""
    image_dist = np.divide(image, 255.0)

    spike_lst = []

    for n in range(len(image_dist)):
        for m in range(len(image_dist[n])):
            pixel = torch.Tensor([image_dist[n, m]])
            spike = spikegen.rate(pixel, num_steps=num_steps).tolist()

            spike_temp = []
            for s in spike:
                spike_temp.append(s[0] * 10)
            
            spike_lst.append(spike_temp)

    return spike_lst


def bar_plot_spike(output):
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    counts = []

    for neuron_output in output:
        counts.append(neuron_output.tolist().count(1))

    fig, ax = plt.subplots()

    ax.bar(labels, counts)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Label number')

    plt.show()
    

def mem_label_plot():
    """График мембранного потенциала для отображения по всем лейблам"""
    pass