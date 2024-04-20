# Bibliotecas

import numpy as np
import gc                          
from keras import backend as K     
import os
import tensorflow as tf
import pathlib

from keras.layers import Add, BatchNormalization, Conv1D, Dense, Dropout,\
                         Flatten, Input, MaxPooling1D, ReLU 

import plot_utils as putils

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def reset_keras():
    '''
    Reseta a sessão do keras e limpa a memória da GPU.
    '''
    
    # Reseta a sessão do Keras
    K.clear_session()
    
    # Limpa a memória da GPU
    try:
        del model, loaded_model, history  
    except NameError:
        pass
    
    gc.collect()
 
    return


def set_gpu(gpu_index):
    '''
    Define a GPU a ser usada pelo TensorFlow.
    
    parametros:
        gpu_index: Índice da GPU a ser usada (índice baseado em 0)
    '''
    # Certifica de que a ordem da GPU siga a ordem PCI_BUS_ID
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID' 
    
    # Lista de GPUs disponíveis
    physical_devices = tf.config.list_physical_devices('GPU')
    
    # Certifica de que o índice GPU esteja dentro do intervalo válido
    if gpu_index < 0 or gpu_index >= len(physical_devices):
        print('Index GPU inválido.')
        return False
    
    try:
        # Define os dispositivos GPU visíveis
        tf.config.set_visible_devices(physical_devices[gpu_index:gpu_index + 1], 'GPU')

        # Valida se apenas a GPU selecionada está definida como um dispositivo lógico
        assert len(tf.config.list_logical_devices('GPU')) == 1
        
        print(f'GPU {gpu_index} foi definido como o dispositivo visível.')
        return True
    except Exception as e:
        print(f'Ocorreu um erro ao configurar a GPU: {e}')
        return False


def get_model_memory_usage(batch_size, model):
    """
    Calcula o uso de memória de um modelo Keras.

    Argumentos:
     batch_size (int): tamanho do lote que está usando ou planeja usar.
     model (keras.Model ou tensorflow.keras.Model): modelo para calcular o uso de memória.

    Retorna:
     tupla: tupla contendo o uso de memória em gigabytes inteiros e o uso restante em megabytes.
    """

    import numpy as np
    # Tenta importar primeiro do back-end tensorflow.keras, pois é mais padrão
    try:
        from keras import backend as K
    except ImportError:
        from keras import backend as K

    # Inicializa contadores para memória de formas e memória de modelo aninhado
    shapes_mem_count = 0
    internal_model_mem_count = 0
    # Itera sobre camadas no modelo
    for l in model.layers:
        layer_type = l.__class__.__name__
        # Se a camada for um modelo em si, calcula recursivamente seu uso de memória
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        else:
            # Calcula o uso de memória do formato de saída da camada
            single_layer_mem = 1
            out_shape = l.output_shape
            # Se a forma de saída for uma lista, considera apenas a primeira forma
            if type(out_shape) is list:
                out_shape = out_shape[0]
            # Multiplica todas as dimensões para obter o número total de unidades
            for s in out_shape:
                if s is not None:
                    single_layer_mem *= s
            shapes_mem_count += single_layer_mem

    # Calcula o número total de parâmetros treináveis e não treináveis
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Define o tamanho do número com base no tipo float usado pelo backend Keras
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    # Calcula o uso total de memória
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    total_memory += internal_model_mem_count

    # Converte de bytes para GB
    gbytes = total_memory / (1024 ** 3)
    # Pega a parte fracionada de GB
    remaining_gbytes = gbytes - int(gbytes)
    # Converte o resto de GB para MB
    mbytes = remaining_gbytes * 1024

    # Retorna GB e MB
    return int(gbytes), round(mbytes, 2)


def VGG_16(input_layer):
    '''
    Implementa a arquitetura VGG-16.

    Entradas:
        input_layer (keras.layers.Input): camada de entrada do modelo.

    Retorno:
        keras.layers.Layer: camada de saída do modelo.
    '''

    # Primeiro bloco
    layers = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Segundo bloco
    layers = Conv1D(128, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(128, 3, activation='relu', padding='same')(layers)
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Terceiro bloco
    layers = Conv1D(256, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(256, 3, activation='relu', padding='same')(layers) 
    layers = Conv1D(256, 3, activation='relu', padding='same')(layers)
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Quarto bloco
    layers = Conv1D(512, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(512, 3, activation='relu', padding='same')(layers) 
    layers = Conv1D(512, 3, activation='relu', padding='same')(layers)
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Quinto bloco
    layers = Conv1D(512, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(512, 3, activation='relu', padding='same')(layers) 
    layers = Conv1D(512, 3, activation='relu', padding='same')(layers)
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Bloco de saída
    layers = Flatten()(layers)
    layers = Dense(4096, activation='relu')(layers)
    layers = Dense(128, activation='relu')(layers)


    return layers


def resnet_cnn(input_layer):
    '''
    Implementa a arquitetura ResNet.

    Entradas:
        input_layer (keras.layers.Input): camada de entrada do modelo.

    Retorno:
        keras.layers.Layer: camada de saída do modelo.
    '''

    # Primeiro bloco
    layers = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Conexão de atalho com ajuste de filtros
    skip = layers

    # Segundo bloco
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Add()([layers, skip])
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Conexão de atalho com ajuste de filtros
    skip = layers

    # Terceiro bloco
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers) 
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Add()([layers, skip])
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Conexão de atalho com ajuste de filtros
    skip = layers

    # Quarto bloco
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers) 
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Add()([layers, skip])
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)

    # Conexão de atalho com ajuste de filtros
    skip = layers

    # Quinto bloco
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers) 
    layers = Conv1D(64, 3, activation='relu', padding='same')(layers)
    layers = Add()([layers, skip])
    layers = MaxPooling1D(pool_size=2, strides=2)(layers)
    
    # Bloco de saída
    layers = Flatten()(layers)
    layers = Dense(4096, activation='relu')(layers)
    layers = Dense(128, activation='relu')(layers)

    return layers


def plot_confusion_matrix(cm, model_name, target_names, plot_path='results'):

    '''
    Entradas:
        cm: np.ndarray
        model_name: str
        target_names: list
        plot_path: str

    Retornos:
        Nenhum retorno
    '''

    # Certifica de que a pasta do gráfico exista
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents = True, exist_ok = True)

    # Matriz de confusão
    target_names = np.array([*target_names, 'NoC'])

    # Calcula a normalização da matriz de confusão
    divide = cm.sum(axis = 1, dtype = 'int64')
    divide[divide == 0] = 1
    cm_norm = 100 * cm / divide[:, None]

    # Plota a matriz de confusão
    fig = plot_cm(cm_norm, target_names)
    name = f"{model_name}-cm"
    tight_kws = {'rect' : (0, 0, 1.1, 1)}
    putils.save_fig(fig, name, path = plot_path, figsize = 'square',
                    tight_scale = 'both', usetex = False, tight_kws = tight_kws)
    
    plt.close(fig)

    return

def plot_cm(confusion_matrix, class_names, fontsize = 10, cmap = 'Blues'):

    '''
    Entradas:
        confusion_matrix: np.ndarray
        class_names: list
        fontsize: int
        cmap: str

    Retornos:
        fig: Figure
    '''

    # Plota a matriz de confusão
    fig, ax = plt.subplots()

    df_cm = pd.DataFrame(confusion_matrix, index = class_names, columns = class_names)

    sns.heatmap(df_cm, annot = True, square = True, fmt = '.1f', cbar = False,
                annot_kws = {"size": fontsize}, cmap = cmap, ax = ax,
                xticklabels = class_names, yticklabels = class_names)
    
    for t in ax.texts:
        t.set_text(t.get_text() + '%')

    xticks = ax.get_xticklabels()
    xticks[-1].set_text('NPL')
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticklabels()
    yticks[-1].set_text('NTL')
    ax.set_yticklabels(yticks)

    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_xlabel('Rótulo predito')
    ax.set_ylabel('Rótulo verdadeiro')
    fig.tight_layout()

    return fig

def plot_results(history, model_name, metric, plot_path = 'results'):

    '''
    Entrada:
        history: keras.callbacks.History
        model_name: str
        metric: str
        plot_path: str

    Retornos:
        Nenhum retorno
    '''

    # Certifica de que a pasta do gráfico exista
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents = True, exist_ok = True)

    # Plota resultados
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.history[metric], '-o')
    ax.plot(history.epoch, history.history[f'val_{metric}'], '-*')
    ax.set_xlabel('Épocas')
    ax.set_ylabel(f'{metric}'.capitalize())
    ax.legend(['Conjunto de treinamento', 'Conjunto de validação'])

    # Salva a figura em diferentes formatos
    filename = f'{plot_path}/{model_name}-{metric}'
    fig.savefig(f'{filename}.png', format='png', dpi=600)
    fig.savefig(f'{filename}.pdf', format='pdf')

    plt.close(fig)

    return

def save_data(learning_rate, batch_size, tic, toc, pat_rlr, fac_rlr, pat_early):

    '''
    Entrada:
        learning_rate: float
        batch_size: int
        tic: int
        toc: int
        pat_rlr: int
        fac_rlr: float
        pat_early: int 

    Retornos:
        Nenhum retorno
    '''

    hyperparameter = {'Hiperparametro':learning_rate, 'batch size': batch_size, 'Tempo de inicio': tic, 'Tempo de fim': toc,
                      'Pat_rlr': pat_rlr, 'Fac_rlr': fac_rlr, 'Pat_early': pat_early}
    hyperparameter = pd.DataFrame(hyperparameter, index = [0])


    hyperparameter.to_csv('results/hyperparameter.csv', index = False)

    return