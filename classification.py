# Bibliotecas
import numpy as np
from datetime import datetime

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint
from keras.models import Model   
import keras.optimizers as kopt  
from keras.layers import Input, Dense

import pathlib

import mlcm, utils


# Carregar os dados
data = np.load('data.npz')

# Conjunto de treinamento
X_train = data['X_train']
y_train = data['y_train']
# Conjunto de validação
X_val = data['X_val']
y_val = data['y_val']
# Conjunto de teste
X_test = data['X_test']
y_test = data['y_test']

# Sequencia de nomes das classes
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# Hiperparâmetros
learning_rate = 0.001
batch_size = 16
optimizer = kopt.Adam(learning_rate=learning_rate)
loss = 'binary_crossentropy'
epochs = 100

# model_name = 'vgg_16'
model_name = 'resnet_cnn' 

# Reset do keras
utils.reset_keras()
# Setar a GPU
utils.set_gpu(0)

# Camada de entrada
input_layer = Input(shape=X_train.shape[1:])

# Criar os modelos
if model_name == 'vgg_16':
    layers = utils.VGG_16(input_layer)
elif model_name == 'resnet_cnn':
    layers = utils.resnet_cnn(input_layer)
else:
    print('Modelo não encontrado.')
    exit()

# Camada de saída
classification = Dense(5, activation = 'sigmoid')(layers)
# Construindo o modelo
model = Model(inputs = input_layer, outputs = classification)
# model.summary()

# Compilar o modelo
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

 # Memória utilizada pelo modelo
gbytes, mbytes = utils.get_model_memory_usage(batch_size, model)
print(f'Model: {model_name} - (GPU) Memory requirements: {gbytes} GB and {mbytes} MB')

# Diretórios
plot_path = pathlib.Path('results')
csv_path = plot_path / f'{model_name}-history.csv'
csv_path.parent.mkdir(parents=True, exist_ok=True)
model_path = f'results/{model_name}-model.h5'

# Hiperparâmetros dos callbacks
pat_rlr = 10
fac_rlr = 0.5
pat_early = 15

# Callbacks
callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=fac_rlr, patience=pat_rlr, mode='min', min_lr=1e-6),
             EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=pat_early),
             ModelCheckpoint(model_path, monitor='val_loss', mode='auto', verbose=1, save_best_only=True),
             CSVLogger(csv_path, separator=',', append=True)]

# Treinamento do modelo
tic = datetime.now().isoformat()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=callbacks)
toc = datetime.now().isoformat()

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(f"Custo de teste = {score[0]:.4f}")
print(f"Acurácia de teste = {100*score[1]:.2f}%")

# Predição do modelo
y_pred = model.predict(X_test, verbose=0)
# Converte as predições para valores binários
y_pred = np.array(y_pred)
y_pred = (y_pred > 0.5).astype('int')

# Matriz de confusão MLCM
print('\n--- Matriz de confusão MLCM ---')
_, cm = mlcm.cm(y_test, y_pred, print_note=False)
# Imprime o report da MLCM
mlcm.stats(cm)

# Salva os hiperparâmetros
utils.save_data(learning_rate, batch_size, tic, toc, pat_rlr, fac_rlr, pat_early)

# Plota e salva a matriz de confusão
utils.plot_confusion_matrix(cm, model_name, target_names)

# Plota e salva o gráfico de custo
utils.plot_results(history, model_name, metric='loss',)
