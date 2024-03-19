import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import load_model

base_dir = os.path.dirname(os.path.realpath(__file__))

# Definição dos caminhos para os diretórios de treino e teste
train_dir = os.path.join(base_dir, 'treino')
test_dir = os.path.join(base_dir, 'teste')

# Dimensões das imagens e tamanho do lote (batch size)
image_size = (224, 224)
batch_size = 8

# Preparação dos dados com aumento data augmentation para o conjunto de treinamento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Normaliza o conjunto de testes (não é aumentado)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Gerador de dados para treinamento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Gerador de dados para teste
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False #pros testes da matriz de confusão ficarem na ordem correta
)

# Carregar a rede VGG16 com os pesos pré-treinados e sem as camadas superiores
base_model = VGG16(weights='imagenet', include_top=False)

# Adicionar novas camadas superiores para nossa tarefa de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

# Modelo a ser treinado
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base para não serem treinadas
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

steps_per_epoch = max(1, train_generator.n // train_generator.batch_size)
validation_steps = max(1, test_generator.n // test_generator.batch_size)

# Treinamento do modelo
model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=test_generator,
    validation_steps=validation_steps
)

model.save('sword_classifier_model.keras')

class_indices = train_generator.class_indices
class_indices_file = 'class_indices.json'

with open(class_indices_file, 'w') as file:
    json.dump(class_indices, file)

#daqui pra baixo é tudo pra matriz de confusão
with open('class_indices.json') as file:
    class_indices = json.load(file)
class_labels = {v: k for k, v in class_indices.items()}

os.makedirs('./data', exist_ok=True)

# Calculando matriz de confusão
model = load_model('sword_classifier_model.keras')
test_generator.reset()
predictions = model.predict(test_generator, steps=np.ceil(test_generator.n / test_generator.batch_size))
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Transformando a matriz de confusão em um DataFrame do pandas para melhor visualização
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels.values(), columns=class_labels.values())

# Usando seaborn para criar um heatmap da matriz de confusão
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues')
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiros')
plt.xlabel('Predições')
plt.savefig('./data/matriz_de_confusao.png')  
#plt.show()

recall = recall_score(true_classes, predicted_classes, average=None, labels=np.unique(true_classes))
precision = precision_score(true_classes, predicted_classes, average=None, labels=np.unique(true_classes))

# Transformando recall e precision em um DataFrame para melhor visualização
performance_df = pd.DataFrame({'Recall': recall, 'Precision': precision}, index=class_labels.values())

performance_df.to_csv('./data/performance.csv', sep=';',index_label='Class')

print(performance_df)