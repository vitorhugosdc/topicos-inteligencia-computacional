import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from keras.models import load_model

from PIL import Image, ImageDraw, ImageFont

base_dir = os.path.dirname(os.path.realpath(__file__))

# Definição dos caminhos para os diretórios de treino e teste
train_dir = os.path.join(base_dir, 'treino')
test_dir = os.path.join(base_dir, 'teste')

# Dimensões das imagens e tamanho do lote (batch size)
image_size = (224, 224)
batch_size = 7

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
x = Dropout(0.30)(x)
predictions = Dense(7, activation='softmax')(x)

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
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=25,
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

for i in range(len(test_generator.filenames)):
    # Caminho para a imagem original
    original_image_path = os.path.join(test_dir, test_generator.filenames[i])
    image = Image.open(original_image_path)

    # Converte a imagem para o modo RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Prepara o texto com a classe real, a prevista e a confiança
    true_label = class_labels[true_classes[i]]
    predicted_label = class_labels[predicted_classes[i]]
    confidence = np.max(predictions[i]) * 100

    text = f'Verdadeiro: {true_label}\nPrevisto: {predicted_label} ({confidence:.2f}%)'

    # Adiciona o texto à imagem
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # ou especifique uma fonte
    draw.text((10, 10), text, fill="white", font=font)

    # Salva a imagem anotada
    save_path = os.path.join('./data', f'annotated_{i}.png')
    image.save(save_path)

# Acurácia do Modelo
plt.figure(figsize=(10, 6))  # Ajuste o tamanho conforme necessário
sns.set_theme(style="whitegrid")  # Estilo de fundo do gráfico
plt.plot(range(1, 26), history.history['accuracy'], label='Treino', color='blue', linewidth=2)  # Linha mais espessa
plt.plot(range(1, 26), history.history['val_accuracy'], label='Validação', color='red', linewidth=2)
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(loc='lower right')
plt.xticks(range(1, 26))  # Define explicitamente os ticks do eixo x
plt.xlim(1, 25)  # Define o limite do eixo x para começar na época 1
plt.ylim([0, 1])  # Ajustar o limite do eixo Y se necessário
plt.savefig('./data/model_accuracy.png')  # Salvar o gráfico

# Perda do Modelo
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
plt.plot(range(1, 26), history.history['loss'], label='Treino', color='blue', linewidth=2)
plt.plot(range(1, 26), history.history['val_loss'], label='Validação', color='red', linewidth=2)
plt.title('Perda do Modelo')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(loc='upper right')
plt.xticks(range(1, 26))  # Define explicitamente os ticks do eixo x
plt.xlim(1, 25)  # Define o limite do eixo x para começar na época 1
plt.ylim(0, max(max(history.history['loss']), max(history.history['val_loss'])) * 1.1)  # Ajustar o limite do eixo Y se necessário
plt.savefig('./data/model_loss.png')

conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Transformando a matriz de confusão em um DataFrame do pandas para melhor visualização
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels.values(), columns=class_labels.values())

plt.figure(figsize=(7, 7))

sns.heatmap(conf_matrix_df, annot=True, fmt='g', cmap='Blues')

plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiros')
plt.xlabel('Predições')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./data/matriz_de_confusao.png', dpi=300)

recall = recall_score(true_classes, predicted_classes, average=None, labels=np.unique(true_classes))
precision = precision_score(true_classes, predicted_classes, average=None, labels=np.unique(true_classes))

# Transformando recall e precision em um DataFrame para melhor visualização
performance_df = pd.DataFrame({'Recall': recall, 'Precision': precision}, index=class_labels.values())

performance_df.to_csv('./data/performance.csv', sep=';',index_label='Class')

print(performance_df)