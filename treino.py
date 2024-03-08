import os
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam

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
    class_mode='categorical'
)

# Carregar a rede VGG16 com os pesos pré-treinados e sem as camadas superiores
base_model = VGG16(weights='imagenet', include_top=False)

# Adicionar novas camadas superiores para nossa tarefa de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

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