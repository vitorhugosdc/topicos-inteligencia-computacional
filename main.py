import os
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.optimizers import Adam

# Obter o caminho do diretório atual onde o script está
base_dir = os.path.dirname(os.path.realpath(__file__))

# Definição dos caminhos para os diretórios de treino e teste
train_dir = os.path.join(base_dir, 'treino')
test_dir = os.path.join(base_dir, 'teste')

# Dimensões das imagens e tamanho do lote (batch size)
image_size = (224, 224)
batch_size = 32

# Preparação dos dados com aumento de dados para o conjunto de treinamento
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

# O conjunto de teste não deve ter aumento de dados, mas ainda precisa ser normalizado
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

# Este é o modelo que vamos treinar
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base para não serem treinadas
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.n // batch_size
)

# Salvar o modelo treinado
model.save('sword_classifier_model.h5')
