import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

# Carregar o modelo treinado
model = load_model('sword_classifier_model.keras')

# Obter o mapeamento de índices de classe para nomes de classe do gerador
base_dir = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(base_dir, 'treino')
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

class_labels = dict((v,k) for k,v in train_generator.class_indices.items())

# Preparar a imagem para predição
def prepare_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

image_path = './straight_sword/teste/warhawks_talon_straight_sword.png'
image = prepare_image(image_path)

# Fazer a predição
predictions = model.predict(image)
predicted_class_index = np.argmax(predictions, axis=1)
predicted_class_name = class_labels[predicted_class_index[0]]
predicted_probability = np.max(predictions)

print(f"Classe Predita: {predicted_class_name}, Probabilidade: {predicted_probability:.4f}")
