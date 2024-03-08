import csv
import os
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Carregar o modelo
model = load_model('sword_classifier_model.keras')

# Diretório com as imagens para testar
test_images_dir = './testes_modelo'

# mapeamento de índices de classe para nomes de classe
class_labels = {0: 'colossal_swords', 1: 'dagger', 2: 'straight_sword'}

# Preparar uma imagem
def prepare_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

results_file = 'test_results.csv'
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Predicted Class', 'Probability'])

    # Iterar sobre todas as imagens no diretório de teste
    for image_name in os.listdir(test_images_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_images_dir, image_name)
            image = prepare_image(image_path)
            predictions = model.predict(image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_labels[predicted_class_index]
            predicted_probability = np.max(predictions)
            writer.writerow([image_name, predicted_class_name, f"{predicted_probability:.4f}"])

print(f"Results saved to {results_file}")
