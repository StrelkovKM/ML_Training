import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision.datasets import MNIST

from matplotlib import pyplot as plt
from IPython.display import clear_output

from PIL import Image
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# do not change the code in the block below
# __________start of block__________

train_mnist_data = MNIST('.', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_mnist_data = MNIST('.', train=False, transform=torchvision.transforms.ToTensor(), download=True)


train_data_loader = torch.utils.data.DataLoader(
    train_mnist_data,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

test_data_loader = torch.utils.data.DataLoader(
    test_mnist_data,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

random_batch = next(iter(train_data_loader))
_image, _label = random_batch[0][0], random_batch[1][0]
plt.figure()
plt.imshow(_image.reshape(28, 28))
plt.title(f'Image label: {_label}')

# __________end of block__________

# Creating model instance

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, obj):
        obj = obj.view(-1, 784)

        obj = F.relu(self.fc1(obj))
        obj = F.relu(self.fc2(obj))
        obj = F.relu(self.fc3(obj))
        obj = self.fc4(obj)

        return obj
    
model = MNISTModel() # your code here


# do not change the code in the block below
# __________start of block__________
assert model is not None, 'Please, use `model` variable to store your model'

try:
    x = random_batch[0].reshape(-1, 784)
    y = random_batch[1]

    # compute outputs given inputs, both are variables
    y_predicted = model(x)    
except Exception as e:
    print('Something is wrong with the model')
    raise e
    
    
assert y_predicted.shape[-1] == 10, 'Model should predict 10 logits/probas'

print('Everything seems fine!')
# __________end of block__________

# __________optimizer__________
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# __________training loop__________

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Шаг 3: Внутренний цикл по батчам
    for batch_idx, (data, target) in enumerate(train_data_loader):
        # Шаг 4: Обнуление градиентов
        optimizer.zero_grad()
        
        # Шаг 5: Forward pass
        outputs = model(data)
        
        # Шаг 6: Вычисление потерь
        loss = criterion(outputs, target)
        
        # Шаг 7: Backward pass
        loss.backward()
        
        # Шаг 8: Обновление весов
        optimizer.step()
        


predicted_labels = []
real_labels = []
model.eval()
with torch.no_grad():
    for batch in train_data_loader:
        y_predicted = model(batch[0].reshape(-1, 784))
        predicted_labels.append(y_predicted.argmax(dim=1))
        real_labels.append(batch[1])

predicted_labels = torch.cat(predicted_labels)
real_labels = torch.cat(real_labels)
train_acc = (predicted_labels == real_labels).type(torch.FloatTensor).mean()

print(f'Neural network accuracy on train set: {train_acc:3.5}')


predicted_labels = []
real_labels = []
model.eval()
with torch.no_grad():
    for batch in test_data_loader:
        y_predicted = model(batch[0].reshape(-1, 784))
        predicted_labels.append(y_predicted.argmax(dim=1))
        real_labels.append(batch[1])

predicted_labels = torch.cat(predicted_labels)
real_labels = torch.cat(real_labels)
test_acc = (predicted_labels == real_labels).type(torch.FloatTensor).mean()

print(f'Neural network accuracy on test set: {test_acc:3.5}')

assert test_acc >= 0.92, 'Test accuracy is below 0.92 threshold'
assert train_acc >= 0.91, 'Train accuracy is below 0.91 while test accuracy is fine. We recommend to check your model and data flow'


# do not change the code in the block below
# __________start of block__________
import os

assert os.path.exists('hw07_data_dict.npy'), 'Please, download `hw07_data_dict.npy` and place it in the working directory'

def get_predictions(model, eval_data, step=10):
    
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for idx in range(0, len(eval_data), step):
            y_predicted = model(eval_data[idx:idx+step].reshape(-1, 784))
            predicted_labels.append(y_predicted.argmax(dim=1))
    
    predicted_labels = torch.cat(predicted_labels)
    return predicted_labels

loaded_data_dict = np.load('hw07_data_dict.npy', allow_pickle=True)

submission_dict = {
    'train': get_predictions(model, torch.FloatTensor(loaded_data_dict.item()['train'])).numpy(),
    'test': get_predictions(model, torch.FloatTensor(loaded_data_dict.item()['test'])).numpy()
}

np.save('submission_dict_hw07.npy', submission_dict, allow_pickle=True)
print('File saved to `submission_dict_hw07.npy`')
# __________end of block__________

data = np.load('submission_dict_hw07.npy' , allow_pickle=True)

print(data)

def predict_digit(image_path, model, save_path='prediction.png'):
    # Загружаем и обрабатываем изображение
    image = Image.open(image_path).convert('L')  # Конвертируем в grayscale
    
    # Трансформации должны быть такими же, как при обучении
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    
    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_digit].item()
    
    # Визуализация и сохранение в файл
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    digits = range(10)
    plt.bar(digits, probabilities[0].numpy())
    plt.title(f'Prediction: {predicted_digit}\nConfidence: {confidence:.2f}')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(digits)
    
    plt.tight_layout()
    plt.savefig(save_path)  # Сохраняем график в файл
    print(f"Graph saved to {save_path}")
    
    return predicted_digit, confidence

# Пример использования
predicted, conf = predict_digit('2.png', model)