import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable

#https://github.com/cfotache/pytorch_imageclassifier/blob/master/PyTorch_Image_Training.ipynb
data_dir = '/home/corleone/pds/resnet-torch/src/sample/val'#
# sudo apt-get install python3-tk

test_transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('/home/schwarm/pds/resnet-torch/kill-the-bits/src/pds.pth',map_location='cpu')
model.eval()

print(model.eval)


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index  

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    print(classes)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels,classes

to_pil = transforms.ToPILImage()
images, labels, classes = get_random_images(50)

rows =10
success =0
#columns = len(images)/rows
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    print(index)
    #sub = fig.add_subplot(1, len(images), ii+1)
    sub = fig.add_subplot(rows, len(images)/rows, ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
    if (res==True):
        success+=1
plt.text(300,300,"accuracy :" + str( ( (success)/len(images))*100 ) + "%")
plt.show()


