import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18


transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])


current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, '..\\dataset')
dataset = ImageFolder(root=data_path, transform=transform)
num_samples = len(dataset)

batch_size = 4
num_workers = 0 

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

classes = dataset.classes
num_classes=len(classes)

dataiter = iter(trainloader)
images, labels = next(dataiter)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model = resnet18(num_classes=num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)
images=images.to(device)
import torch.optim as optim


num_epochs=20
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion.to(device)

def train_resnet(model, trainloader, criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"{epoch+1}/{num_epochs}training...")
    print("Finish！")

dataiter = iter(testloader)


     
        
train_resnet(model,trainloader,criterion,optimizer,device,num_epochs)
current_folder = os.path.dirname(os.path.abspath(__file__))
save_folder = os.path.join(current_folder, '..', 'model')
os.makedirs(save_folder, exist_ok=True)  

save_path = os.path.join(save_folder, 'resnet18.pth')
torch.save(model.state_dict(), save_path)
state_dict = torch.load(save_path)