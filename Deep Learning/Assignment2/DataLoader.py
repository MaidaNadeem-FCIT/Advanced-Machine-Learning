class OcularDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    @property
    def classes(self):
        return self.data.classes

    
#DatLoader
transform = transforms.Compose([
    transforms.Resize((128, 128)),    
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dir = 'Ocular_Disease_Classification/Train_Test_Validate/train'
test_dir = 'Ocular_Disease_Classification/Train_Test_Validate/test'
validate_dir = 'Ocular_Disease_Classification/Train_Test_Validate/validate'

trainset = OcularDataset(train_dir, transform)
testset = OcularDataset(test_dir, transform)
validateset = OcularDataset(validate_dir, transform)


trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=True)
validateloader = DataLoader(validateset, batch_size=32, shuffle=True)