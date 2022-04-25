from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('Running on device: {}'.format(device))


# initializing mtcnn for face detection
mtcnn = MTCNN(image_size=240, margin=0, device=device, min_face_size=20)
# initializing resnet for face img to embeding conversion
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
dataset = datasets.ImageFolder('data/test_images')  # photos folder path
# accessing names of peoples from folder names
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]


loader = DataLoader(dataset, collate_fn=collate_fn)

face_list = []  # list of cropped faces from photos folder
name_list = []  # list of names corrospoing to cropped photos
# list of embeding matrix after conversion from cropped faces to embedding matrix using resnet
embedding_list = []

transform = T.ToPILImage()

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob > 0.90:  # if face detected and porbability > 90%
        # passing cropped face into resnet model to get embedding matrix
        emb = resnet(face.unsqueeze(0).to(device))
        # resulten embedding matrix is stored in a list
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])  # names are stored in a list

data = [embedding_list, name_list]
torch.save(data, 'data/data.pt')  # saving data.pt file
