import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# 1. Identificar as classes treinadas
data_dir = 'dataset/train'
class_names = sorted(os.listdir(data_dir))

# 2. Carregar a arquitetura do modelo
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# 3. Carregar os pesos que a tua IA aprendeu
model.load_state_dict(torch.load('biovision_especialista.pth'))
model.eval()

# 4. Transformações de imagem para o teste
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

nome_imagem = "inseto.jpg" # Garanta que esta foto está na pasta BioVisionIA

try:
    print(f"1. Analisando a foto '{nome_imagem}' com a BioVision Especialista...")
    img = Image.open(nome_imagem).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, len(class_names))

    print("\n=== RESULTADO DA ANÁLISE ESPECIALIZADA ===")
    for i in range(top_prob.size(0)):
        idx = top_catid[i].item()
        nome_categoria = class_names[idx]
        confianca = top_prob[i].item() * 100
        print(f"{i+1}. {nome_categoria} — Confiança: {confianca:.2f}%")

except FileNotFoundError:
    print(f"\n❌ Erro: O ficheiro '{nome_imagem}' não foi encontrado!")