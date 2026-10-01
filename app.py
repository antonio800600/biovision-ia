import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import subprocess

DATA_DIR = 'dataset/train'
MODEL_PATH = 'biovision_especialista.pth'

def carregar_modelo():
    class_names = sorted(os.listdir(DATA_DIR))
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, class_names

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classificar_e_sugerir(imagem):
    if imagem is None:
        return "Nenhuma imagem enviada.", ""
    
    model, class_names = carregar_modelo()
    img = Image.fromarray(imagem).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, len(class_names))

    resultado = ""
    especie_prevista = class_names[top_catid[0].item()]
    confianca_max = top_prob[0].item() * 100

    for i in range(top_prob.size(0)):
        idx = top_catid[i].item()
        resultado += f"{class_names[idx]}: {top_prob[i].item() * 100:.2f}%\n"

    return resultado, especie_prevista

def salvar_feedback_e_retreinar(imagem, especie_confirmada):
    if imagem is None or not especie_confirmada:
        return "Erro: Forneça a imagem e o nome da espécie."
    
    # 1. Salva a nova imagem na pasta correspondente
    pasta_destino = os.path.join(DATA_DIR, especie_confirmada)
    os.makedirs(pasta_destino, exist_ok=True)
    
    num_existentes = len(os.listdir(pasta_destino))
    caminho_imagem = os.path.join(pasta_destino, f"auto_{num_existentes + 1}.jpg")
    
    img = Image.fromarray(imagem).convert('RGB')
    img.save(caminho_imagem)
    
    # 2. Executa o treinamento em segundo plano
    subprocess.run(["python", "treinar.py"])
    
    return f"✅ Imagem salva em '{especie_confirmada}'! A BioVision IA re-treinou e atualizou o modelo automaticamente."

# Criar a Interface Gráfica
classes_existentes = sorted(os.listdir(DATA_DIR))

with gr.Blocks(title="BioVision IA - Sistema de Aprendizagem Contínua") as demo:
    gr.Markdown("# 🪲 BioVision IA — Classificação & Aprendizagem Autônoma")
    
    with gr.Row():
        input_img = gr.Image(label="Carregar foto do inseto")
        with gr.Column():
            output_txt = gr.Textbox(label="Análise da IA", lines=5)
            btn_analisar = gr.Button("Analisar Imagem")
            
    gr.Markdown("---")
    gr.Markdown("### 🔄 Loop de Retroalimentação (Ensinar a IA)")
    with gr.Row():
        especie_input = gr.Textbox(label="Confirmar/Corrigir Espécie (Nome da Pasta)", placeholder="Ex: Ascalapha_odorata")
        btn_feedback = gr.Button("Confirmar & Re-treinar IA")
    
    output_feedback = gr.Textbox(label="Status do Treinamento Contínuo")

    btn_analisar.click(classificar_e_sugerir, inputs=[input_img], outputs=[output_txt, especie_input])
    btn_feedback.click(salvar_feedback_e_retreinar, inputs=[input_img, especie_input], outputs=[output_feedback])

demo.launch()