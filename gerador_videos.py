#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para sincronização labial utilizando a estrutura do Wav2Lip.
Este script recebe um vídeo de rosto base e um arquivo de áudio narrado
para gerar o vídeo final sincronizado, utilizando o modelo 'wav2lip_gan.pth'
para priorizar a qualidade da boca.

INTEGRAÇÃO COM UPSCALE (GFPGAN/CodeFormer):
Os pontos de integração para ferramentas de upscale estão marcados com
comentários começando com "[UPSCALE INTEGRATION]" ao longo deste código.
"""

import os
import sys
import argparse
import subprocess
import platform
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Importações do Wav2Lip
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Wav2Lip'))
import audio
from models import Wav2Lip
import face_detection

# ============================================================================
# [UPSCALE INTEGRATION] - IMPORTAÇÕES DE FERRAMENTAS DE UPSCALE
# ============================================================================
# Para integrar GFPGAN ou CodeFormer, descomente e configure as linhas abaixo:
#
# EXEMPLO COM GFPGAN:
# from gfpgan import GFPGANer
# gfpgan_enhancer = GFPGANer(
#     model_path='experiments/pretrained_models/GFPGANv1.4.pth',
#     upscale=2,
#     arch='clean',
#     channel_multiplier=2,
#     bg_upsampler=None  # Ou use real-esrgan para upscaling do fundo também
# )
#
# EXEMPLO COM CODEFORMER:
# from facelib.utils.face_restoration_helper import FaceRestoreHelper
# from basicsr.archs.codeformer_arch import CodeFormer
# codeformer_net = CodeFormer(
#     dim_embd=512,
#     codebook_size=1024,
#     n_head=8,
#     n_layers=9,
#     connect_list=['32', '64', '128', '256']
# ).cuda()
# codeformer_net.load_state_dict(torch.load('weights/CodeFormer/codeformer.pth'))
# codeformer_net.eval()
# ============================================================================


def parse_arguments():
    """
    Configura e retorna os argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description='Gerador de vídeos com sincronização labial usando Wav2Lip'
    )
    
    # Arquivos de entrada e saída
    parser.add_argument(
        '--face', 
        type=str, 
        required=True,
        help='Caminho do arquivo de vídeo ou imagem que contém o rosto base'
    )
    parser.add_argument(
        '--audio', 
        type=str, 
        required=True,
        help='Caminho do arquivo de áudio para sincronização labial'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='results/video_sincronizado.mp4',
        help='Caminho do arquivo de vídeo de saída (padrão: results/video_sincronizado.mp4)'
    )
    
    # Modelo
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        default='checkpoints/wav2lip_gan.pth',
        help='Caminho do checkpoint do modelo Wav2Lip (padrão: checkpoints/wav2lip_gan.pth)'
    )
    
    # Parâmetros de processamento
    parser.add_argument(
        '--static', 
        type=bool, 
        default=False,
        help='Se True, usa apenas o primeiro frame do vídeo para inferência (padrão: False)'
    )
    parser.add_argument(
        '--fps', 
        type=float, 
        default=25.0,
        help='FPS do vídeo de saída (padrão: 25.0)'
    )
    parser.add_argument(
        '--pads', 
        nargs='+', 
        type=int, 
        default=[0, 10, 0, 0],
        help='Padding (top, bottom, left, right) para detecção facial (padrão: [0, 10, 0, 0])'
    )
    parser.add_argument(
        '--resize_factor', 
        type=int, 
        default=1,
        help='Reduz a resolução por este fator para melhor desempenho (padrão: 1)'
    )
    parser.add_argument(
        '--face_det_batch_size', 
        type=int, 
        default=16,
        help='Batch size para detecção facial (padrão: 16)'
    )
    parser.add_argument(
        '--wav2lip_batch_size', 
        type=int, 
        default=128,
        help='Batch size para o modelo Wav2Lip (padrão: 128)'
    )
    parser.add_argument(
        '--nosmooth', 
        action='store_true',
        help='Desativa suavização das detecções faciais temporais'
    )
    
    # [UPSCALE INTEGRATION] - PARÂMETROS DE UPSCALE
    # Adicione estes argumentos quando integrar GFPGAN ou CodeFormer:
    # parser.add_argument(
    #     '--upscale',
    #     action='store_true',
    #     help='Ativa pós-processamento com GFPGAN/CodeFormer para maior nitidez'
    # )
    # parser.add_argument(
    #     '--upscale_model',
    #     type=str,
    #     default='gfpgan',
    #     choices=['gfpgan', 'codeformer'],
    #     help='Modelo de upscale a ser utilizado (padrão: gfpgan)'
    # )
    # parser.add_argument(
    #     '--upscale_factor',
    #     type=int,
    #     default=2,
    #     help='Fator de upscale (padrão: 2)'
    # )
    
    return parser.parse_args()


def get_smoothened_boxes(boxes, T):
    """
    Suaviza as bounding boxes faciais ao longo do tempo para evitar tremores.
    
    Args:
        boxes: Array de bounding boxes detectadas
        T: Janela temporal para suavização
    
    Returns:
        Array de bounding boxes suavizadas
    """
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, args, device):
    """
    Detecta rostos nos frames de vídeo.
    
    Args:
        images: Lista de frames de imagem
        args: Argumentos de configuração
        device: Dispositivo para inferência (cuda/cpu)
    
    Returns:
        Lista de resultados contendo recortes faciais e coordenadas
    """
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, 
        flip_input=False, 
        device=device
    )

    batch_size = args.face_det_batch_size
    
    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size), desc="Detectando rostos"):
                predictions.extend(
                    detector.get_detections_for_batch(np.array(images[i:i + batch_size]))
                )
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError(
                    'Imagem muito grande para detecção facial na GPU. '
                    'Use o argumento --resize_factor'
                )
            batch_size //= 2
            print(f'Recuperando de erro OOM; Novo batch size: {batch_size}')
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    for rect, image in zip(predictions, images):
        if rect is None:
            os.makedirs('temp', exist_ok=True)
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError(
                'Rosto não detectado! Verifique se o vídeo contém rostos em todos os frames.'
            )

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    
    results = [
        [image[y1: y2, x1:x2], (y1, y2, x1, x2)] 
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    del detector
    return results 


# ============================================================================
# [UPSCALE INTEGRATION] - FUNÇÃO DE UPSCALE OPCIONAL
# ============================================================================
# def apply_upscale(face_image, args, device):
#     """
#     Aplica upscale/enhancement no rosto usando GFPGAN ou CodeFormer.
#     
#     Esta função deve ser chamada após o Wav2Lip gerar o frame sincronizado,
#     antes de mesclá-lo de volta ao frame original.
#     
#     Args:
#         face_image: Recorte facial gerado pelo Wav2Lip
#         args: Argumentos de configuração
#         device: Dispositivo para inferência
#     
#     Returns:
#         Imagem facial com upscale aplicado
#     """
#     if not hasattr(args, 'upscale') or not args.upscale:
#         return face_image
#     
#     # Converter BGR para RGB se necessário
#     face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#     
#     if args.upscale_model == 'gfpgan':
#         # GFPGAN retorna imagem em RGB e já redimensionada
#         _, _, enhanced = gfpgan_enhancer.enhance(
#             face_rgb,
#             has_aligned=False,
#             only_center_face=False,
#             paste_back=True
#         )
#         # Converter de volta para BGR
#         enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
#         # Redimensionar para o tamanho original se necessário
#         enhanced_bgr = cv2.resize(enhanced_bgr, (face_image.shape[1], face_image.shape[0]))
#         return enhanced_bgr
#     
#     elif args.upscale_model == 'codeformer':
#         # CodeFormer requer pré-processamento específico
#         # Implementação depende da versão específica do CodeFormer
#         pass
#     
#     return face_image
# ============================================================================


def datagen(frames, mels, args, device):
    """
    Gerador de batches de dados para inferência do Wav2Lip.
    
    Args:
        frames: Lista de frames de vídeo
        mels: Lista de chunks mel-spectrogram
        args: Argumentos de configuração
        device: Dispositivo para inferência
    
    Yields:
        Tuple contendo (img_batch, mel_batch, frame_batch, coords_batch)
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames, args, device)
        else:
            face_det_results = face_detect([frames[0]], args, device)
    else:
        print('Usando bounding box especificada ao invés de detecção facial...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, 
                [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(
            mel_batch, 
            [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        yield img_batch, mel_batch, frame_batch, coords_batch


def load_model(checkpoint_path, device):
    """
    Carrega o modelo Wav2Lip a partir de um checkpoint.
    
    Args:
        checkpoint_path: Caminho para o arquivo .pth do modelo
        device: Dispositivo para carregar o modelo
    
    Returns:
        Modelo Wav2Lip carregado em modo de avaliação
    """
    model = Wav2Lip()
    print(f"Carregando checkpoint de: {checkpoint_path}")
    
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage
        )
    
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    
    model.load_state_dict(new_s)
    model = model.to(device)
    
    return model.eval()


# ============================================================================
# [UPSCALE INTEGRATION] - FUNÇÃO PARA APPLY UPSCALE NO VÍDEO FINAL
# ============================================================================
# def apply_upscale_to_video(input_video_path, output_video_path, args):
#     """
#     Aplica upscale em todo o vídeo após a sincronização labial.
#     
#     Esta é uma abordagem alternativa que aplica o upscale no vídeo completo
#     ao invés de frame-a-frame durante o processo. Pode ser mais lento mas
#     garante consistência.
#     
#     Args:
#         input_video_path: Caminho do vídeo sincronizado pelo Wav2Lip
#         output_video_path: Caminho do vídeo de saída com upscale
#         args: Argumentos de configuração
#     """
#     if not hasattr(args, 'upscale') or not args.upscale:
#         # Se upscale não estiver habilitado, apenas copie o arquivo
#         subprocess.run(['cp', input_video_path, output_video_path])
#         return
#     
#     # Abrir vídeo de entrada
#     cap = cv2.VideoCapture(input_video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     
#     # Configurar escritor de vídeo com resolução aumentada
#     new_width = width * args.upscale_factor
#     new_height = height * args.upscale_factor
#     
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))
#     
#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         
#         # Aplicar upscale no frame
#         # Aqui você integraria GFPGAN ou CodeFormer
#         enhanced_frame = apply_upscale(frame, args, 'cuda')
#         
#         out.write(enhanced_frame)
#         frame_count += 1
#         if frame_count % 30 == 0:
#             print(f"Processando upscale: {frame_count} frames...")
#     
#     cap.release()
#     out.release()
#     
#     # Combinar áudio do vídeo original
#     # (implementação similar à do main)
# ============================================================================


def main():
    """
    Função principal que coordena todo o processo de sincronização labial.
    """
    args = parse_arguments()
    
    # Configurar tamanho da imagem para Wav2Lip (fixo em 96x96)
    args.img_size = 96
    
    # Configurar bounding box opcional
    args.box = [-1, -1, -1, -1]  # Default: auto-detect
    
    # Verificar se arquivo de face existe
    if not os.path.isfile(args.face):
        raise ValueError(
            'O argumento --face deve ser um caminho válido para arquivo de vídeo/imagem'
        )

    # Determinar se é imagem estática
    if args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        args.static = True

    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Usando {device} para inferência.')

    # Criar diretórios temporários e de resultado
    os.makedirs('temp', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Carregar frames do vídeo ou imagem
    if args.static:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Lendo frames de vídeo...')

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            
            if args.resize_factor > 1:
                frame = cv2.resize(
                    frame, 
                    (frame.shape[1] // args.resize_factor, 
                     frame.shape[0] // args.resize_factor)
                )

            full_frames.append(frame)

    print(f"Número de frames disponíveis para inferência: {len(full_frames)}")

    # Processar áudio
    if not args.audio.endswith('.wav'):
        print('Extraindo áudio RAW...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(f"Forma do mel-spectrogram: {mel.shape}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            'Mel contém NaN! Usando voz TTS? Adicione um pequeno ruído epsilon '
            'ao arquivo wav e tente novamente.'
        )

    # Gerar chunks de mel para sincronização
    mel_step_size = 16
    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps 
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print(f"Tamanho dos chunks de mel: {len(mel_chunks)}")

    # Ajustar número de frames ao número de chunks de mel
    full_frames = full_frames[:len(mel_chunks)]

    # Gerador de batches
    gen = datagen(full_frames.copy(), mel_chunks, args, device)

    # Processar batches e gerar vídeo
    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(
        gen, 
        total=int(np.ceil(float(len(mel_chunks)) / args.wav2lip_batch_size)),
        desc="Sincronizando lábios"
    )):
        if i == 0:
            # Carregar modelo Wav2Lip GAN para melhor qualidade
            model = load_model(args.checkpoint, device)
            print("Modelo carregado com sucesso")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter(
                'temp/result.avi', 
                cv2.VideoWriter_fourcc(*'DIVX'), 
                fps, 
                (frame_w, frame_h)
            )

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            
            # Redimensionar predição para o tamanho da região facial
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            # ================================================================
            # [UPSCALE INTEGRATION] - PONTO DE INTEGRAÇÃO FRAME-A-FRAME
            # ================================================================
            # Para aplicar upscale em cada frame após a sincronização labial,
            # descomente a linha abaixo e implemente a função apply_upscale:
            #
            # p = apply_upscale(p, args, device)
            #
            # Isso permitirá que GFPGAN ou CodeFormer melhorem a qualidade
            # da região da boca antes de mesclar com o frame original.
            # ================================================================

            # Mesclar rosto sincronizado de volta ao frame original
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    # Combinar áudio e vídeo finais
    print('Combinando áudio e vídeo...')
    command = (
        f'ffmpeg -y -i {args.audio} -i temp/result.avi '
        f'-strict -2 -q:v 1 {args.output}'
    )
    subprocess.call(command, shell=platform.system() != 'Windows')

    print(f'Vídeo sincronizado salvo em: {args.output}')
    
    # ================================================================
    # [UPSCALE INTEGRATION] - PÓS-PROCESSAMENTO DO VÍDEO COMPLETO
    # ================================================================
    # Se desejar aplicar upscale no vídeo final completo (alternativa
    # ao processamento frame-a-frame), descomente as linhas abaixo:
    #
    # if hasattr(args, 'upscale') and args.upscale:
    #     print('Aplicando upscale no vídeo final...')
    #     upscaled_output = args.output.replace('.mp4', '_upscaled.mp4')
    #     apply_upscale_to_video(args.output, upscaled_output, args)
    #     print(f'Vídeo com upscale salvo em: {upscaled_output}')
    # ================================================================


if __name__ == '__main__':
    main()
