#!/usr/bin/env python
# coding: utf-8


# 실행할라면 아래 코드 터미널에 입력 

"""
    cd SageMaker
    source activate pytorch_p310
    pip install optuna
    nohup python hpo_model_list.py > hpo_log.txt 2>&1 &

"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, models, transforms
import os
import time
import copy
from tqdm import tqdm
from collections import Counter
import sys
import optuna 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shutil

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
ROOT_DIR = "preprocesed_dataset" 
FONT_PATH = './NanumGothic.ttf'  
print(f"사용 장치: {DEVICE}")

checkpoint_path = os.path.join(ROOT_DIR, '.ipynb_checkpoints')

if os.path.isdir(checkpoint_path):
    print(f"Warning: Deleting recurring system folder {checkpoint_path}...")
    shutil.rmtree(checkpoint_path) 
    print("Cleanup complete.")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    #transforms.RandomResizedCrop(224),  # 224x224
    #transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.classes = subset.dataset.classes
        self.class_to_idx = subset.dataset.class_to_idx

    def __getitem__(self, index):
        # 1. Subset에서 원본 (PIL Image, label)을 가져옴
        x, y = self.subset[index]
        
        # 2. 이 클래스에 할당된 transform을 적용
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

full_dataset_raw = datasets.ImageFolder(ROOT_DIR)
train_size = int(len(full_dataset_raw) * 0.8)
val_size = int(len(full_dataset_raw) * 0.1)
test_size = int(len(full_dataset_raw)) - train_size - val_size 

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset_raw,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

full_datasets = {
    'train' : TransformedSubset(train_dataset, train_transforms),
    'val' : TransformedSubset(val_dataset, val_test_transforms),
    'test' : TransformedSubset(test_dataset, val_test_transforms) 
}

train_labels = [full_dataset_raw.targets[i] for i in train_dataset.indices]
train_class_counts = Counter(train_labels)
class_weights = {label: 1.0/count for label, count in train_class_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]

train_sampler = WeightedRandomSampler(
    weights=sample_weights, 
    num_samples=len(sample_weights), 
    replacement=True
)

dataloaders = {
    'train': DataLoader(
        full_datasets['train'], 
        batch_size=128, 
        shuffle=train_sampler, # 데이터가 적은 샘플러는 여러번 뽑음
        num_workers=8
    ),
    'val': DataLoader(
        full_datasets['val'], 
        batch_size=128, 
        shuffle=False, # val 섞을 필요 없음
        num_workers=8
    ),
    'test': DataLoader(
        full_datasets['test'], 
        batch_size=128, 
        shuffle=False, # test 섞을 필요 없음
        num_workers=8
    )
    }

def train_model_HPO(model, criterion, optimizer, scheduler, num_epochs=13, patience = 3, min_delta = 0.001, trial = None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 에폭은 훈련(train) 단계와 검증(val) 단계를 거칩니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase] :
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # --- 순전파 (Forward) ---
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # --- 역전파 (Backward) + 최적화 ---
                    # 훈련 모드일 때만 역전파 및 가중치 업데이트 수행
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 배치별 통계 기록
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # 훈련 단계가 끝나면 스케줄러 업데이트
            if phase == 'train':
                scheduler.step()

            # 에폭별 평균 손실 및 정확도 계산
            epoch_loss = running_loss / len(full_datasets[phase])
            epoch_acc = running_corrects.double() / len(full_datasets[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # --- 최고 성능 모델 저장 ---
            if phase == 'val' and trial is not None :
                trial.report(epoch_acc, epoch)

                # 2. 가지치기(Pruning) 결정
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch}")
                    raise optuna.exceptions.TrialPruned()
                if epoch_acc > best_acc + min_delta :
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # 최고 모델 저장
                    torch.save(model.state_dict(), 'best_model_hpo.pth')
                    early_stopping_counter = 0 
                else :
                    early_stopping_counter += 1
                    print(f"성능 향상 X, 스탑 카운터 : {early_stopping_counter}")
                    if early_stopping_counter >= patience:
                        time_elapsed = time.time() - since
                        print(f'\n--- 얼리 스탑: {patience} 에폭 동안 성능 향상이 없어 훈련을 중단합니다. ---')
                        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                        print(f'Best val Acc: {best_acc:.4f}')
                        model.load_state_dict(best_model_wts)
                        return best_acc
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # 최고 성능 모델 가중치로 교체
    model.load_state_dict(best_model_wts)
    return best_acc

def objective(trial):
    """Optuna가 1회 실행할 훈련 프로세스"""
    
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    model_name = trial.suggest_categorical("model_name", ["efficientnet_b0", "convnext_tiny", "resnet50", "mobilenet_v3"])
    target_epochs = trial.suggest_int("epochs", 10, 20)
    jitter_strength = trial.suggest_float("jitter_strength", 0.03, 0.4)
    
    scheduler_step = trial.suggest_int("scheduler_step", 3, 10)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    NUM_CLASSES = len(full_dataset_raw.classes)

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        input_size = 224 # B0용 사이즈
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights='IMAGENET1K_V1')
        input_size = 224 # convnext_tiny 사이즈
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        input_size = 224 # ResNet용 사이즈
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif model_name == "mobilenet_v3":
        # 가장 최신 버전인 V3 Large 로드
        model = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        input_size = 224 # MobileNet 권장 사이즈
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)

    model = model.to(DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: # SGD
        # SGD는 
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss() 
    
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=jitter_strength, 
                               contrast=jitter_strength, 
                               saturation=jitter_strength, 
                               hue=0.05),
        transforms.Resize(int(input_size*1.2)),
        transforms.RandomCrop(int(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize(int(input_size*1.2)),
        transforms.CenterCrop(int(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    full_datasets = {
        'train' : TransformedSubset(train_dataset, train_transforms),
        'val' : TransformedSubset(val_dataset, val_test_transforms),
        'test' : TransformedSubset(test_dataset, val_test_transforms) 
    }

    
    dataloaders = {
    'train': DataLoader(
        full_datasets['train'], 
        batch_size=128, 
        shuffle=train_sampler, # 데이터가 적은 샘플러는 여러번 뽑음
        num_workers=8
    ),
    'val': DataLoader(
        full_datasets['val'], 
        batch_size=128, 
        shuffle=False, # val 섞을 필요 없음
        num_workers=8
    ),
    'test': DataLoader(
        full_datasets['test'], 
        batch_size=128, 
        shuffle=False, # test 섞을 필요 없음
        num_workers=8
    )
    }

    best_val_acc = train_model_HPO(model, criterion, optimizer, exp_lr_scheduler, 
                               num_epochs=15, 
                               patience=4, 
                               min_delta=0.001, trial = trial)
    
    return best_val_acc

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=4)
)
study.optimize(objective, n_trials=50) 

print("="*30)
print("HPO가 완료되었습니다.")
print("최고 점수 (Best val Acc):", study.best_value)
print("최적의 하이퍼파라미터:", study.best_params)
print("="*30)