import argparse

import torch
import torch.nn as nn
import torchvision

from transformers import Wav2Vec2Model, ClapModel, ClapProcessor

# Configuration path
MODEL_PATH = "/users/zfne/mustun/Documents/GitHub/Dolph2Vec/dolph2vec-base/"
#MODEL_PATH = "/users/zfne/mustun/Documents/GitHub/Dolph2Vec/model-dolph2vec_type-singleCB_data-DolphinChat_version-v0/"
# MODEL_PATH = "dolphinteam/model-dolph2vec_type-base_data-DolphinChat_version-v0"
# with open("/home/robertodessi/Dolph2Vec/.hf_token") as f:
#     token = f.read().strip()

class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class BiolingualClassifier(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
        self.model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
        
        # Freeze the audio encoder
        for param in self.model.audio_model.parameters():
            param.requires_grad = False
            
        # Keep the projection layers trainable (if any)
        for param in self.model.audio_projection.parameters():
            param.requires_grad = True
            
        self.classification_head = nn.Linear(512, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        device = x.device
        # inputs = self.processor(audios=x.cpu(), return_tensors="pt", sampling_rate=48000).to(device)
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=48000, padding=True).to(device)

        audio_embed = self.model.get_audio_features(**inputs)
        
        # pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classification_head(audio_embed)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class Wav2vec2Classifier(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        # self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_PATH, token=token, force_download=True)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_PATH)
        self.classification_head = nn.Linear(768, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        wav2vec2_out = self.wav2vec2(x)
        # Extract last_hidden_state from wav2vec2 output
        hidden_states = wav2vec2_out.last_hidden_state
        # Take mean over sequence length dimension
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classification_head(pooled_output)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class Wav2vec2Classifier_zarr(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        # self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_PATH, token=token, force_download=True)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(MODEL_PATH)
        self.classification_head = nn.Linear(768, num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        wav2vec2_out = self.wav2vec2.encoder(x)
        # Extract last_hidden_state from wav2vec2 output
        hidden_states = wav2vec2_out.last_hidden_state
        # Take mean over sequence length dimension
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classification_head(pooled_output)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
