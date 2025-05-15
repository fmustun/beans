import argparse

import torch
import torch.nn as nn
import torchvision

from aves import AVESClassifier
from transformers import (
    ClapModel,
    ClapProcessor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)


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
    def __init__(self, sample_rate, num_classes=None):
        super().__init__()
        self.processor = ClapProcessor.from_pretrained("davidrrobinson/biolingual")
        self.model = ClapModel.from_pretrained("davidrrobinson/biolingual")
        self.linear = nn.Linear(in_features=512, out_features=num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self, x, y=None):
        device = x.device
        x = x.cpu().numpy()

        processed = self.processor(audios=x, return_tensors="pt", sampling_rate=48000)
        inputs = processed["input_features"].to(device)

        outputs = self.model.get_audio_features(input_features=inputs)
        logits = self.linear(outputs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class AvesClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None):
        super().__init__()

        self.model = AVESClassifier(
            config_path="/home/rdessi/Dolph2Vec/aves_models/aves_bio/aves-base-bio.torchaudio.model_config.json",
            model_path="/home/rdessi/Dolph2Vec/aves_models/aves_bio/aves-base-bio.torchaudio.pt",
            num_classes=num_classes,
            freeze_feature_extractor=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
            for_inference=False,
        )

    def __call__(self, x, y=None):
        loss, logits = self.model(x, y)
        return loss, logits


class Dolph2VecClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_json_file(
            "/home/rdessi/Dolph2Vec/preprocessor_dolphin.json",
        )

        self.model = Wav2Vec2Model.from_pretrained(
            "dolphinteam/model-dolph2vec_type-base_data-DolphinChat_version-v0",
        )

        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def __call__(self, x, y=None):
        device = x.device
        features = self.feature_extractor(
            raw_speech=[w.cpu().numpy() for w in x],
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            truncation=False,
            # max_length = int(20* self.feature_extractor.sampling_rate),
            # min_length = int(2*self.feature_extractor.sampling_rate)
        )["input_values"]

        features = features.to(device)

        out = self.model(features, output_hidden_states=True)
        logits = out.hidden_states[-1].mean(1)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
