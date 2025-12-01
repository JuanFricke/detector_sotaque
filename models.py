"""
Modelos de Deep Learning para classificação de sotaques
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNAccentClassifier(nn.Module):
    """Modelo CNN para classificação de sotaques"""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(CNNAccentClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResidualBlock(nn.Module):
    """Bloco residual para ResNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class ResNetAccentClassifier(nn.Module):
    """Modelo ResNet para classificação de sotaques"""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(ResNetAccentClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, 
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class AttentionBlock(nn.Module):
    """Bloco de atenção"""
    
    def __init__(self, channels: int):
        super(AttentionBlock, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class AttentionCNNAccentClassifier(nn.Module):
    """Modelo CNN com mecanismo de atenção para classificação de sotaques"""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(AttentionCNNAccentClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention1 = AttentionBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.attention2 = AttentionBlock(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.attention3 = AttentionBlock(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.attention4 = AttentionBlock(512)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1 with attention
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        x = self.pool(x)
        
        # Conv block 2 with attention
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        x = self.pool(x)
        
        # Conv block 3 with attention
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        x = self.pool(x)
        
        # Conv block 4 with attention
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.attention4(x)
        x = self.pool(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class LSTMAccentClassifier(nn.Module):
    """Modelo LSTM para classificação de sotaques"""
    
    def __init__(self, num_classes: int, input_size: int = 128, 
                 hidden_size: int = 256, num_layers: int = 3):
        super(LSTMAccentClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def attention_net(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Aplica mecanismo de atenção"""
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, freq, time)
        # Reshape para LSTM: (batch, time, freq)
        batch_size, channels, freq, time = x.shape
        x = x.squeeze(1)  # Remove channel dimension
        x = x.permute(0, 2, 1)  # (batch, time, freq)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        context = self.attention_net(lstm_out)
        
        # Fully connected
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def get_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Factory function para criar modelos
    
    Args:
        model_name: Nome do modelo ('cnn', 'resnet', 'attention_cnn', 'lstm')
        num_classes: Número de classes
        **kwargs: Argumentos adicionais para o modelo
    
    Returns:
        Modelo instanciado
    """
    models = {
        'cnn': CNNAccentClassifier,
        'resnet': ResNetAccentClassifier,
        'attention_cnn': AttentionCNNAccentClassifier,
        'lstm': LSTMAccentClassifier
    }
    
    if model_name not in models:
        raise ValueError(f"Modelo '{model_name}' não encontrado. "
                        f"Opções disponíveis: {list(models.keys())}")
    
    model_class = models[model_name]
    model = model_class(num_classes=num_classes, **kwargs)
    
    return model


if __name__ == "__main__":
    # Teste dos modelos
    num_classes = 10
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 128, 313)  # (batch, channel, freq, time)
    
    print("Testando modelos...")
    
    # CNN
    model_cnn = get_model('cnn', num_classes)
    output_cnn = model_cnn(input_tensor)
    print(f"CNN output shape: {output_cnn.shape}")
    print(f"CNN total params: {sum(p.numel() for p in model_cnn.parameters()):,}")
    
    # ResNet
    model_resnet = get_model('resnet', num_classes)
    output_resnet = model_resnet(input_tensor)
    print(f"\nResNet output shape: {output_resnet.shape}")
    print(f"ResNet total params: {sum(p.numel() for p in model_resnet.parameters()):,}")
    
    # Attention CNN
    model_attention = get_model('attention_cnn', num_classes)
    output_attention = model_attention(input_tensor)
    print(f"\nAttention CNN output shape: {output_attention.shape}")
    print(f"Attention CNN total params: {sum(p.numel() for p in model_attention.parameters()):,}")
    
    # LSTM
    model_lstm = get_model('lstm', num_classes, input_size=128)
    output_lstm = model_lstm(input_tensor)
    print(f"\nLSTM output shape: {output_lstm.shape}")
    print(f"LSTM total params: {sum(p.numel() for p in model_lstm.parameters()):,}")


