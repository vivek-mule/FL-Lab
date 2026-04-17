import torch
import torch.nn as nn

# Neural network for diabetes prediction
# Input: 10 features (gender, age, hypertension, heart_disease,
#        smoking_history, bmi, HbA1c_level, blood_glucose_level + 2 encoded)
# Output: 2 classes (0 = no diabetes, 1 = diabetes)

class DiabetesNN(nn.Module):
    def __init__(self, input_dim=10):
        super(DiabetesNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)
