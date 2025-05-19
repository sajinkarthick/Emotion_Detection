import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# FER-2013 class labels
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define CNN model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.dropout_conv = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 7)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.LeakyReLU(self.batch_norm1(self.conv1(x)))
        x = self.LeakyReLU(self.batch_norm2(self.conv2(x)))
        x = self.LeakyReLU(self.batch_norm3(self.conv3(x)))
        x = self.dropout_conv(x)
        x = self.pool1(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.LeakyReLU(self.fc1(x)))
        x = self.dropout2(self.LeakyReLU(self.fc2(x)))
        x = self.fc3(x)
        return x

# Load model
device = torch.device("cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.eval()

# Image transform pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Emotion prediction function
def predict_emotion(image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 1, 48, 48)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = class_names[np.argmax(probs)]
    return pred_label, probs

# Function to display a bar chart
def display_bar_chart(probs, color='orange'):
    fig, ax = plt.subplots()
    ax.bar(class_names, probs, color=color)
    ax.set_ylabel("Probability")
    ax.set_title("Emotion Probabilities")
    ax.set_ylim([0, 1])
    st.pyplot(fig, use_container_width=True)

st.title("Emotion Detection")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, probabilities = predict_emotion(image)
    st.markdown(f"### 🧠 Predicted Emotion: **{label}**")
    display_bar_chart(probabilities, color='orange')
else:
    st.info("Please upload an image to see predictions and probabilities.")
    dummy_probs = [0] * len(class_names)
    display_bar_chart(dummy_probs, color='lightgray')
