import torch
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
from simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load("model_cat_rabbit.pth", weights_only=True, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = ['cat', 'rabbit']

cat_dir = r"archive\test-images\cat"
rabbit_dir = r"archive\test-images\rabbit"

cat_files = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png'))]
rabbit_files = [os.path.join(rabbit_dir, f) for f in os.listdir(rabbit_dir) if f.lower().endswith(('.jpg', '.png','.jpeg'))]
all_files = cat_files + rabbit_files
random.shuffle(all_files)

print("CAT", cat_files)
print("RABBIT", rabbit_files)

for img_path in all_files:
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]
        print(f"Image: {os.path.basename(img_path)} -> Detected as: {predicted_class}")

    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        continue

    text = f"Detected: {predicted_class}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    img_height, img_width = img_cv2.shape[:2]
    x_center = (img_width - text_width) // 2
    y_center = (img_height + text_height) // 2

    cv2.putText(img_cv2, text, (x_center, y_center), font, scale, (0, 255, 0), thickness)

    cv2.imshow("Demo Inference", img_cv2)
    key = cv2.waitKey(0)
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
