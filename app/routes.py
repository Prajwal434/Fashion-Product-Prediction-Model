from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create the Flask application instance
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Load the trained model (adjust path and number of classes if needed) ---
MODEL_PATH = '../outputs/models/best_model_subset.pth'  # Updated path to the subset model
NUM_BASECOLOUR_CLASSES = 46
NUM_ARTICLETYPE_CLASSES = 143
NUM_SEASON_CLASSES = 5
NUM_GENDER_CLASSES = 5

# --- Define the load_model function here ---
import torch.nn as nn
import torch.hub
from torchvision.models import ResNet50_Weights

class FashionClassifier(nn.Module):
    def __init__(self, num_baseColour_classes, num_articleType_classes, num_season_classes, num_gender_classes, pretrained=True):
        super(FashionClassifier, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_baseColour = nn.Linear(in_features, num_baseColour_classes)
        self.fc_articleType = nn.Linear(in_features, num_articleType_classes)
        self.fc_season = nn.Linear(in_features, num_season_classes)
        self.fc_gender = nn.Linear(in_features, num_gender_classes)

    def forward(self, x):
        x = self.backbone(x)
        out_baseColour = self.fc_baseColour(x)
        out_articleType = self.fc_articleType(x)
        out_season = self.fc_season(x)
        out_gender = self.fc_gender(x)
        return out_baseColour, out_articleType, out_season, out_gender

def load_model(model_path, num_baseColour_classes, num_articleType_classes, num_season_classes, num_gender_classes):
    """Loads the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionClassifier(num_baseColour_classes, num_articleType_classes, num_season_classes, num_gender_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Define attribute mappings
attribute_mappings = {
    'baseColour': {
        '1': 'Navy Blue', '2': 'Black', '3': 'Grey', '4': 'White', '5': 'Red',
        '6': 'Green', '7': 'Yellow', '8': 'Orange', '9': 'Pink', '10': 'Purple',
        '11': 'Brown', '12': 'Beige', '13': 'Khaki', '14': 'Maroon', '15': 'Cream',
        '16': 'Lavender', '17': 'Blue', '18': 'Silver', '19': 'Teal', '20': 'Gold',
        '21': 'Peach', '22': 'Copper', '23': 'Bronze', '24': 'Tan', '25': 'Olive',
        '26': 'Coral', '27': 'Magenta', '28': 'Turquoise', '29': 'Mint', '30': 'Blue',
        '31': 'Cyan', '32': 'Burgundy', '33': 'Rust', '34': 'Off White', '35': 'Charcoal',
        '36': 'Sea Green', '37': 'Mauve', '38': 'Mustard', '39': 'Lime', '40': 'Coffee',
        '41': 'Slate', '42': 'Indigo', '43': 'Fuchsia', '44': 'Taupe', '45': 'Multi',
        '46': 'Metallic'
    },
    'articleType': {
        '1': 'Tshirt', '2': 'Casual Shoes', '3': 'Sports Shoes', '4': 'Formal Shoes', '5': 'Sweatshirt',
        '6': 'Jeans', '7': 'Top', '8': 'Shirt', '9': 'Dress', '10': 'Sandals',
        '11': 'Jacket', '12': 'Shorts', '13': 'Track Pants', '14': 'Casual Trousers', '15': 'Kurta',
        '16': 'Handbag', '17': 'Flats', '18': 'Heels', '19': 'Innerwear Vests', '20': 'Kurta Sets',
        '21': 'Shorts', '22': 'Tshirt', '23': 'Wallet', '24': 'Watches', '25': 'Night Suits',
        '26': 'Bracelet', '27': 'Camisole', '28': 'Cap', '29': 'Clutch', '30': 'Earrings',
        '31': 'Flip Flops', '32': 'Gloves', '33': 'Hat', '34': 'Headband', '35': 'Necklace',
        '36': 'Ring', '37': 'Scarves', '38': 'Socks', '39': 'Sunglasses', '40': 'Sweater',
        '41': 'Belt', '42': 'Blazer', '43': 'Jumpsuit', '44': 'Pendant', '45': 'Churidar',
        '46': 'Duffel Bag', '47': 'Leggings', '48': 'Lounge Pants', '49': 'Waistcoat', '50': 'Sandals',
        '51': 'Skirt', '52': 'Backpack', '53': 'Shirt', '104': 'Formal Shirt', '143': 'Casual Shirt'
        # Add more article types as needed... I've added a few key ones for demonstration
    },
    'season': {
        '1': 'Summer', '2': 'Winter', '3': 'Spring', '4': 'Fall', '5': 'All-Season'
    },
    'gender': {
        '1': 'Male', '2': 'Female', '3': 'Unisex', '4': 'Boys', '5': 'Girls'
    }
}

try:
    model = load_model(MODEL_PATH, NUM_BASECOLOUR_CLASSES, NUM_ARTICLETYPE_CLASSES, NUM_SEASON_CLASSES, NUM_GENDER_CLASSES)
    model.eval()  # Set to evaluation mode for inference
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Load LabelEncoder classes for mapping back to original labels ---
try:
    combined_df = pd.concat(
        [pd.read_csv('../data/processed/train_encoded_metadata.csv'),
         pd.read_csv('../data/processed/val_encoded_metadata.csv')], ignore_index=True)
    label_encoders = {}
    for col in ['baseColour', 'articleType', 'season', 'gender']:
        label_encoders[col] = LabelEncoder().fit(combined_df[col])
except Exception as e:
    print(f"Error loading label encoders: {e}")
    label_encoders = None

# --- Image transformation pipeline (must match training) ---
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path):
    """Predicts the attributes of the fashion image."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        model.to(device)  # Ensure model is on the correct device
        with torch.no_grad():
            out_baseColour, out_articleType, out_season, out_gender = model(image_tensor)
            _, pred_baseColour = torch.max(out_baseColour, 1)
            _, pred_articleType = torch.max(out_articleType, 1)
            _, pred_season = torch.max(out_season, 1)
            _, pred_gender = torch.max(out_gender, 1)
        return pred_baseColour.item(), pred_articleType.item(), pred_season.item(), pred_gender.item()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None, None  # Return None values on error


# --- Route Handlers ---
@app.route('/', methods=['GET'])
def upload_form():
    """Renders the image upload form."""
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    """Handles image upload and prediction."""
    if 'file' not in request.files:
        return render_template('upload.html', error='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', error='No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure directory exists
        file.save(filepath)
        try:
            pred_baseColour_encoded, pred_articleType_encoded, pred_season_encoded, pred_gender_encoded = predict_image(
                filepath)
            if (pred_baseColour_encoded, pred_articleType_encoded, pred_season_encoded,
                    pred_gender_encoded) == (None, None, None, None):
                return render_template('upload.html', error='Failed to predict attributes')

            # Get the numeric codes from label encoders
            predicted_labels = {}
            if label_encoders:  # Only decode if label encoders are loaded
                predicted_labels = {
                    'baseColour': label_encoders['baseColour'].inverse_transform(
                        [pred_baseColour_encoded])[0],
                    'articleType': label_encoders['articleType'].inverse_transform(
                        [pred_articleType_encoded])[0],
                    'season': label_encoders['season'].inverse_transform([pred_season_encoded])[0],
                    'gender': label_encoders['gender'].inverse_transform([pred_gender_encoded])[0]
                }
            else:
                predicted_labels = {  # Return encoded values if decoding fails
                    'baseColour': pred_baseColour_encoded,
                    'articleType': pred_articleType_encoded,
                    'season': pred_season_encoded,
                    'gender': pred_gender_encoded
                }
            
            # Convert the numeric codes to human-readable labels
            human_readable_labels = {}
            for attr, value in predicted_labels.items():
                str_value = str(value)
                if attr in attribute_mappings and str_value in attribute_mappings[attr]:
                    human_readable_labels[attr] = attribute_mappings[attr][str_value]
                else:
                    human_readable_labels[attr] = f"{attr} Code: {str_value}"
            
            # Pass both the numeric codes and human-readable labels to the template
            return render_template('result.html', 
                                  image_path=filename,  # Just the filename, not the full path
                                  predictions=predicted_labels,
                                  human_readable=human_readable_labels)

        except Exception as e:
            return render_template('upload.html', error=f'Error processing image: {e}')
        finally:
            # Don't remove the file, so it can be displayed in the result page
            pass
    else:
        return render_template('upload.html', error='Invalid file type')
    return render_template('upload.html') #Default response

if __name__ == '__main__':
    app.run(debug=True)