import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.hub
from torchvision.models import ResNet50_Weights
import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Image transformation for prediction
def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Custom dataset class for single image prediction
class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img = Image.open(self.image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

class FashionClassifier(nn.Module):
    def __init__(self, num_baseColour_classes, num_articleType_classes, num_season_classes, num_gender_classes, pretrained=False):
        super(FashionClassifier, self).__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=ResNet50_Weights.DEFAULT if pretrained else None)
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

def load_model(model_path, num_baseColour_classes, num_articleType_classes, num_season_classes, num_gender_classes, device='cpu'):
    """Loads the trained model.

    Args:
        model_path (str): Path to the trained model file (.pth).
        num_baseColour_classes (int): Number of base colour classes.
        num_articleType_classes (int): Number of article type classes.
        num_season_classes (int): Number of season classes.
        num_gender_classes (int): Number of gender classes.
        device (str, optional): Device to load the model on. Defaults to 'cpu'.

    Returns:
        nn.Module: The loaded model.
    """
    model = FashionClassifier(num_baseColour_classes, num_articleType_classes, num_season_classes, num_gender_classes, pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) # Load to specified device
    model.eval()
    return model

def create_label_encoders(train_df, val_df):
    """
    Creates LabelEncoders for the specified columns, fits them to the combined training and
    validation data, and returns the fitted encoders.
    """
    label_encoders = {}
    combined_df = pd.concat([train_df, val_df], ignore_index=True)

    for col in ['baseColour', 'articleType', 'season', 'gender']:
        encoder = LabelEncoder()
        encoder.fit(combined_df[col])  # Fit on combined data
        label_encoders[col] = encoder  # Store the fitted encoder

    return label_encoders

def predict_image_attributes(image_path, model, label_encoders,
                             num_baseColour_classes, num_articleType_classes,
                             num_season_classes, num_gender_classes,
                             device='cpu'):
    """
    Predicts the attributes of a single image and returns decoded attribute names.

    Args:
        image_path (str): Path to the image file.
        model (nn.Module):  Loaded FashionClassifier model.
        label_encoders (dict): A dictionary of fitted LabelEncoder objects for each attribute.
        num_baseColour_classes (int): Number of base colour classes.
        num_articleType_classes (int): Number of article type classes.
        num_season_classes (int): Number of season classes.
        num_gender_classes (int): Number of gender classes.
        device (str):  'cpu' or 'cuda'

    Returns:
        dict: A dictionary containing the predicted attribute values.
    """
    try:
        # Prepare the image
        transform = get_image_transform()
        dataset = SingleImageDataset(image_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for img in dataloader:
                img = img.to(device)
                out_baseColour, out_articleType, out_season, out_gender = model(img)

                # Get the predicted class indices
                _, predicted_baseColour_idx = torch.max(out_baseColour, 1)
                _, predicted_articleType_idx = torch.max(out_articleType, 1)
                _, predicted_season_idx = torch.max(out_season, 1)
                _, predicted_gender_idx = torch.max(out_gender, 1)

                # Decode the predictions using the provided LabelEncoders
                decoded_predictions = {}
                decoded_predictions['baseColour'] = label_encoders['baseColour'].inverse_transform([predicted_baseColour_idx.item()])[0]
                decoded_predictions['articleType'] = label_encoders['articleType'].inverse_transform([predicted_articleType_idx.item()])[0]
                decoded_predictions['season'] = label_encoders['season'].inverse_transform([predicted_season_idx.item()])[0]
                decoded_predictions['gender'] = label_encoders['gender'].inverse_transform([predicted_gender_idx.item()])[0]

                return decoded_predictions

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

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
        # Add more article types as needed...
    },
    'season': {
        '1': 'Summer', '2': 'Winter', '3': 'Spring', '4': 'Fall', '5': 'All-Season'
    },
    'gender': {
        '1': 'Male', '2': 'Female', '3': 'Unisex', '4': 'Boys', '5': 'Girls'
    }
}

def get_human_readable_label(attribute, code):
    """
    Get human-readable label for an attribute code.

    Args:
        attribute (str): The attribute name (baseColour, articleType, season, gender)
        code (str or int): The numeric code for the attribute

    Returns:
        str: Human-readable label or the original code if not found
    """
    str_code = str(code)
    if attribute in attribute_mappings and str_code in attribute_mappings[attribute]:
        return attribute_mappings[attribute][str_code]
    return f"{attribute} Code: {str_code}"

if __name__ == '__main__':
    # Example usage:
    num_baseColour = 46
    num_articleType = 143
    num_season = 5
    num_gender = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the model
    model = FashionClassifier(num_baseColour, num_articleType, num_season, num_gender).to(device)
    print("FashionClassifier model created:")
    print(model)

    # Example of loading the *subset trained* model
    trained_model_path = '../outputs/models/best_model_subset.pth'
    loaded_model = load_model(trained_model_path, num_baseColour, num_articleType, num_season, num_gender, device)
    print("\nTrained subset model loaded:")
    print(loaded_model)

    # Example of making a prediction with the loaded model
    # Load training and validation data for LabelEncoders
    try:
        train_df = pd.read_csv('../data/processed/train_encoded_metadata.csv')
        val_df = pd.read_csv('../data/processed/val_encoded_metadata.csv')
        label_encoders = create_label_encoders(train_df, val_df)
        print("\nLabel encoders created from training and validation data.")
        
        # Make prediction with a test image
        image_path = 'test_image.jpg'  # Replace with your image path
        if os.path.exists(image_path):
            predicted_attributes = predict_image_attributes(
                image_path, loaded_model, label_encoders,
                num_baseColour, num_articleType, num_season, num_gender, device
            )
            
            if predicted_attributes:
                print("\nPredicted Attributes:")
                for attr, code in predicted_attributes.items():
                    human_readable = get_human_readable_label(attr, code)
                    print(f"{attr}: {human_readable} (Code: {code})")
            else:
                print("\nFailed to predict attributes.")
        else:
            print(f"Error: Image not found at {image_path}. Please make sure the path is correct and the image exists.")
    except Exception as e:
        print(f"Error: {e}")