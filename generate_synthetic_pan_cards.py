#!/usr/bin/env python3
"""
Synthetic PAN Card Generator for ML Dataset Creation
Generates fake PAN card images with random data for OCR training.
All data is synthetic - NO real personal information.
"""

import os
import json
import random
import string
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Configuration
OUTPUT_DIR = "dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.jsonl")
NUM_SAMPLES = 1000

# PAN Card dimensions (similar to real size ratio)
CARD_WIDTH = 850
CARD_HEIGHT = 540

# Sample data for generation
FIRST_NAMES = [
    "RAHUL", "PRIYA", "AMIT", "SNEHA", "VIJAY", "ANITA", "RAJESH", "DEEPIKA",
    "SANJAY", "POOJA", "ARJUN", "KAVITA", "MANOJ", "NEHA", "ROHAN", "SWATI",
    "ADITYA", "SHRUTI", "KARAN", "RITU", "VISHAL", "MEERA", "NIKHIL", "ASHA"
]

LAST_NAMES = [
    "KUMAR", "SHARMA", "PATEL", "SINGH", "GUPTA", "VERMA", "REDDY", "MEHTA",
    "AGARWAL", "JOSHI", "DESAI", "NAIR", "IYER", "PANDEY", "MISHRA", "RAO"
]

def generate_pan_number():
    """Generate a synthetic PAN number in format: AAAAA1234A"""
    letters1 = ''.join(random.choices(string.ascii_uppercase, k=5))
    digits = ''.join(random.choices(string.digits, k=4))
    letter2 = random.choice(string.ascii_uppercase)
    return f"{letters1}{digits}{letter2}"

def generate_name():
    """Generate a random Indian name"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def generate_father_name():
    """Generate a random father's name"""
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"

def generate_dob():
    """Generate a random date of birth"""
    start_date = datetime(1960, 1, 1)
    end_date = datetime(2005, 12, 31)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    dob = start_date + timedelta(days=random_days)
    return dob.strftime("%d/%m/%Y")

def create_pan_card_image(pan_data, output_path):
    """Create a synthetic PAN card image"""
    # Create image with gradient background
    img = Image.new('RGB', (CARD_WIDTH, CARD_HEIGHT), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add border
    border_color = (0, 0, 150)
    draw.rectangle([10, 10, CARD_WIDTH-10, CARD_HEIGHT-10], outline=border_color, width=3)
    
    # Try to use a system font, fallback to default
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        large_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        medium_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        title_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
        medium_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Header
    draw.text((250, 30), "INCOME TAX DEPARTMENT", fill=(0, 0, 150), font=title_font)
    draw.text((280, 75), "Permanent Account Number Card", fill=(100, 100, 100), font=small_font)
    
    # Photo placeholder (gray box)
    photo_x, photo_y = 50, 120
    draw.rectangle([photo_x, photo_y, photo_x+120, photo_y+140], fill=(200, 200, 200), outline=(0, 0, 0), width=2)
    draw.text((photo_x+35, photo_y+60), "PHOTO", fill=(100, 100, 100), font=small_font)
    
    # Details section
    details_x = 200
    y_offset = 140
    line_spacing = 50
    
    # Name
    draw.text((details_x, y_offset), "Name", fill=(50, 50, 50), font=small_font)
    draw.text((details_x, y_offset + 25), pan_data['name'], fill=(0, 0, 0), font=medium_font)
    
    # Father's Name
    y_offset += line_spacing + 20
    draw.text((details_x, y_offset), "Father's Name", fill=(50, 50, 50), font=small_font)
    draw.text((details_x, y_offset + 25), pan_data['father_name'], fill=(0, 0, 0), font=medium_font)
    
    # Date of Birth
    y_offset += line_spacing + 20
    draw.text((details_x, y_offset), "Date of Birth", fill=(50, 50, 50), font=small_font)
    draw.text((details_x, y_offset + 25), pan_data['dob'], fill=(0, 0, 0), font=medium_font)
    
    # PAN Number (prominent)
    draw.rectangle([details_x-10, 420, details_x+350, 490], fill=(240, 240, 255), outline=(0, 0, 150), width=2)
    draw.text((details_x, 435), pan_data['pan'], fill=(0, 0, 150), font=large_font)
    
    # Signature placeholder
    sig_x, sig_y = 600, 380
    draw.line([sig_x, sig_y+40, sig_x+150, sig_y+40], fill=(0, 0, 0), width=1)
    draw.text((sig_x+30, sig_y+45), "Signature", fill=(100, 100, 100), font=small_font)
    
    # Save image
    img.save(output_path)
    return img

def apply_augmentations(image_path):
    """Apply random augmentations to simulate real-world conditions"""
    img = cv2.imread(image_path)
    
    # Random rotation (-5 to 5 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-5, 5)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))
    
    # Random brightness
    if random.random() > 0.5:
        brightness = random.uniform(0.7, 1.3)
        img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
    
    # Random blur
    if random.random() > 0.7:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Random noise
    if random.random() > 0.7:
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    cv2.imwrite(image_path, img)

def generate_dataset(num_samples=NUM_SAMPLES):
    """Generate the complete synthetic PAN card dataset"""
    # Create output directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    labels = []
    
    print(f"Generating {num_samples} synthetic PAN cards...")
    
    for i in range(num_samples):
        # Generate random data
        pan_data = {
            'image': f"pan_{i:05d}.png",
            'pan': generate_pan_number(),
            'name': generate_name(),
            'father_name': generate_father_name(),
            'dob': generate_dob()
        }
        
        # Create image
        output_path = os.path.join(IMAGES_DIR, pan_data['image'])
        create_pan_card_image(pan_data, output_path)
        
        # Apply augmentations
        if i % 2 == 0:  # Apply to 50% of images
            apply_augmentations(output_path)
        
        # Save label
        labels.append(pan_data)
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} images")
    
    # Save all labels to JSONL file
    with open(LABELS_FILE, 'w') as f:
        for label in labels:
            f.write(json.dumps(label) + '\n')
    
    print(f"\nDataset generation complete!")
    print(f"Images saved to: {IMAGES_DIR}")
    print(f"Labels saved to: {LABELS_FILE}")
    print(f"\nTotal samples: {num_samples}")
    
    # Generate metadata
    metadata = {
        'dataset_name': 'Synthetic PAN Card OCR Dataset',
        'num_samples': num_samples,
        'image_format': 'PNG',
        'image_dimensions': f"{CARD_WIDTH}x{CARD_HEIGHT}",
        'fields': ['pan', 'name', 'father_name', 'dob'],
        'synthetic': True,
        'created_at': datetime.now().isoformat(),
        'disclaimer': 'All data is synthetic. No real personal information.'
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {os.path.join(OUTPUT_DIR, 'metadata.json')}")

if __name__ == "__main__":
    print("="*60)
    print("Synthetic PAN Card Dataset Generator")
    print("All generated data is FAKE and for ML training only")
    print("="*60)
    print()
    
    generate_dataset()
    
    print("\n" + "="*60)
    print("Dataset ready for upload to Hugging Face, GitHub, and Kaggle!")
    print("="*60)
