# Revised version of the provided code:
# - All comments are translated into English.
# - Additional explanatory comments are added where helpful.
# - Imports are cleaned and grouped logically.

import os
import re
import pickle
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2
import torch
import faiss

from torchvision import models, transforms
from torchvision.models import VGG16_Weights

from skimage.feature import hog
from skimage.color import rgb2hsv
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity


# Device setup for CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:4].to(device).eval()

transform_cnn = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


def extract_avg_rgb_feature(image_path: str) -> Tuple[str, np.ndarray]:
    image_id = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB')
    array = np.array(image).reshape(-1, 3)
    avg_rgb = np.mean(array, axis=0)
    return image_id, avg_rgb


def extract_avg_hsv_feature(image_path: str) -> Tuple[str, np.ndarray]:
    image_id = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB')
    array = np.array(image) / 255.0
    hsv = rgb2hsv(array)
    avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    return image_id, avg_hsv


def extract_color_hist_feature(image_path: str) -> Tuple[str, np.ndarray]:
    image_id = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB').resize((64, 64))
    array = np.array(image)
    hist_r, _ = np.histogram(array[:, :, 0], bins=8, range=(0, 256), density=True)
    hist_g, _ = np.histogram(array[:, :, 1], bins=8, range=(0, 256), density=True)
    hist_b, _ = np.histogram(array[:, :, 2], bins=8, range=(0, 256), density=True)
    color_hist = np.concatenate([hist_r, hist_g, hist_b])
    return image_id, color_hist


def extract_hog_feature(image_path: str) -> Tuple[str, np.ndarray]:
    image_id = os.path.basename(image_path)
    image = Image.open(image_path).convert('L').resize((64, 64))
    array = np.array(image)
    hog_feat = hog(array, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return image_id, hog_feat


def extract_cnn_feature(image_path: str) -> Tuple[str, np.ndarray]:
    """
    Extract low-dimensional CNN feature using shallow VGG16 + Global Average Pooling
    """
    image_id = os.path.basename(image_path)
    image = Image.open(image_path).convert('RGB')
    x = transform_cnn(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = vgg16(x)  # shape: (1, C, H, W)
        pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))  # → shape: (1, C, 1, 1)
        vector = pooled.view(-1).cpu().numpy().astype(np.float32)  # → shape: (C,)

    return image_id, vector



def extract_edge_density_feature(image_path: str) -> Tuple[str, np.ndarray]:
    image_id = os.path.basename(image_path)
    image = Image.open(image_path).convert('L').resize((64, 64))
    img_array = np.array(image)
    edges = cv2.Canny(img_array, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size  # Ratio of edge pixels
    return image_id, np.array([edge_density])


def extract_combined_feature(
    image_path: str,
    w_rgb: float = 0.0,
    w_color: float = 0.0,
    w_hog: float = 0.0,
    w_cnn: float = 0.0,
    w_edge: float = 0.0
) -> Tuple[str, np.ndarray]:
    """
    Combine multiple feature types with optional weights.
    Only compute features with positive weights to reduce computation.
    """
    image_id = os.path.basename(image_path)
    feature_parts = []

    if w_rgb > 0:
        _, rgb = extract_avg_rgb_feature(image_path)
        rgb_norm = normalize(rgb.reshape(1, -1))[0] * w_rgb
        feature_parts.append(rgb_norm)

    if w_color > 0:
        _, color = extract_color_hist_feature(image_path)
        color_norm = normalize(color.reshape(1, -1))[0] * w_color
        feature_parts.append(color_norm)

    if w_hog > 0:
        _, hog_feat = extract_hog_feature(image_path)
        hog_norm = normalize(hog_feat.reshape(1, -1))[0] * w_hog
        feature_parts.append(hog_norm)

    if w_cnn > 0:
        _, cnn_feat = extract_cnn_feature(image_path)
        cnn_norm = normalize(cnn_feat.reshape(1, -1))[0] * w_cnn
        feature_parts.append(cnn_norm)

    if w_edge > 0:
        _, edge_feat = extract_edge_density_feature(image_path)
        edge_norm = normalize(edge_feat.reshape(1, -1))[0] * w_edge
        feature_parts.append(edge_norm)

    if not feature_parts:
        raise ValueError("At least one feature weight must be greater than 0.")

    combined = np.concatenate(feature_parts)
    final_feature = normalize(combined.reshape(1, -1))[0]
    return image_id, final_feature


def extract_feature(image_path: str, method: str, **kwargs) -> Tuple[str, np.ndarray]:
    """
    Dispatch to appropriate feature extraction function by name.
    """
    if method == "avg_rgb":
        return extract_avg_rgb_feature(image_path)
    elif method == "avg_hsv":
        return extract_avg_hsv_feature(image_path)
    elif method == "cnn":
        return extract_cnn_feature(image_path)
    elif method == "color":
        return extract_color_hist_feature(image_path)
    elif method == "hog":
        return extract_hog_feature(image_path)
    elif method == "edge":
        return extract_edge_density_feature(image_path)
    elif method == "combined":
        return extract_combined_feature(image_path, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

# Revised version of the second half of your code:
# - Translated all Chinese comments into English.
# - Added clarifying comments where helpful.
# - Cleaned up import statements.

def extract_features_from_folder(folder_path: str, method: str, output_path: str, **kwargs) -> None:
    """
    Extract features from all images in a given folder and save to a pickle file.
    """
    feature_dict = {}
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(image_files, desc=f"Extracting features with {method}"):
        image_path = os.path.join(folder_path, filename)
        try:
            image_id, feature = extract_feature(image_path, method, **kwargs)
            feature_dict[image_id] = feature
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    with open(output_path, 'wb') as f:
        pickle.dump(feature_dict, f)


def split_image_into_tiles(image_path: str, tile_size: int) -> List[Tuple[str, np.ndarray]]:
    """
    Split an image into square tiles of a given size.
    """
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    h, w, _ = img_array.shape
    tiles = []
    for row, y in enumerate(range(0, h, tile_size)):
        for col, x in enumerate(range(0, w, tile_size)):
            tile = img_array[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                tile_id = f"{image_id}_tile_{row:03d}_{col:03d}"
                tiles.append((tile_id, tile))
    return tiles


def extract_tile_features(image_path: str, tile_size: int, method: str, output_path: str, **kwargs) -> None:
    """
    Extract features from tiles of an input image and save them.
    """
    tiles = split_image_into_tiles(image_path, tile_size)
    feature_dict = {}
    for tile_id, tile_array in tqdm(tiles, desc=f"Extracting tile features ({method})"):
        try:
            tile_img = Image.fromarray(tile_array)
            temp_path = "__temp_tile.jpg"
            tile_img.save(temp_path)
            _, feature = extract_feature(temp_path, method, **kwargs)
            feature_dict[tile_id] = feature
            os.remove(temp_path)
        except Exception as e:
            print(f"Error processing {tile_id}: {e}")
    with open(output_path, 'wb') as f:
        pickle.dump(feature_dict, f)


def match_tiles_to_gallery_kmeans(tile_feature_path: str, gallery_feature_path: str, output_match_path: str, n_clusters: int = 100, use_cosine: bool = True) -> None:
    """
    Match tiles to gallery images using KMeans clustering.
    """
    with open(tile_feature_path, 'rb') as f:
        tile_features = pickle.load(f)
    with open(gallery_feature_path, 'rb') as f:
        gallery_features = pickle.load(f)

    tile_ids = list(tile_features.keys())
    gallery_ids = list(gallery_features.keys())
    X_tile = np.stack([tile_features[tid] for tid in tile_ids])
    X_gallery = np.stack([gallery_features[gid] for gid in gallery_ids])

    if use_cosine:
        X_tile = X_tile / np.linalg.norm(X_tile, axis=1, keepdims=True)
        X_gallery = X_gallery / np.linalg.norm(X_gallery, axis=1, keepdims=True)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1000,
        n_init=1,
        max_iter=100,
        verbose=1  # Print clustering progress
    )

    gallery_labels = kmeans.fit_predict(X_gallery)
    match_dict = {}

    for i, tile_vec in tqdm(enumerate(X_tile), total=len(X_tile), desc="KMeans matching"):
        if use_cosine:
            sims = cosine_similarity(tile_vec.reshape(1, -1), kmeans.cluster_centers_)[0]
            best_cluster = np.argmax(sims)
        else:
            dists = np.linalg.norm(kmeans.cluster_centers_ - tile_vec, axis=1)
            best_cluster = np.argmin(dists)

        candidate_idx = [j for j, label in enumerate(gallery_labels) if label == best_cluster]
        candidate_vecs = X_gallery[candidate_idx]
        sims = cosine_similarity(tile_vec.reshape(1, -1), candidate_vecs)[0]
        best_local_idx = np.argmax(sims)
        best_match = gallery_ids[candidate_idx[best_local_idx]]
        match_dict[tile_ids[i]] = best_match

    with open(output_match_path, 'wb') as f:
        pickle.dump(match_dict, f)


def match_tiles_to_gallery_faiss(tile_feature_path: str, gallery_feature_path: str, output_match_path: str) -> None:
    """
    Match tiles to gallery images using FAISS (cosine similarity, efficient for large-scale retrieval).
    """
    with open(tile_feature_path, 'rb') as f:
        tile_features: Dict[str, np.ndarray] = pickle.load(f)
    with open(gallery_feature_path, 'rb') as f:
        gallery_features: Dict[str, np.ndarray] = pickle.load(f)

    tile_ids = list(tile_features.keys())
    gallery_ids = list(gallery_features.keys())

    tile_matrix = np.stack([tile_features[tid] for tid in tile_ids])
    gallery_matrix = np.stack([gallery_features[gid] for gid in gallery_ids])

    tile_matrix = tile_matrix / np.linalg.norm(tile_matrix, axis=1, keepdims=True)
    gallery_matrix = gallery_matrix / np.linalg.norm(gallery_matrix, axis=1, keepdims=True)

    dim = gallery_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # Use inner product (same as cosine similarity after normalization)
    index.add(gallery_matrix)

    _, I = index.search(tile_matrix, 1)
    match_dict = {tile_ids[i]: gallery_ids[I[i][0]] for i in range(len(tile_ids))}

    with open(output_match_path, 'wb') as f:
        pickle.dump(match_dict, f)


def reconstruct_mosaic_image(match_dict_path: str, gallery_folder: str, tile_size: int, output_image_path: str, background_color=(255, 255, 255)) -> None:
    """
    Reconstruct the final mosaic image from tile matches.
    """
    with open(match_dict_path, 'rb') as f:
        match_dict = pickle.load(f)

    tile_ids = sorted(match_dict.keys())
    coords = []

    for tid in tile_ids:
        match = re.search(r'_tile_(\d+)_(\d+)', tid)
        if match:
            y, x = int(match.group(1)), int(match.group(2))
            coords.append((x, y))
        else:
            raise ValueError(f"Cannot extract (row,col) from tile_id: {tid}")

    max_x = max(c[0] for c in coords)
    max_y = max(c[1] for c in coords)

    canvas = Image.new('RGB', ((max_x + 1) * tile_size, (max_y + 1) * tile_size), color=background_color)

    for tid, (x_idx, y_idx) in tqdm(zip(tile_ids, coords), desc="Reconstructing mosaic", total=len(tile_ids)):
        matched_img_name = match_dict[tid]
        matched_img_path = os.path.join(gallery_folder, matched_img_name)
        if not os.path.exists(matched_img_path):
            print(f"Warning: {matched_img_path} not found.")
            continue
        tile_img = Image.open(matched_img_path).resize((tile_size, tile_size))
        canvas.paste(tile_img, (x_idx * tile_size, y_idx * tile_size))

    canvas.save(output_image_path)
