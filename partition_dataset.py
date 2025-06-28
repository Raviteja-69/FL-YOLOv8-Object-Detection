import os
import shutil
import random
import argparse
import yaml
from collections import defaultdict
import numpy as np  # <-- Add numpy for Dirichlet

# Helper to read YOLO label file and get all classes present in an image
def get_classes_from_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return set(int(line.split()[0]) for line in lines if line.strip())

def parse_args():
    parser = argparse.ArgumentParser(description="Partition YOLO dataset for federated learning.")
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root (with train/, valid/, test/)')
    parser.add_argument('--output_dir', type=str, default='partitioned_clients', help='Where to save client folders')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of clients')
    parser.add_argument('--split_type', type=str, choices=['iid', 'non-iid-class', 'custom', 'non-iid-dirichlet'], default='iid', help='Split type')
    parser.add_argument('--custom_proportions', type=str, default=None, help='YAML file with custom proportions (for split_type=custom)')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, help='Dirichlet alpha parameter (for split_type=non-iid-dirichlet)')
    return parser.parse_args()

def copy_files(image_list, src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    for img_path in image_list:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(src_lbl_dir, base + '.txt')
        shutil.copy2(img_path, os.path.join(dst_img_dir, os.path.basename(img_path)))
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(dst_lbl_dir, os.path.basename(label_path)))

def write_data_yaml(client_dir, nc, names, rel_train, rel_val, rel_test):
    data = {
        'train': rel_train,
        'valid': rel_val,
        'test': rel_test,
        'nc': nc,
        'names': names
    }
    with open(os.path.join(client_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f)

def partition_iid(train_images, num_clients):
    random.shuffle(train_images)
    split = len(train_images) // num_clients
    return [train_images[i*split:(i+1)*split] if i < num_clients-1 else train_images[i*split:] for i in range(num_clients)]

def partition_non_iid_class(train_images, label_dir, num_clients, nc):
    # Assign each class to a client (round robin)
    class_to_client = {c: c % num_clients for c in range(nc)}
    client_imgs = [[] for _ in range(num_clients)]
    for img_path in train_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if not os.path.exists(label_path):
            continue
        classes = get_classes_from_label(label_path)
        if not classes:
            continue
        # Assign image to the client of its first class (could be improved)
        client = class_to_client[list(classes)[0]]
        client_imgs[client].append(img_path)
    # Fill up to balance
    all_imgs = set(train_images)
    assigned = set(sum(client_imgs, []))
    leftovers = list(all_imgs - assigned)
    for i, img in enumerate(leftovers):
        client_imgs[i % num_clients].append(img)
    return client_imgs

def partition_custom(train_images, label_dir, num_clients, nc, proportions):
    # proportions: dict[class_id][client_id] = fraction (should sum to 1 for each class)
    class_to_imgs = defaultdict(list)
    for img_path in train_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if not os.path.exists(label_path):
            continue
        classes = get_classes_from_label(label_path)
        for c in classes:
            class_to_imgs[c].append(img_path)
    client_imgs = [[] for _ in range(num_clients)]
    for c in range(nc):
        imgs = class_to_imgs[c]
        random.shuffle(imgs)
        start = 0
        for client_id in range(num_clients):
            frac = proportions.get(str(c), [1/num_clients]*num_clients)[client_id]
            n = int(frac * len(imgs))
            client_imgs[client_id].extend(imgs[start:start+n])
            start += n
    # Remove duplicates
    for i in range(num_clients):
        client_imgs[i] = list(set(client_imgs[i]))
    return client_imgs

def partition_non_iid_dirichlet(train_images, label_dir, num_clients, nc, alpha=0.5):
    """
    Dirichlet-based non-IID partitioning. Each class's images are distributed to clients
    according to proportions drawn from a Dirichlet distribution (parameter alpha).
    Lower alpha = more non-IID, higher alpha = more IID.
    """
    # 1. Group images by class
    class_to_imgs = defaultdict(list)
    for img_path in train_images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + '.txt')
        if not os.path.exists(label_path):
            continue
        classes = get_classes_from_label(label_path)
        for c in classes:
            class_to_imgs[c].append(img_path)
    # 2. For each class, sample Dirichlet proportions and assign images
    client_imgs = [[] for _ in range(num_clients)]
    for c in range(nc):
        imgs = class_to_imgs[c]
        if not imgs:
            continue
        np.random.shuffle(imgs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Compute split indices
        split_indices = (np.cumsum(proportions) * len(imgs)).astype(int)
        prev = 0
        for client_id, idx in enumerate(split_indices):
            client_imgs[client_id].extend(imgs[prev:idx])
            prev = idx
    # Remove duplicates (images with multiple classes)
    for i in range(num_clients):
        client_imgs[i] = list(set(client_imgs[i]))
    return client_imgs

def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)  # For reproducibility
    # Load global data.yaml for class info
    with open(os.path.join(args.dataset_root, 'data.yaml'), 'r') as f:
        data_yaml = yaml.safe_load(f)
    nc = data_yaml['nc']
    names = data_yaml['names']
    # Gather all train images
    train_img_dir = os.path.join(args.dataset_root, 'train', 'images')
    train_lbl_dir = os.path.join(args.dataset_root, 'train', 'labels')
    train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    # Partition
    if args.split_type == 'iid':
        client_splits = partition_iid(train_images, args.num_clients)
    elif args.split_type == 'non-iid-class':
        client_splits = partition_non_iid_class(train_images, train_lbl_dir, args.num_clients, nc)
    elif args.split_type == 'custom':
        if args.custom_proportions:
            with open(args.custom_proportions, 'r') as f:
                proportions = yaml.safe_load(f)
        else:
            # Example: class 0 is 80% client0, 20% client1; class 1 is 20% client0, 80% client1
            proportions = {str(c): [1/args.num_clients]*args.num_clients for c in range(nc)}
        client_splits = partition_custom(train_images, train_lbl_dir, args.num_clients, nc, proportions)
    elif args.split_type == 'non-iid-dirichlet':
        client_splits = partition_non_iid_dirichlet(train_images, train_lbl_dir, args.num_clients, nc, alpha=args.dirichlet_alpha)
    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    # Copy valid and test sets globally
    valid_img_dir = os.path.join(args.dataset_root, 'valid', 'images')
    valid_lbl_dir = os.path.join(args.dataset_root, 'valid', 'labels')
    test_img_dir = os.path.join(args.dataset_root, 'test', 'images')
    test_lbl_dir = os.path.join(args.dataset_root, 'test', 'labels')
    for i, imgs in enumerate(client_splits):
        client_dir = os.path.join(args.output_dir, f'client{i+1}')
        train_img_out = os.path.join(client_dir, 'train', 'images')
        train_lbl_out = os.path.join(client_dir, 'train', 'labels')
        copy_files(imgs, train_img_dir, train_lbl_dir, train_img_out, train_lbl_out)
        # Copy valid and test sets (global)
        valid_img_out = os.path.join(client_dir, 'valid', 'images')
        valid_lbl_out = os.path.join(client_dir, 'valid', 'labels')
        test_img_out = os.path.join(client_dir, 'test', 'images')
        test_lbl_out = os.path.join(client_dir, 'test', 'labels')
        copy_files([os.path.join(valid_img_dir, f) for f in os.listdir(valid_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))], valid_img_dir, valid_lbl_dir, valid_img_out, valid_lbl_out)
        copy_files([os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))], test_img_dir, test_lbl_dir, test_img_out, test_lbl_out)
        # Write data.yaml for client
        write_data_yaml(client_dir, nc, names, 'train/images', 'valid/images', 'test/images')
    print(f"Partitioned data for {args.num_clients} clients in {args.output_dir}/")

if __name__ == '__main__':
    main() 