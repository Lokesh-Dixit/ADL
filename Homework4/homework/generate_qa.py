import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)

    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        return np.array(pil_image)

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        color = (255, 0, 0) if track_id == 0 else COLORS.get(class_id, (255, 255, 255))
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    with open(info_path) as f:
        info = json.load(f)

    kart_names = info["karts"]
    detections = info["detections"][view_index]
    valid_karts = []

    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        if int(class_id) != 1 or int(track_id) >= len(kart_names):
            continue

        w, h = x2 - x1, y2 - y1
        if w < min_box_size or h < min_box_size:
            continue

        x_center = (x1 + x2) / 2 * (img_width / ORIGINAL_WIDTH)
        y_center = (y1 + y2) / 2 * (img_height / ORIGINAL_HEIGHT)

        if not (0 <= x_center <= img_width and 0 <= y_center <= img_height):
            continue

        valid_karts.append({
            "instance_id": int(track_id),
            "kart_name": kart_names[int(track_id)],
            "center": (x_center, y_center),
        })

    img_center = (img_width / 2, img_height / 2)
    for kart in valid_karts:
        kart["distance_to_center"] = np.linalg.norm(np.array(kart["center"]) - np.array(img_center))

    if valid_karts:
        ego = min(valid_karts, key=lambda x: x["distance_to_center"])
        for kart in valid_karts:
            kart["is_center_kart"] = (kart == ego)

    return valid_karts


def extract_track_info(info_path: str) -> str:
    with open(info_path) as f:
        info = json.load(f)
    return info.get("track", "unknown track")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    qa_pairs = []
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    if not karts:
        return []

    ego = next((k for k in karts if k.get("is_center_kart")), None)
    if not ego:
        return []

    ex, ey = ego["center"]

    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego["kart_name"]
    })
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track
    })

    for kart in karts:
        if kart["kart_name"] == ego["kart_name"]:
            continue

        x, y = kart["center"]
        horiz = "right" if x > ex else "left"
        vert = "front" if y < ey else "behind"

        qa_pairs.append({
            "question": f"Where is {kart['kart_name']} relative to the ego car?",
            "answer": f"{vert} and {horiz}"
        })

    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][0] < ex))
    })
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][0] > ex))
    })
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][1] < ey))
    })
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][1] > ey))
    })

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(info_file, view_index)

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def generate_dataset(info_dir: str, output_path: str):
    """
    Generate a dataset of QA pairs for all views in all info.json files.

    Args:
        info_dir: Path to the folder containing *_info.json files
        output_path: Path to store the output dataset JSON
    """
    from tqdm import tqdm

    info_dir = Path(info_dir)
    qa_dataset = []

    # Iterate over all *_info.json files
    for info_file in tqdm(sorted(info_dir.glob("*_info.json"))):
        for view_index in range(10):
            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                for qa in qa_pairs:
                    qa["image_file"] = f"{info_file.stem.replace('_info', '')}_{view_index:02d}_im.jpg"
                    qa_dataset.append(qa)
            except Exception as e:
                print(f"Skipping {info_file.name} view {view_index} due to error: {e}")
                continue

    with open(output_path, "w") as f:
        json.dump(qa_dataset, f, indent=2)

    print(f"\n✅ Generated {len(qa_dataset)} QA pairs and saved to {output_path}")


def main():
    fire.Fire(
    {"check": check_qa_pairs,
    "generate_dataset": generate_dataset})

if __name__ == "__main__":
    main()
