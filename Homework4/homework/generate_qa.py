import json
from pathlib import Path
import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
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

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400

def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    img_height, img_width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with open(info_path) as f:
        info = json.load(f)

    _, view_index = extract_frame_info(image_path)
    if view_index >= len(info["detections"]):
        return image

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    detections = info["detections"][view_index]

    for det in detections:
        class_id, track_id, x1, y1, x2, y2 = det
        if class_id != 1:
            continue
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
            continue
        if x2 < 0 or x1 > img_width or y2 < 0 or y1 > img_height:
            continue
        color = (255, 0, 0) if track_id == 0 else COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    with open(info_path) as f:
        info = json.load(f)

    karts = []
    kart_names = info["karts"]
    detections = info["detections"][view_index]

    for d in detections:
        class_id, track_id, x1, y1, x2, y2 = d
        if int(class_id) != 1:
            continue
        if int(track_id) >= len(kart_names):
            continue
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue

        x_center = (x1 + x2) / 2 * (img_width / ORIGINAL_WIDTH)
        y_center = (y1 + y2) / 2 * (img_height / ORIGINAL_HEIGHT)

        if not (0 <= x_center <= img_width and 0 <= y_center <= img_height):
            continue

        karts.append({
            "instance_id": int(track_id),
            "kart_name": kart_names[int(track_id)],
            "center": (x_center, y_center),
        })

    if not karts:
        return []

    img_center = (img_width / 2, img_height / 2)
    for kart in karts:
        kart["dist"] = np.linalg.norm(np.array(kart["center"]) - np.array(img_center))

    ego = min(karts, key=lambda k: k["dist"])
    for kart in karts:
        kart["is_center_kart"] = (kart == ego)

    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        return json.load(f).get("track", "unknown")


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa_pairs = []
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    if not karts:
        return []

    ego = next((k for k in karts if k.get("is_center_kart")), None)
    if not ego:
        return []

    ex, ey = ego["center"]
    threshold = 1.5     # threshold for tie-breaking

    # Q1. Find ego kart name.
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego["kart_name"]
    })

    # Q2. Count the number of karts.
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })

    # Q3. Find the Track name.
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track
    })

    # Q4. Find the Kart position of each kart.
    for kart in karts:
        if kart["kart_name"] == ego["kart_name"]:
            continue

        x, y = kart["center"]
        horiz = "right" if x > ex + threshold else "left" if x < ex - threshold else "same"
        vert = "front" if y < ey - threshold else "back" if y > ey + threshold else "same"

        # Prefer combined question if position is diagonal
        if horiz != "same" and vert != "same":
            qa_pairs.append({
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": f"{vert} and {horiz}"
            })
        else:
            if horiz != "same":
                qa_pairs.append({
                    "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                    "answer": horiz
                })
            if vert != "same":
                qa_pairs.append({
                    "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                    "answer": vert
                })

    # Q5. Count number of karts and their position relative to ego kart
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][0] < ex - threshold))
    })
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][0] > ex + threshold))
    })
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][1] < ey - threshold))
    })
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(sum(1 for k in karts if k["center"][1] > ey + threshold))
    })

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated = draw_detections(str(image_file), str(info_path))
    plt.figure(figsize=(10, 6))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(str(info_path), view_index)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 40)


def generate_dataset(info_dir: str, output_path: str):
    """
    This function is created to generate dataset based on 'generate_qa_pairs' 

    Args:
        info_file: Path to the info.json file
        output_path: Path to the my_balanced_qa_pair.json file in 'train' fodler under 'data'
    """
    info_dir = Path(info_dir)
    qa_dataset = []

    for info_file in tqdm(sorted(info_dir.glob("*_info.json"))):
        for view_index in range(10):
            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                for qa in qa_pairs:
                    img_base = f"{info_file.stem.replace('_info', '')}_{view_index:02d}_im.jpg"
                    qa["image_file"] = f"train/{img_base}" 
                    qa_dataset.append(qa)
            except Exception as e:
                print(f" Skipping {info_file.name} view {view_index} due to error: {e}")
                continue

    with open(output_path, "w") as f:
        json.dump(qa_dataset, f, indent=2)

    print(f"\n Generated {len(qa_dataset)} QA pairs and saved to {output_path}")


def main():
    fire.Fire({"check": check_qa_pairs, "generate_dataset": generate_dataset})


if __name__ == "__main__":
    main()
