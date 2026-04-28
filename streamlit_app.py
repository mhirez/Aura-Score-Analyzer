import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from html import escape
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

try:
    import tf_keras
except Exception:
    tf_keras = None


APP_DIR = Path(__file__).resolve().parent
ASSETS_DIR = APP_DIR / "assets"
MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "aura_score_analyzer_models"
RUN_DIR = Path(tempfile.gettempdir()) / "aura_score_analyzer_run"

NORMALIZATION_MODE = "teachable_machine"
DEFAULT_INPUT_SIZE = (224, 224)
EQUAL_MODEL_WEIGHT = 33.3333
AURA_EQUATION = "final_aura_score = 100 * ((P(closed_arms) + P(serious_face) + P(glasses)) / 3)"

MODEL_ZIPS = {
    "closed_arms": APP_DIR / "Feature Detection Models" / "crossed_open_arms.zip",
    "serious_face": APP_DIR / "Feature Detection Models" / "smile_not_smile_model.zip",
    "glasses": APP_DIR / "Feature Detection Models" / "glasses_no_glasses.zip",
    "general_aura": APP_DIR / "general_model.zip",
}

MODEL_DIRS = {
    "closed_arms": MODEL_CACHE_DIR / "closed_arms_model",
    "serious_face": MODEL_CACHE_DIR / "serious_face_model",
    "glasses": MODEL_CACHE_DIR / "glasses_model",
    "general_aura": MODEL_CACHE_DIR / "general_model",
}

TARGET_CLASS_NAMES = {
    "closed_arms": [
        "closed arms",
        "closed_arms",
        "closed",
        "closed hands",
        "closed_hands",
        "crossed arms",
        "crossed_arms",
        "crossed",
        "crossed hands",
        "crossed_hands",
    ],
    "serious_face": [
        "serious face",
        "serious_face",
        "serious",
        "non_smile",
        "not_smile",
        "not smile",
        "no_smile",
        "not smiling",
        "neutral",
    ],
    "glasses": ["glasses", "wearing_glasses", "with_glasses", "has_glasses"],
    "general_aura": ["aura", "has_aura", "has aura", "aura_score", "aura score"],
}

NEGATIVE_CLASS_NAMES = {
    "closed_arms": ["open arms", "open_arms", "open hands", "open_hands"],
    "serious_face": [
        "not serious face",
        "not_serious_face",
        "not serious",
        "not_serious",
        "smile",
        "smiling",
    ],
    "glasses": ["no_glasses", "no glasses", "without_glasses", "without glasses"],
    "general_aura": ["not_aura", "not aura", "no_aura", "no aura", "without_aura", "without aura"],
}

MODEL_SPECS = [
    {
        "key": "closed_arms",
        "display_name": "Closed arms",
        "image_source": "full_image",
        "reason_subject": "closed arms",
    },
    {
        "key": "serious_face",
        "display_name": "Serious face",
        "image_source": "face_crop",
        "reason_subject": "a serious face",
    },
    {
        "key": "glasses",
        "display_name": "Glasses",
        "image_source": "face_crop",
        "reason_subject": "glasses",
    },
]

GENERAL_MODEL_SPEC = {
    "key": "general_aura",
    "display_name": "General black-box Aura model",
    "image_source": "full_image",
    "reason_subject": "Aura",
}

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = Image.LANCZOS


class CompatibleDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("groups", None)
        return super().from_config(config)


class SavedModelPredictor:
    def __init__(self, saved_model_dir):
        self.saved_model_dir = str(saved_model_dir)
        self.loaded = tf.saved_model.load(self.saved_model_dir)

        if "serving_default" in self.loaded.signatures:
            self.signature = self.loaded.signatures["serving_default"]
        else:
            signature_keys = list(self.loaded.signatures.keys())
            if not signature_keys:
                raise ValueError("SavedModel has no callable signatures.")
            self.signature = self.loaded.signatures[signature_keys[0]]

        positional_inputs, keyword_inputs = self.signature.structured_input_signature
        if keyword_inputs:
            self.input_name, self.input_spec = next(iter(keyword_inputs.items()))
            self.uses_keyword_input = True
        elif positional_inputs:
            self.input_name = None
            self.input_spec = positional_inputs[0]
            self.uses_keyword_input = False
        else:
            raise ValueError("SavedModel signature does not expose an input tensor.")

        try:
            self.input_shape = tuple(self.input_spec.shape.as_list())
        except Exception:
            self.input_shape = None

    def predict(self, batch, verbose=0):
        input_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        if self.uses_keyword_input:
            outputs = self.signature(**{self.input_name: input_tensor})
        else:
            outputs = self.signature(input_tensor)

        if isinstance(outputs, dict):
            output_tensor = next(iter(outputs.values()))
        else:
            output_tensor = outputs
        return output_tensor.numpy()


def safe_extract_zip(zip_path, output_dir):
    zip_path = Path(zip_path).resolve()
    output_dir = Path(output_dir).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = (output_dir / member.filename).resolve()
            if not (member_path == output_dir or str(member_path).startswith(str(output_dir) + os.sep)):
                raise ValueError(f"Unsafe path found inside zip file: {member.filename}")
        zip_ref.extractall(output_dir)

    return output_dir


def find_model_file(model_dir):
    model_dir = Path(model_dir)
    search_patterns = ["**/*.keras", "**/keras_model.h5", "**/model.h5", "**/*.h5"]

    for pattern in search_patterns:
        matches = sorted(model_dir.glob(pattern), key=lambda path: (len(str(path)), str(path).lower()))
        if matches:
            return matches[0]

    saved_model_dirs = sorted(
        {path.parent for path in model_dir.rglob("saved_model.pb")},
        key=lambda path: (len(str(path)), str(path).lower()),
    )
    if saved_model_dirs:
        return saved_model_dirs[0]

    model_json_files = sorted(model_dir.rglob("model.json"), key=lambda path: (len(str(path)), str(path).lower()))
    if model_json_files:
        return model_json_files[0]

    raise FileNotFoundError(f"No supported model file was found inside: {model_dir}")


def clean_label(label_text):
    label_text = str(label_text).strip()
    label_text = re.sub(r"^\s*\d+\s*[:.)-]?\s*", "", label_text)
    return label_text.strip()


def read_labels(model_dir):
    model_dir = Path(model_dir)

    for label_file in sorted(model_dir.rglob("labels.txt"), key=lambda path: (len(str(path)), str(path).lower())):
        labels = []
        with label_file.open("r", encoding="utf-8") as file:
            for line in file:
                cleaned = clean_label(line)
                if cleaned:
                    labels.append(cleaned)
        if labels:
            return labels

    for metadata_file in sorted(model_dir.rglob("metadata.json"), key=lambda path: (len(str(path)), str(path).lower())):
        with metadata_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        raw_labels = metadata.get("labels") or metadata.get("modelLabels") or metadata.get("classes")
        if isinstance(raw_labels, list):
            labels = []
            for item in raw_labels:
                if isinstance(item, dict):
                    value = item.get("name") or item.get("label") or item.get("displayName")
                else:
                    value = item
                cleaned = clean_label(value)
                if cleaned:
                    labels.append(cleaned)
            if labels:
                return labels

    return []


def load_tfjs_layers_model(model_json_path):
    model_json_path = Path(model_json_path)
    with model_json_path.open("r", encoding="utf-8") as file:
        model_json = json.load(file)

    topology = model_json.get("modelTopology")
    if not topology:
        raise ValueError("The TensorFlow.js model.json file does not contain modelTopology.")

    topology_json = json.dumps(topology)
    load_errors = []
    loaders = [tf.keras.models]
    if tf_keras is not None:
        loaders.insert(0, tf_keras.models)

    model = None
    for loader in loaders:
        try:
            model = loader.model_from_json(
                topology_json,
                custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
            )
            break
        except Exception as error:
            load_errors.append(str(error))

    if model is None:
        raise RuntimeError("Could not rebuild the TensorFlow.js model architecture: " + " | ".join(load_errors))

    weight_arrays, weight_names = read_tfjs_weight_arrays(model_json_path, model_json)
    set_tfjs_weights(model, weight_arrays, weight_names)
    return model


def read_tfjs_weight_arrays(model_json_path, model_json):
    model_dir = Path(model_json_path).parent
    weight_arrays = []
    weight_names = []
    dtype_map = {
        "float32": np.float32,
        "int32": np.int32,
        "bool": np.bool_,
    }

    for group in model_json.get("weightsManifest", []):
        binary_parts = []
        for relative_path in group.get("paths", []):
            binary_parts.append((model_dir / relative_path).read_bytes())
        binary_data = b"".join(binary_parts)
        offset = 0

        for weight_info in group.get("weights", []):
            dtype_name = weight_info.get("dtype", "float32")
            if dtype_name not in dtype_map:
                raise ValueError(f"Unsupported TensorFlow.js weight dtype: {dtype_name}")

            shape = tuple(int(dim) for dim in weight_info.get("shape", []))
            count = int(np.prod(shape)) if shape else 1
            dtype = np.dtype(dtype_map[dtype_name])
            byte_count = count * dtype.itemsize
            chunk = binary_data[offset : offset + byte_count]
            if len(chunk) != byte_count:
                raise ValueError(f"Weight data ended early while reading {weight_info.get('name')}.")

            array = np.frombuffer(chunk, dtype=dtype, count=count).reshape(shape).copy()
            weight_arrays.append(array)
            weight_names.append(str(weight_info.get("name", "")))
            offset += byte_count

    if not weight_arrays:
        raise ValueError("The TensorFlow.js model does not contain any readable weights.")

    return weight_arrays, weight_names


def set_tfjs_weights(model, weight_arrays, weight_names):
    current_weights = model.get_weights()
    if len(current_weights) == len(weight_arrays):
        shapes_match = all(tuple(current.shape) == tuple(new.shape) for current, new in zip(current_weights, weight_arrays))
        if shapes_match:
            model.set_weights(weight_arrays)
            return

    used_indexes = set()
    mapped_arrays = []
    for variable in model.weights:
        variable_name = variable.name.split(":")[0]
        try:
            variable_shape = tuple(variable.shape.as_list())
        except AttributeError:
            variable_shape = tuple(variable.shape)
        match_index = None

        for index, (weight_name, weight_array) in enumerate(zip(weight_names, weight_arrays)):
            if index in used_indexes:
                continue
            if tuple(weight_array.shape) != variable_shape:
                continue
            if variable_name.endswith(weight_name) or weight_name.endswith(variable_name):
                match_index = index
                break

        if match_index is None:
            raise ValueError(f"Could not match TensorFlow.js weight for Keras variable: {variable_name}")

        used_indexes.add(match_index)
        mapped_arrays.append(weight_arrays[match_index])

    model.set_weights(mapped_arrays)


def load_keras_model(model_path):
    model_path = Path(model_path)
    first_error = None
    load_errors = []

    is_h5_model = model_path.suffix.lower() in {".h5", ".hdf5"}
    is_tfjs_model = model_path.name.lower() == "model.json"

    if is_h5_model and tf_keras is not None:
        try:
            return tf_keras.models.load_model(str(model_path), compile=False)
        except Exception as error:
            first_error = error
            load_errors.append(f"tf_keras legacy loader failed: {error}")

    if not is_tfjs_model:
        try:
            return tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as error:
            first_error = first_error or error
            load_errors.append(f"tf.keras loader failed: {error}")

    error_message = "\n".join(load_errors) if load_errors else str(first_error)
    if "DepthwiseConv2D" in error_message and "groups" in error_message:
        try:
            return tf.keras.models.load_model(
                str(model_path),
                compile=False,
                custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
            )
        except Exception as depthwise_error:
            first_error = first_error or depthwise_error
            load_errors.append(f"DepthwiseConv2D compatibility loader failed: {depthwise_error}")

    if model_path.is_dir() or (model_path / "saved_model.pb").exists():
        try:
            return SavedModelPredictor(model_path)
        except Exception as saved_model_error:
            first_error = first_error or saved_model_error
            load_errors.append(f"SavedModel loader failed: {saved_model_error}")

    if is_tfjs_model:
        try:
            return load_tfjs_layers_model(model_path)
        except Exception as tfjs_error:
            raise RuntimeError(
                "A TensorFlow.js model.json was found, but it could not be loaded by the built-in TFJS loader. "
                f"Original error: {tfjs_error}"
            ) from tfjs_error

    details = "\n\n".join(load_errors) if load_errors else str(first_error)
    raise RuntimeError(f"Could not load model at {model_path}. Loader details:\n{details}") from first_error


def get_input_size(model):
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]

    if shape is None and hasattr(model, "inputs") and model.inputs:
        shape = model.inputs[0].shape

    if hasattr(shape, "as_list"):
        shape = shape.as_list()

    try:
        dims = [None if dim is None else int(dim) for dim in list(shape)]
    except Exception:
        return DEFAULT_INPUT_SIZE

    height = None
    width = None
    if len(dims) == 4:
        if dims[-1] in (1, 3, 4):
            height, width = dims[1], dims[2]
        elif dims[1] in (1, 3, 4):
            height, width = dims[2], dims[3]
    elif len(dims) == 3:
        if dims[-1] in (1, 3, 4):
            height, width = dims[0], dims[1]
        elif dims[0] in (1, 3, 4):
            height, width = dims[1], dims[2]

    if height is None or width is None or height <= 0 or width <= 0:
        return DEFAULT_INPUT_SIZE

    return int(height), int(width)


def preprocess_image(image_path, model):
    height, width = get_input_size(model)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (width, height), RESAMPLE_FILTER)
    image_array = np.asarray(image).astype(np.float32)

    if NORMALIZATION_MODE == "teachable_machine":
        image_array = (image_array / 127.5) - 1.0
    elif NORMALIZATION_MODE == "zero_to_one":
        image_array = image_array / 255.0
    else:
        raise ValueError("NORMALIZATION_MODE must be teachable_machine or zero_to_one.")

    return np.expand_dims(image_array, axis=0)


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-float(value)))


def convert_to_probability_vector(raw_values, labels):
    values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("Model returned an empty prediction.")

    if values.size == 1 and len(labels) == 2:
        p = float(values[0])
        if p < 0.0 or p > 1.0:
            p = sigmoid(p)
        p = float(np.clip(p, 0.0, 1.0))
        return np.asarray([1.0 - p, p], dtype=np.float32)

    if values.size == 1:
        p = float(values[0])
        if p < 0.0 or p > 1.0:
            p = sigmoid(p)
        p = float(np.clip(p, 0.0, 1.0))
        return np.asarray([p], dtype=np.float32)

    values = values.astype(np.float64)
    values_sum = float(np.sum(values))
    values_are_probabilities = (
        np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.isclose(values_sum, 1.0, atol=1e-3)
    )

    if values_are_probabilities:
        probabilities = values
    else:
        shifted_values = values - np.max(values)
        exp_values = np.exp(shifted_values)
        probabilities = exp_values / np.sum(exp_values)

    return probabilities.astype(np.float32)


def align_labels(labels, probability_count):
    aligned = [clean_label(label) for label in labels if clean_label(label)]
    if len(aligned) < probability_count:
        aligned.extend([f"class_{index}" for index in range(len(aligned), probability_count)])
    elif len(aligned) > probability_count:
        aligned = aligned[:probability_count]
    return aligned


def predict_model(model, image_path, labels):
    batch = preprocess_image(image_path, model)
    raw_prediction = model.predict(batch, verbose=0)

    if isinstance(raw_prediction, (list, tuple)):
        raw_prediction = raw_prediction[0]

    raw_array = np.asarray(raw_prediction)
    raw_values = raw_array[0] if raw_array.ndim > 1 else raw_array

    probabilities = convert_to_probability_vector(raw_values, labels)
    aligned_labels = align_labels(labels, len(probabilities))
    probability_map = {
        aligned_labels[index]: float(probabilities[index])
        for index in range(len(probabilities))
    }

    return {
        "labels": aligned_labels,
        "raw_values": [float(value) for value in np.asarray(raw_values).reshape(-1)],
        "probabilities": [float(value) for value in probabilities],
        "probability_map": probability_map,
    }


def normalize_label_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"^\s*\d+\s*[:.)-]?\s*", "", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_target_label(labels, possible_names):
    normalized_labels = [normalize_label_text(label) for label in labels]
    normalized_targets = [normalize_label_text(name) for name in possible_names]

    for index, label in enumerate(normalized_labels):
        if label in normalized_targets:
            return index, labels[index]

    for index, label in enumerate(normalized_labels):
        for target in sorted(normalized_targets, key=len, reverse=True):
            if not target:
                continue
            target_word_count = len(target.split())
            if target_word_count >= 2 and target in label:
                return index, labels[index]
            if target_word_count == 1 and label.startswith(target + " "):
                return index, labels[index]

    return None, None


def get_target_probability(prediction, target_names, negative_names=None):
    labels = prediction["labels"]
    probabilities = prediction["probabilities"]

    target_index, target_label = find_target_label(labels, target_names)
    if target_index is not None:
        return float(probabilities[target_index]), target_label, "direct target label match"

    if negative_names:
        negative_index, negative_label = find_target_label(labels, negative_names)
        if negative_index is not None and len(probabilities) == 2:
            return float(1.0 - probabilities[negative_index]), f"not {negative_label}", "binary complement from negative label"

    raise ValueError(f"Could not identify target class from labels: {labels}")


def friendly_label_name(label):
    replacements = {
        "crossed_arms": "closed_arms",
        "crossed arms": "closed arms",
        "crossed_hands": "closed_hands",
        "crossed hands": "closed hands",
        "crossed": "closed",
        "non_smile": "serious_face",
        "not_smile": "serious_face",
        "not smile": "serious face",
        "no_smile": "serious_face",
        "not smiling": "serious face",
        "neutral": "serious_face",
        "smile": "not_serious_face",
        "smiling": "not_serious_face",
    }
    return replacements.get(str(label).lower(), str(label))


def grade_from_score(score):
    if score >= 80.0:
        return "Very strong aura"
    if score >= 60.0:
        return "Strong aura"
    if score >= 40.0:
        return "Medium aura"
    if score >= 20.0:
        return "Low aura"
    return "Very low aura"


def build_reason(reason_subject, probability):
    probability = float(probability)
    if probability >= 0.80:
        return f"The model strongly detected {reason_subject}, so this increased the aura score."
    if probability >= 0.60:
        return f"The model detected mostly {reason_subject}, so this moderately increased the aura score."
    if probability >= 0.40:
        return f"The model was uncertain about {reason_subject}, so this gave a medium contribution to the aura score."
    if probability >= 0.20:
        return f"The model weakly detected {reason_subject}, so this only slightly increased the aura score."
    return f"The model did not strongly detect {reason_subject}, so this added very little to the aura score."


def crop_face(image_path, output_path):
    pil_image = Image.open(image_path).convert("RGB")
    rgb_image = np.array(pil_image)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        pil_image.save(output_path)
        return output_path, False, "Warning: Haar Cascade could not be loaded. Full image fallback was used."

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        pil_image.save(output_path)
        return output_path, False, "Warning: no face was detected. Full image fallback was used."

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    margin_x = int(w * 0.35)
    margin_y = int(h * 0.35)

    image_height, image_width = rgb_image.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_width, x + w + margin_x)
    y2 = min(image_height, y + h + margin_y)

    face_crop = rgb_image[y1:y2, x1:x2]
    Image.fromarray(face_crop).save(output_path)
    return output_path, True, "Face crop succeeded. Serious face and glasses models used the face crop."


def compare_general_model_to_program(final_score, general_model_result):
    general_probability_percent = float(general_model_result["target_probability"] * 100.0)
    signed_difference = general_probability_percent - float(final_score)
    absolute_difference = abs(signed_difference)

    if absolute_difference <= 10.0:
        agreement_level = "Close agreement"
    elif absolute_difference <= 25.0:
        agreement_level = "Partial agreement"
    else:
        agreement_level = "Large difference"

    direction = "higher than" if signed_difference >= 0 else "lower than"
    return {
        "transparent_program_score": float(final_score),
        "general_model_aura_probability": general_probability_percent,
        "signed_difference_percentage_points": signed_difference,
        "absolute_difference_percentage_points": absolute_difference,
        "agreement_level": agreement_level,
        "summary": (
            f"The general model gave {general_probability_percent:.2f}% Aura, which is "
            f"{absolute_difference:.2f} percentage points {direction} the transparent score of "
            f"{final_score:.2f}%."
        ),
        "transparency_note": (
            "The transparent program shows how closed arms, serious face, and glasses contributed. "
            "The general model gives a probability without explaining which visual feature caused it."
        ),
    }


def make_report(results):
    final_score = float(np.clip(sum(item["contribution"] for item in results["models"]), 0.0, 100.0))
    grade = grade_from_score(final_score)
    general_comparison = compare_general_model_to_program(final_score, results["general_model"])

    report_lines = [
        "AURA SCORE REPORT",
        "",
        "Equation:",
        AURA_EQUATION,
        "",
        f"Face crop status: {results['face_crop_note']}",
        "",
    ]

    for index, item in enumerate(results["models"], start=1):
        report_lines.append(f"Model {index}: {item['display_name']}")
        report_lines.append("All class probabilities:")
        for label, probability in item["all_probabilities"].items():
            report_lines.append(f"- {friendly_label_name(label)}: {probability * 100:.2f}%")
        report_lines.append(f"Predicted target probability: {item['target_probability'] * 100:.2f}%")
        report_lines.append(f"Contribution: {item['contribution']:.2f} / 100")
        report_lines.append(f"Reason: {item['explanation']}")
        report_lines.append("")

    report_lines.extend(
        [
            "Transparent program final score:",
            f"{final_score:.2f} / 100",
            "Grade:",
            grade,
            "",
            "General model comparison:",
            "All class probabilities:",
        ]
    )

    for label, probability in results["general_model"]["all_probabilities"].items():
        report_lines.append(f"- {friendly_label_name(label)}: {probability * 100:.2f}%")

    report_lines.extend(
        [
            f"General model Aura probability: {results['general_model']['target_probability'] * 100:.2f}%",
            "Reason: This model gives a direct Aura or not Aura decision, but it does not explain which visual feature caused the decision.",
            f"Comparison: {general_comparison['summary']}",
            f"Agreement level: {general_comparison['agreement_level']}",
            "",
            "Transparency lesson:",
            general_comparison["transparency_note"],
        ]
    )

    json_result = {
        "input_image_path": results["input_image_path"],
        "face_crop_path": results["face_crop_path"],
        "whether_face_crop_was_used": bool(results["whether_face_crop_was_used"]),
        "face_crop_note": results["face_crop_note"],
        "equation": AURA_EQUATION,
        "normalization_mode": NORMALIZATION_MODE,
        "transparent_component_models": results["models"],
        "general_model": results["general_model"],
        "general_model_comparison": general_comparison,
        "final_aura_score": final_score,
        "grade": grade,
    }

    return "\n".join(report_lines), json_result


@st.cache_resource(show_spinner="Extracting and loading TensorFlow/Keras models...")
def load_all_models():
    for key, zip_path in MODEL_ZIPS.items():
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing model zip: {zip_path}")
        safe_extract_zip(zip_path, MODEL_DIRS[key])

    loaded_models = []
    for spec in MODEL_SPECS:
        model_dir = MODEL_DIRS[spec["key"]]
        model_path = find_model_file(model_dir)
        labels = read_labels(model_dir)
        model = load_keras_model(model_path)
        loaded_item = dict(spec)
        loaded_item.update(
            {
                "model": model,
                "model_path": str(model_path),
                "labels": labels,
                "input_size": list(get_input_size(model)),
            }
        )
        loaded_models.append(loaded_item)

    general_model_path = find_model_file(MODEL_DIRS["general_aura"])
    general_labels = read_labels(MODEL_DIRS["general_aura"])
    general_model = load_keras_model(general_model_path)
    loaded_general = dict(GENERAL_MODEL_SPEC)
    loaded_general.update(
        {
            "model": general_model,
            "model_path": str(general_model_path),
            "labels": general_labels,
            "input_size": list(get_input_size(general_model)),
        }
    )

    return loaded_models, loaded_general


def save_uploaded_image(uploaded_file):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.open(uploaded_file).convert("RGB")
    input_path = RUN_DIR / "input_image.png"
    image.save(input_path)
    return input_path


def run_aura_analysis(input_image_path, loaded_models, loaded_general):
    face_crop_path = RUN_DIR / "face_crop.png"
    face_crop_path, face_crop_used, face_crop_message = crop_face(input_image_path, face_crop_path)

    results = {
        "input_image_path": str(input_image_path),
        "face_crop_path": str(face_crop_path),
        "whether_face_crop_was_used": face_crop_used,
        "face_crop_note": face_crop_message,
        "models": [],
    }

    for item in loaded_models:
        image_path = input_image_path if item["image_source"] == "full_image" else face_crop_path
        prediction = predict_model(item["model"], image_path, item["labels"])
        target_probability, target_label, target_match_type = get_target_probability(
            prediction,
            TARGET_CLASS_NAMES[item["key"]],
            NEGATIVE_CLASS_NAMES.get(item["key"]),
        )
        contribution = float(np.clip(target_probability * EQUAL_MODEL_WEIGHT, 0.0, EQUAL_MODEL_WEIGHT))
        explanation = build_reason(item["reason_subject"], target_probability)

        results["models"].append(
            {
                "key": item["key"],
                "display_name": item["display_name"],
                "model_path": item["model_path"],
                "image_used": str(image_path),
                "input_size": item["input_size"],
                "labels": prediction["labels"],
                "raw_values": prediction["raw_values"],
                "all_probabilities": prediction["probability_map"],
                "target_label": target_label,
                "target_match_type": target_match_type,
                "target_probability": float(target_probability),
                "contribution": contribution,
                "explanation": explanation,
            }
        )

    general_prediction = predict_model(loaded_general["model"], input_image_path, loaded_general["labels"])
    general_target_probability, general_target_label, general_target_match_type = get_target_probability(
        general_prediction,
        TARGET_CLASS_NAMES[loaded_general["key"]],
        NEGATIVE_CLASS_NAMES.get(loaded_general["key"]),
    )

    results["general_model"] = {
        "key": loaded_general["key"],
        "display_name": loaded_general["display_name"],
        "model_path": loaded_general["model_path"],
        "image_used": str(input_image_path),
        "input_size": loaded_general["input_size"],
        "labels": general_prediction["labels"],
        "raw_values": general_prediction["raw_values"],
        "all_probabilities": general_prediction["probability_map"],
        "target_label": general_target_label,
        "target_match_type": general_target_match_type,
        "target_probability": float(general_target_probability),
        "explanation": "The general model gives an Aura probability, but it does not explain which visual feature caused the decision.",
    }

    report_text, result_json = make_report(results)
    return report_text, result_json


def render_probability_bar(label, probability):
    probability = float(np.clip(probability, 0.0, 1.0))
    percent_value = probability * 100.0
    color = probability_color(probability)
    st.markdown(
        f"""
        <div class="probability-block">
            <div class="probability-head">
                <span>{escape(label)}</span>
                <strong>{percent_value:.2f}%</strong>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: {percent_value:.2f}%; background: {color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_color(probability):
    probability = float(probability)
    if probability >= 0.80:
        return "#16785f"
    if probability >= 0.60:
        return "#1f6f8b"
    if probability >= 0.40:
        return "#c78522"
    return "#a84646"


def score_color(score):
    score = float(score)
    if score >= 80.0:
        return "#16785f"
    if score >= 60.0:
        return "#1f6f8b"
    if score >= 40.0:
        return "#c78522"
    return "#a84646"


def inject_design_css():
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f4f7fb;
            --surface: #ffffff;
            --surface-soft: #eef4f8;
            --ink: #102033;
            --muted: #5d6f82;
            --line: #d8e2ea;
            --navy: #123b57;
            --teal: #1f8a8a;
            --amber: #c78522;
        }

        .stApp {
            background: var(--page-bg);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background: #0f2538;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label {
            color: #eaf2f8;
        }

        section[data-testid="stSidebar"] code {
            color: #143047;
            white-space: pre-wrap;
        }

        div[data-testid="stFileUploader"] section {
            background: var(--surface);
            border: 1px dashed #9bb0c2;
            border-radius: 8px;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
            border: 1px solid #0e334d;
            background: #123b57;
            color: #ffffff;
            font-weight: 700;
            min-height: 3rem;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #1f8a8a;
            background: #0f3148;
            color: #ffffff;
        }

        .hero-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.6rem 1.7rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 14px 34px rgba(15, 37, 56, 0.08);
        }

        .eyebrow {
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .app-title {
            color: var(--ink);
            font-size: clamp(2rem, 3.4vw, 3.35rem);
            font-weight: 850;
            line-height: 1.02;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 780px;
            margin: 0.85rem 0 0 0;
        }

        .section-kicker {
            color: var(--teal);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin: 0.35rem 0 0.45rem 0;
        }

        .section-title {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 800;
            margin: 1.4rem 0 0.75rem 0;
        }

        .score-panel {
            display: grid;
            grid-template-columns: minmax(170px, 0.72fr) minmax(260px, 1.45fr);
            gap: 1.2rem;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            box-shadow: 0 12px 30px rgba(15, 37, 56, 0.07);
        }

        .score-ring {
            --score-deg: 0deg;
            --score-color: #1f6f8b;
            width: min(190px, 100%);
            aspect-ratio: 1;
            border-radius: 50%;
            display: grid;
            place-items: center;
            margin: 0 auto;
            background: conic-gradient(var(--score-color) var(--score-deg), #e5edf4 0deg);
            position: relative;
        }

        .score-ring::before {
            content: "";
            position: absolute;
            inset: 14px;
            background: var(--surface);
            border-radius: 50%;
            border: 1px solid #e4edf3;
        }

        .score-number {
            position: relative;
            z-index: 1;
            font-size: 2.2rem;
            font-weight: 850;
            color: var(--ink);
            line-height: 1;
        }

        .score-number span {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.9rem;
        }

        .summary-card,
        .feature-card,
        .comparison-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            min-height: 100%;
        }

        .summary-label,
        .feature-label,
        .comparison-label {
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.35rem;
        }

        .summary-value,
        .feature-value,
        .comparison-value {
            color: var(--ink);
            font-size: 1.45rem;
            font-weight: 850;
            line-height: 1.08;
        }

        .summary-caption,
        .feature-caption,
        .comparison-caption {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.4;
            margin-top: 0.45rem;
        }

        .feature-card {
            box-shadow: 0 10px 24px rgba(15, 37, 56, 0.06);
            position: relative;
            overflow: hidden;
        }

        .feature-card::after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 6px;
            height: 100%;
            background: var(--feature-accent, var(--teal));
        }

        .feature-top {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-bottom: 0.85rem;
            padding-right: 0.5rem;
        }

        .feature-icon {
            width: 54px;
            height: 54px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            flex: 0 0 auto;
            background: #eef6f7;
            border: 1px solid #d6e8ed;
            color: var(--feature-accent, var(--teal));
        }

        .feature-icon svg {
            width: 34px;
            height: 34px;
            stroke: currentColor;
        }

        .probability-block {
            margin-top: 0.85rem;
        }

        .probability-head {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            color: var(--muted);
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }

        .probability-head strong {
            color: var(--ink);
        }

        .probability-track {
            height: 0.62rem;
            background: #e8eef4;
            border-radius: 999px;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            border-radius: 999px;
        }

        .image-note {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.45rem;
        }

        .formula-box {
            background: #f6f9fb;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem;
            color: #143047;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.88rem;
            overflow-x: auto;
        }

        .flow-panel,
        .stack-panel,
        .scale-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 37, 56, 0.06);
        }

        .flow-grid {
            display: grid;
            grid-template-columns: 1fr 0.35fr 1fr 0.35fr 1fr 0.35fr 1fr;
            gap: 0.55rem;
            align-items: stretch;
        }

        .flow-node {
            min-height: 104px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #f8fbfd;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        .flow-node.dark {
            background: #123b57;
            color: #ffffff;
            border-color: #123b57;
        }

        .flow-node.dark .flow-caption,
        .flow-node.dark .flow-label {
            color: #eaf2f8;
        }

        .flow-icon {
            color: var(--teal);
            margin-bottom: 0.45rem;
        }

        .flow-node.dark .flow-icon {
            color: #ffffff;
        }

        .flow-icon svg {
            width: 34px;
            height: 34px;
            stroke: currentColor;
        }

        .flow-label {
            color: var(--ink);
            font-size: 0.95rem;
            font-weight: 850;
            line-height: 1.2;
        }

        .flow-caption {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.3;
            margin-top: 0.25rem;
        }

        .flow-arrow {
            display: grid;
            place-items: center;
            color: #7f94a8;
            font-size: 1.4rem;
            font-weight: 800;
        }

        .stack-bar {
            display: flex;
            height: 2.2rem;
            overflow: hidden;
            border-radius: 8px;
            background: #e8eef4;
            border: 1px solid #d8e2ea;
        }

        .stack-segment {
            min-width: 2px;
            display: grid;
            place-items: center;
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 850;
            white-space: nowrap;
        }

        .stack-empty {
            flex: 1;
            display: grid;
            place-items: center;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 750;
        }

        .stack-legend {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 0.8rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.25;
        }

        .legend-swatch {
            width: 0.85rem;
            height: 0.85rem;
            border-radius: 3px;
            flex: 0 0 auto;
        }

        .scale-track {
            position: relative;
            height: 3.1rem;
            margin: 1rem 0 0.6rem 0;
            border-radius: 8px;
            background: linear-gradient(90deg, #a84646 0%, #c78522 40%, #1f6f8b 65%, #16785f 100%);
            box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.08);
        }

        .scale-marker {
            position: absolute;
            top: -0.55rem;
            transform: translateX(-50%);
            width: 0.95rem;
            height: 4.2rem;
            border-radius: 999px;
            border: 3px solid #ffffff;
            box-shadow: 0 5px 16px rgba(15, 37, 56, 0.25);
        }

        .scale-marker.transparent {
            background: #102033;
        }

        .scale-marker.general {
            background: #ffffff;
            border-color: #102033;
        }

        .scale-labels {
            display: flex;
            justify-content: space-between;
            color: var(--muted);
            font-size: 0.8rem;
            font-weight: 750;
        }

        .marker-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.9rem;
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.75rem;
        }

        .marker-dot {
            display: inline-block;
            width: 0.75rem;
            height: 0.75rem;
            border-radius: 999px;
            margin-right: 0.35rem;
            vertical-align: -0.05rem;
        }

        @media (max-width: 760px) {
            .score-panel,
            .summary-grid,
            .flow-grid,
            .stack-legend {
                grid-template-columns: 1fr;
            }

            .flow-arrow {
                min-height: 1rem;
                transform: rotate(90deg);
            }

            .hero-panel {
                padding: 1.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_score_panel(score, grade, general_probability, comparison):
    score = float(score)
    general_probability = float(general_probability)
    score_accent = score_color(score)
    score_degrees = min(max(score, 0.0), 100.0) * 3.6
    difference = float(comparison["absolute_difference_percentage_points"])
    st.markdown(
        f"""
        <div class="score-panel">
            <div class="score-ring" style="--score-deg: {score_degrees:.2f}deg; --score-color: {score_accent};">
                <div class="score-number">{score:.1f}<span>out of 100</span></div>
            </div>
            <div>
                <div class="section-kicker">Final result</div>
                <div class="app-title" style="font-size: 2.25rem;">{escape(grade)}</div>
                <p class="app-subtitle">{escape(comparison["summary"])}</p>
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-label">Transparent score</div>
                        <div class="summary-value">{score:.2f}</div>
                        <div class="summary-caption">Built from three visible feature probabilities.</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">General model</div>
                        <div class="summary-value">{general_probability:.2f}%</div>
                        <div class="summary-caption">Direct Aura probability without a clear visual reason.</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Difference</div>
                        <div class="summary-value">{difference:.2f} pts</div>
                        <div class="summary-caption">{escape(comparison["agreement_level"])}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_accent(key):
    accents = {
        "closed_arms": "#245f73",
        "serious_face": "#7b5a9b",
        "glasses": "#1f8a8a",
        "general_aura": "#123b57",
        "score": "#16785f",
    }
    return accents.get(str(key), "#1f8a8a")


def icon_svg(name):
    icons = {
        "image": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect x="8" y="9" width="32" height="30" rx="4"></rect>
                <circle cx="18" cy="19" r="3"></circle>
                <path d="M12 34l9-9 6 6 4-4 6 7"></path>
            </svg>
        """,
        "crop": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M15 7v26h26"></path>
                <path d="M7 15h26v26"></path>
                <circle cx="24" cy="24" r="7"></circle>
                <path d="M20 31c2.5 2 5.5 2 8 0"></path>
            </svg>
        """,
        "closed_arms": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="24" cy="10" r="5"></circle>
                <path d="M24 16v18"></path>
                <path d="M14 22l20 9"></path>
                <path d="M34 22l-20 9"></path>
                <path d="M18 41l6-7 6 7"></path>
            </svg>
        """,
        "serious_face": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="24" cy="24" r="16"></circle>
                <path d="M17 21h.01"></path>
                <path d="M31 21h.01"></path>
                <path d="M18 31h12"></path>
            </svg>
        """,
        "glasses": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="16" cy="25" r="8"></circle>
                <circle cx="32" cy="25" r="8"></circle>
                <path d="M24 25h0"></path>
                <path d="M8 22l-4-4"></path>
                <path d="M40 22l4-4"></path>
            </svg>
        """,
        "models": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect x="8" y="9" width="12" height="12" rx="2"></rect>
                <rect x="28" y="9" width="12" height="12" rx="2"></rect>
                <rect x="8" y="29" width="12" height="12" rx="2"></rect>
                <rect x="28" y="29" width="12" height="12" rx="2"></rect>
                <path d="M20 15h8M20 35h8M14 21v8M34 21v8"></path>
            </svg>
        """,
        "score": """
            <svg viewBox="0 0 48 48" fill="none" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M10 36h28"></path>
                <path d="M14 36V24"></path>
                <path d="M24 36V14"></path>
                <path d="M34 36V20"></path>
                <path d="M12 14l7 6 8-10 9 7"></path>
            </svg>
        """,
    }
    return icons.get(str(name), icons["score"])


def render_model_flow_graphic():
    st.markdown(
        f"""
        <div class="flow-panel">
            <div class="flow-grid">
                <div class="flow-node">
                    <div class="flow-icon">{icon_svg("image")}</div>
                    <div class="flow-label">Uploaded photo</div>
                    <div class="flow-caption">One image enters the system.</div>
                </div>
                <div class="flow-arrow">-&gt;</div>
                <div class="flow-node">
                    <div class="flow-icon">{icon_svg("crop")}</div>
                    <div class="flow-label">Two views</div>
                    <div class="flow-caption">Full image and face crop.</div>
                </div>
                <div class="flow-arrow">-&gt;</div>
                <div class="flow-node">
                    <div class="flow-icon">{icon_svg("models")}</div>
                    <div class="flow-label">Four models</div>
                    <div class="flow-caption">Three transparent features plus one general model.</div>
                </div>
                <div class="flow-arrow">-&gt;</div>
                <div class="flow-node dark">
                    <div class="flow-icon">{icon_svg("score")}</div>
                    <div class="flow-label">Aura report</div>
                    <div class="flow-caption">Score, grade, probabilities, and comparison.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_contribution_stack(result_json):
    items = result_json["transparent_component_models"]
    final_score = float(result_json["final_aura_score"])
    segments = []
    legend_items = []
    for item in items:
        contribution = float(item["contribution"])
        width = max(0.0, min(100.0, contribution))
        color = feature_accent(item["key"])
        label = escape(item["display_name"])
        inner_label = f"{contribution:.1f}" if width >= 7.0 else ""
        segments.append(
            f'<div class="stack-segment" style="width: {width:.2f}%; background: {color};">{inner_label}</div>'
        )
        legend_items.append(
            f"""
            <div class="legend-item">
                <span class="legend-swatch" style="background: {color};"></span>
                <span>{label}: {contribution:.2f} pts</span>
            </div>
            """
        )

    remainder = max(0.0, 100.0 - final_score)
    remainder_html = ""
    if remainder > 0.01:
        remainder_html = f'<div class="stack-empty" style="width: {remainder:.2f}%;">unused {remainder:.1f}</div>'

    st.markdown(
        f"""
        <div class="stack-panel">
            <div class="section-kicker">Score composition</div>
            <div class="stack-bar">
                {''.join(segments)}
                {remainder_html}
            </div>
            <div class="stack-legend">
                {''.join(legend_items)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(item):
    target_probability = float(item["target_probability"])
    contribution = float(item["contribution"])
    percent_value = float(np.clip(target_probability, 0.0, 1.0)) * 100.0
    color = probability_color(target_probability)
    accent = feature_accent(item["key"])
    st.markdown(
        f"""
        <div class="feature-card" style="--feature-accent: {accent};">
            <div class="feature-top">
                <div class="feature-icon">{icon_svg(item["key"])}</div>
                <div>
                    <div class="feature-label">{escape(item["display_name"])}</div>
                    <div class="feature-value">{target_probability * 100:.2f}%</div>
                    <div class="feature-caption">Contribution: {contribution:.2f} / 100</div>
                </div>
            </div>
            <div class="probability-block">
                <div class="probability-head">
                    <span>Target probability</span>
                    <strong>{percent_value:.2f}%</strong>
                </div>
                <div class="probability-track">
                    <div class="probability-fill" style="width: {percent_value:.2f}%; background: {color};"></div>
                </div>
            </div>
            <div class="feature-caption">{escape(item["explanation"])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_scale(result_json):
    comparison = result_json["general_model_comparison"]
    transparent_score = float(np.clip(result_json["final_aura_score"], 0.0, 100.0))
    general_probability = float(np.clip(result_json["general_model"]["target_probability"] * 100.0, 0.0, 100.0))
    st.markdown(
        f"""
        <div class="scale-panel">
            <div class="section-kicker">Visual comparison scale</div>
            <div class="scale-track">
                <div class="scale-marker transparent" style="left: {transparent_score:.2f}%;"></div>
                <div class="scale-marker general" style="left: {general_probability:.2f}%;"></div>
            </div>
            <div class="scale-labels">
                <span>0</span>
                <span>Low</span>
                <span>Medium</span>
                <span>Strong</span>
                <span>100</span>
            </div>
            <div class="marker-legend">
                <span><span class="marker-dot" style="background: #102033;"></span>Transparent score: {transparent_score:.2f}%</span>
                <span><span class="marker-dot" style="background: #ffffff; border: 2px solid #102033;"></span>General model: {general_probability:.2f}%</span>
                <span>{escape(comparison["agreement_level"])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_cards(result_json):
    comparison = result_json["general_model_comparison"]
    transparent_score = float(result_json["final_aura_score"])
    general_probability = float(result_json["general_model"]["target_probability"] * 100.0)
    col1, col2, col3 = st.columns([1, 1, 1.2], gap="large")
    with col1:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-label">Transparent program</div>
                <div class="comparison-value">{transparent_score:.2f}%</div>
                <div class="comparison-caption">Closed arms + serious face + glasses, each with equal weight.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-label">General model</div>
                <div class="comparison-value">{general_probability:.2f}%</div>
                <div class="comparison-caption">A single Aura prediction with no feature-level explanation.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="comparison-label">Transparency lesson</div>
                <div class="comparison-caption" style="font-size: 0.95rem;">{escape(comparison["transparency_note"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.set_page_config(
    page_title="Aura Score Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_design_css()

st.markdown(
    """
    <div class="hero-panel">
        <div class="eyebrow">Explainable AI project</div>
        <h1 class="app-title">Aura Score Analyzer</h1>
        <p class="app-subtitle">
            A transparent image-analysis app that separates visible feature evidence from a general black-box Aura decision.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline_image = ASSETS_DIR / "aura_pipeline.png"

with st.sidebar:
    st.markdown("### Project")
    st.caption(
        "Upload one photo. The app checks closed arms, serious face, and glasses, "
        "then compares the transparent score with a general Aura model."
    )
    st.markdown("### Transparent Formula")
    st.markdown(f'<div class="formula-box">{escape(AURA_EQUATION)}</div>', unsafe_allow_html=True)
    st.markdown("### Model Inputs")
    st.markdown(
        """
        - Closed arms model: full image
        - Serious face model: face crop
        - Glasses model: face crop
        - General Aura model: full image
        """
    )

input_col, graphic_col = st.columns([0.82, 1.18], gap="large")
with input_col:
    st.markdown('<div class="section-kicker">Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )

    upload_key = None
    if uploaded_file is not None:
        upload_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 'unknown')}"

    analyze_clicked = st.button("Analyze Aura", type="primary", use_container_width=True)

with graphic_col:
    st.markdown('<div class="section-kicker">Pipeline</div>', unsafe_allow_html=True)
    if pipeline_image.exists():
        st.image(str(pipeline_image), use_container_width=True)
    else:
        st.info("Pipeline graphic is not available.")

render_model_flow_graphic()

if uploaded_file is None:
    st.info("Upload a photo to start the Aura Score analysis.")
    st.stop()

if analyze_clicked:
    try:
        with st.spinner("Loading models..."):
            loaded_models, loaded_general = load_all_models()
    except Exception as error:
        st.error("The models could not be loaded.")
        st.exception(error)
        st.stop()

    with st.spinner("Analyzing image..."):
        input_image_path = save_uploaded_image(uploaded_file)
        report_text, result_json = run_aura_analysis(input_image_path, loaded_models, loaded_general)

    st.session_state["aura_upload_key"] = upload_key
    st.session_state["aura_report_text"] = report_text
    st.session_state["aura_result_json"] = result_json
elif st.session_state.get("aura_upload_key") == upload_key:
    report_text = st.session_state["aura_report_text"]
    result_json = st.session_state["aura_result_json"]
else:
    st.info("Click Analyze Aura after uploading the photo.")
    st.stop()

score = result_json["final_aura_score"]
grade = result_json["grade"]
general_probability = result_json["general_model"]["target_probability"] * 100.0
comparison = result_json["general_model_comparison"]

st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
render_score_panel(score, grade, general_probability, comparison)
render_contribution_stack(result_json)

st.markdown('<div class="section-title">Images Used By The Models</div>', unsafe_allow_html=True)
image_col1, image_col2 = st.columns(2)
with image_col1:
    st.markdown('<div class="section-kicker">Full image</div>', unsafe_allow_html=True)
    st.image(result_json["input_image_path"], use_container_width=True)
with image_col2:
    st.markdown('<div class="section-kicker">Face crop or fallback</div>', unsafe_allow_html=True)
    st.image(result_json["face_crop_path"], use_container_width=True)
    st.markdown(f'<div class="image-note">{escape(result_json["face_crop_note"])}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Transparent Feature Contributions</div>', unsafe_allow_html=True)
feature_cols = st.columns(3)
for column, item in zip(feature_cols, result_json["transparent_component_models"]):
    with column:
        render_feature_card(item)

st.markdown('<div class="section-title">Transparent Score vs General Model</div>', unsafe_allow_html=True)
render_comparison_cards(result_json)
render_comparison_scale(result_json)

with st.expander("All class probabilities"):
    for item in result_json["transparent_component_models"]:
        st.write(f"**{item['display_name']}**")
        st.json({friendly_label_name(label): probability for label, probability in item["all_probabilities"].items()})
    st.write("**General black-box Aura model**")
    st.json({
        friendly_label_name(label): probability
        for label, probability in result_json["general_model"]["all_probabilities"].items()
    })

with st.expander("Full text report", expanded=False):
    st.text(report_text)

st.download_button(
    "Download aura_result.json",
    data=json.dumps(result_json, indent=2),
    file_name="aura_result.json",
    mime="application/json",
    use_container_width=True,
)
