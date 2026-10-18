"""Dataset and sensor type detection without loading file contents eagerly."""

from pathlib import PurePosixPath


TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".log", ".py"}
SENSOR_TERMS = {
    "accelerometer": {"accelerometer", "accel", "acc"},
    "gyroscope": {"gyroscope", "gyro", "gyr"},
    "magnetometer": {"magnetometer", "mag", "magnetic"},
}


def detect_sensor_columns(columns):
    """Group common sensor axes by family and return their source column names."""
    normalized = {str(column).strip().lower().replace("-", "_"): str(column) for column in columns}
    groups = {}
    axes = ("x", "y", "z")
    for family, terms in SENSOR_TERMS.items():
        matches = []
        for normalized_name, original_name in normalized.items():
            parts = normalized_name.replace(" ", "_").split("_")
            if any(term in parts for term in terms) and any(axis in parts for axis in axes):
                matches.append(original_name)
        if matches:
            groups[family] = matches
    return groups


def detect_dataset_type(filename, columns=None):
    """Classify a dataset using its extension and optional tabular columns."""
    path = PurePosixPath(str(filename).replace("\\", "/"))
    suffix = path.suffix.lower()
    if suffix in TABULAR_EXTENSIONS:
        sensor_groups = detect_sensor_columns(columns if columns is not None else [])
        return "sensor" if sensor_groups else "tabular"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix == ".ipynb":
        return "notebook"
    if suffix == ".zip":
        return "archive"
    return "unsupported"
