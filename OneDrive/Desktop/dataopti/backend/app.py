import os
import io
import codecs
import re
import json
import time
import zipfile
import traceback
from collections import Counter

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# OPTIONAL LIBRARIES
# ---------------------------------------------------------

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import nbformat
except ImportError:
    nbformat = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Machine learning
try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        StandardScaler,
        LabelEncoder,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        balanced_accuracy_score,
    )

    SKLEARN_AVAILABLE = True

except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------
# FLASK
# ---------------------------------------------------------

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# LARGE FILE SETTINGS
# ---------------------------------------------------------
#
# This is an HTTP request limit, not a memory allocation.
# Change it if your machine/server requires another limit.
#
# 100 GB is intentionally used so that a large dataset does
# not immediately fail because of an arbitrary 4 GB limit.
#
# The actual ML operation is controlled separately by
# MAX_ML_ROWS.
# ---------------------------------------------------------

MAX_GB = 100
MAX_BYTES = MAX_GB * 1024 * 1024 * 1024

app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

# Do NOT set MAX_FORM_MEMORY_SIZE to the whole dataset.
# Werkzeug can spool uploaded data instead of forcing the
# entire upload into Python RAM.
app.config["MAX_FORM_MEMORY_SIZE"] = 1024 * 1024 * 64

# ---------------------------------------------------------
# PERFORMANCE SETTINGS
# ---------------------------------------------------------

CSV_CHUNK_SIZE = 50_000

# Maximum rows used for expensive ML operations.
# The complete dataset is still profiled separately.
MAX_ML_ROWS = 20_000
FAST_ML_ROWS = 3_000
FAST_PROFILE_ROWS = 10_000
# Switch to bounded analysis early enough to avoid a full scan of files
# that are large in practice (including compressed spreadsheet files).
FAST_DATASET_THRESHOLD_BYTES = 10 * 1024 * 1024
FAST_PDF_PAGES = 50

# Maximum rows kept for before/after cleaning demonstration.
MAX_DEMO_ROWS = 50_000

# Maximum number of columns considered by ML.
MAX_FEATURE_COLUMNS = 150

# Maximum unique classes for classification.
MAX_CLASSES = 30

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------
# HTTP 413
# ---------------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
@app.errorhandler(413)
def handle_request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "message": (
            f"Upload exceeds the server limit of {MAX_GB} GB. "
            "For very large datasets, upload them in smaller batches."
        )
    }), 413


# ---------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------

def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
        return default
    except Exception:
        return default


def format_bytes(size):
    size = float(size)

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def stream_size_bytes(stream):
    try:
        position = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(position)
        return size
    except Exception:
        return 0


def ml_sample_limit(stream, requested_limit=MAX_ML_ROWS):
    if stream_size_bytes(stream) >= FAST_DATASET_THRESHOLD_BYTES:
        return min(requested_limit, FAST_ML_ROWS)
    return requested_limit


def read_json_lines_sample(stream, row_limit):
    """Read only the first bounded batch from a JSON Lines file, if detected."""

    stream.seek(0)
    first_line = stream.readline()
    second_line = stream.readline()

    try:
        first_record = json.loads(first_line)
        second_record = json.loads(second_line)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        stream.seek(0)
        return None

    if not isinstance(first_record, dict) or not isinstance(second_record, dict):
        stream.seek(0)
        return None

    stream.seek(0)
    try:
        batches = pd.read_json(stream, lines=True, chunksize=row_limit)
        return next(batches, pd.DataFrame())
    except Exception:
        stream.seek(0)
        return None


def read_json_array_sample(stream, row_limit):
    """Stream a bounded sample from a top-level JSON array of records."""

    stream.seek(0)
    json_decoder = json.JSONDecoder()
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    position = 0
    end_of_stream = False
    records = []

    def read_more():
        nonlocal buffer, end_of_stream
        raw = stream.read(1024 * 1024)
        if raw:
            buffer += utf8_decoder.decode(raw, final=False)
        else:
            buffer += utf8_decoder.decode(b"", final=True)
            end_of_stream = True

    while True:
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position < len(buffer):
            if buffer[position] == "\ufeff":
                position += 1
                continue
            if buffer[position] != "[":
                stream.seek(0)
                return None
            position += 1
            break
        if end_of_stream:
            stream.seek(0)
            return None
        read_more()

    while len(records) < row_limit:
        while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
            position += 1

        if position >= len(buffer):
            if end_of_stream:
                break
            read_more()
            continue

        if buffer[position] == "]":
            break

        try:
            record, next_position = json_decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            if end_of_stream:
                stream.seek(0)
                return None
            read_more()
            continue

        records.append(record)
        position = next_position

        if position >= 1024 * 1024:
            buffer = buffer[position:]
            position = 0

    stream.seek(0)
    return pd.DataFrame(records)


def clean_column_names(df):
    df = df.copy()

    new_columns = []

    for i, column in enumerate(df.columns):
        name = str(column).strip()

        if not name:
            name = f"column_{i + 1}"

        new_columns.append(name)

    # Make duplicate names unique.
    seen = {}

    final_columns = []

    for name in new_columns:
        if name not in seen:
            seen[name] = 0
            final_columns.append(name)
        else:
            seen[name] += 1
            final_columns.append(f"{name}_{seen[name]}")

    df.columns = final_columns

    return df


def duplicate_row_mask(df):
    """Find duplicate rows even when object cells contain nested values."""

    try:
        return df.duplicated()
    except (TypeError, ValueError):
        comparable = df.copy()

        def comparable_value(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, set):
                value = sorted(value, key=repr)
            if isinstance(value, (list, tuple, dict)):
                return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if isinstance(value, np.generic):
                return value.item()
            return value

        for column in comparable.columns:
            if comparable[column].dtype == object:
                comparable[column] = comparable[column].map(comparable_value)

        return comparable.duplicated()


# ---------------------------------------------------------
# OUTLIER DETECTION
# ---------------------------------------------------------

def count_iqr_outliers(df):
    numeric = df.select_dtypes(include=[np.number])

    if numeric.empty:
        return 0

    try:
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (numeric.lt(lower)) | (numeric.gt(upper))

        return int(mask.sum().sum())

    except Exception:
        return 0


# ---------------------------------------------------------
# CSV - LARGE DATASET OPTIMIZED
# ---------------------------------------------------------

def profile_csv_stream(file_storage):
    """
    Profiles CSV without loading the complete CSV into RAM.

    Uses chunks for large files.
    """

    file_storage.stream.seek(0)

    if stream_size_bytes(file_storage.stream) >= FAST_DATASET_THRESHOLD_BYTES:
        sample = pd.read_csv(
            file_storage.stream,
            nrows=FAST_PROFILE_ROWS,
            low_memory=True,
        )
        profile = build_dataframe_profile(
            sample,
            processing="sampled_csv",
        )
        profile["profile_sampled"] = True
        profile["profile_sample_rows"] = len(sample)
        return profile

    total_rows = 0
    total_missing = 0
    total_duplicates = 0
    total_outliers = 0

    column_names = []
    dtypes = {}

    numeric_values = {}

    try:
        for chunk in pd.read_csv(
            file_storage.stream,
            chunksize=CSV_CHUNK_SIZE,
            low_memory=True
        ):

            chunk = clean_column_names(chunk)

            if not column_names:
                column_names = list(chunk.columns.astype(str))

            total_rows += len(chunk)

            total_missing += int(chunk.isna().sum().sum())

            # Duplicate detection inside chunks.
            total_duplicates += int(chunk.duplicated().sum())

            # Keep small numeric samples for fast outlier estimation.
            numeric = chunk.select_dtypes(include=[np.number])

            for col in numeric.columns:

                if col not in numeric_values:
                    numeric_values[col] = []

                values = numeric[col].dropna()

                if len(values) > 500:
                    values = values.sample(
                        n=500,
                        random_state=42
                    )

                numeric_values[col].extend(
                    values.tolist()
                )

            for col in chunk.columns:
                if col not in dtypes:
                    dtypes[col] = str(chunk[col].dtype)

        # Approximate IQR outliers using sampled numeric values.
        for col, values in numeric_values.items():

            if not values:
                continue

            arr = np.asarray(values, dtype=float)

            if len(arr) < 10:
                continue

            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)

            iqr = q3 - q1

            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            total_outliers += int(
                ((arr < lower) | (arr > upper)).sum()
            )

        return {
            "rows": total_rows,
            "columns": len(column_names),
            "missing_values": total_missing,
            "duplicate_rows": total_duplicates,
            "outliers": total_outliers,
            "column_names": column_names,
            "dtypes": dtypes,
            "processing": "chunked",
        }

    except Exception as e:

        print(f"CSV profiling error: {e}")

        return None


# ---------------------------------------------------------
# LOAD A REPRESENTATIVE TABULAR SAMPLE
# ---------------------------------------------------------

def load_tabular_sample(file_storage, filename, max_rows=MAX_ML_ROWS):

    ext = os.path.splitext(filename.lower())[1]
    sample_limit = ml_sample_limit(file_storage.stream, max_rows)

    try:

        file_storage.stream.seek(0)

        if ext == ".csv":

            df = pd.read_csv(
                file_storage.stream,
                nrows=sample_limit,
                low_memory=True
            )

        elif ext in [".xlsx", ".xls"]:

            # Excel readers generally need more memory than CSV.
            df = pd.read_excel(
                file_storage.stream,
                nrows=sample_limit
            )

        elif ext == ".parquet":
            if stream_size_bytes(file_storage.stream) >= FAST_DATASET_THRESHOLD_BYTES:
                try:
                    import pyarrow.parquet as parquet
                    parquet_file = parquet.ParquetFile(file_storage.stream)
                    first_batch = next(
                        parquet_file.iter_batches(batch_size=sample_limit),
                        None,
                    )
                    df = first_batch.to_pandas() if first_batch is not None else pd.DataFrame()
                except Exception:
                    file_storage.stream.seek(0)
                    df = pd.read_parquet(file_storage.stream)
                    if len(df) > sample_limit:
                        df = df.sample(n=sample_limit, random_state=42)
            else:
                df = pd.read_parquet(file_storage.stream)
                if len(df) > sample_limit:
                    df = df.sample(n=sample_limit, random_state=42)

        elif ext == ".json":
            df = read_json_lines_sample(file_storage.stream, sample_limit)
            if df is None and stream_size_bytes(file_storage.stream) >= FAST_DATASET_THRESHOLD_BYTES:
                df = read_json_array_sample(file_storage.stream, sample_limit)
            if df is None:
                df = pd.read_json(file_storage.stream)
                if len(df) > sample_limit:
                    df = df.sample(n=sample_limit, random_state=42)

        else:
            return None

        df = clean_column_names(df)
        df.attrs["ml_sample_limit"] = sample_limit
        return df

    except Exception as e:

        print(
            f"Could not load sample for {filename}: {e}"
        )

        return None


# ---------------------------------------------------------
# TABULAR PROFILE
# ---------------------------------------------------------

def profile_tabular_file(file_storage, filename):

    ext = os.path.splitext(filename.lower())[1]

    # CSV uses chunk processing.
    if ext == ".csv":

        return profile_csv_stream(file_storage)

    try:

        file_storage.stream.seek(0)

        if ext in [".xlsx", ".xls"]:

            df = pd.read_excel(
                file_storage.stream,
                nrows=FAST_PROFILE_ROWS
            )

            # Get approximate/full row count where possible.
            sampled = clean_column_names(df)
            profile = build_dataframe_profile(
                sampled,
                processing="sampled_excel"
            )
            profile["profile_sampled"] = True
            profile["profile_sample_rows"] = len(sampled)
            return profile

        if ext == ".parquet":

            if stream_size_bytes(file_storage.stream) >= FAST_DATASET_THRESHOLD_BYTES:
                try:
                    import pyarrow.parquet as parquet
                    parquet_file = parquet.ParquetFile(file_storage.stream)
                    first_batch = next(
                        parquet_file.iter_batches(batch_size=FAST_PROFILE_ROWS),
                        None,
                    )
                    sample = first_batch.to_pandas() if first_batch is not None else pd.DataFrame()
                    profile = build_dataframe_profile(
                        sample,
                        processing="sampled_parquet",
                    )
                    profile["rows"] = int(parquet_file.metadata.num_rows)
                    profile["profile_sampled"] = True
                    profile["profile_sample_rows"] = len(sample)
                    return profile
                except Exception as sample_error:
                    print(f"Fast Parquet profile unavailable for {filename}: {sample_error}")
                    file_storage.stream.seek(0)

            df = pd.read_parquet(
                file_storage.stream
            )

            return build_dataframe_profile(
                df,
                processing="parquet"
            )

        if ext == ".json":
            sample = read_json_lines_sample(
                file_storage.stream,
                FAST_PROFILE_ROWS,
            )
            if sample is None and stream_size_bytes(file_storage.stream) >= FAST_DATASET_THRESHOLD_BYTES:
                sample = read_json_array_sample(
                    file_storage.stream,
                    FAST_PROFILE_ROWS,
                )
            if sample is not None:
                profile = build_dataframe_profile(
                    sample,
                    processing="sampled_json_lines",
                )
                profile["profile_sampled"] = True
                profile["profile_sample_rows"] = len(sample)
                return profile

            df = pd.read_json(
                file_storage.stream
            )

            return build_dataframe_profile(
                df,
                processing="json"
            )

    except Exception as e:

        print(
            f"Tabular parsing error for {filename}: {e}"
        )

    return None


def build_dataframe_profile(df, processing="normal"):

    if df is None:
        return None

    df = clean_column_names(df)

    rows, cols = df.shape

    missing = int(
        df.isna().sum().sum()
    )

    duplicates = int(
        duplicate_row_mask(df).sum()
    )

    outliers = count_iqr_outliers(df)

    dtypes = {
        str(col): str(dtype)
        for col, dtype in df.dtypes.items()
    }

    return {
        "rows": int(rows),
        "columns": int(cols),
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "outliers": outliers,
        "column_names": [
            str(c) for c in df.columns
        ],
        "dtypes": dtypes,
        "processing": processing,
    }


# ---------------------------------------------------------
# PDF
# ---------------------------------------------------------

def parse_pdf(file_storage):

    result = {
        "rows": 0,
        "columns": 1,
        "missing_values": 0,
        "duplicate_rows": 0,
        "outliers": 0,
        "column_names": [],
        "status": "success",
        "reason": "",
        "pdf_pages": 0,
        "pdf_words": 0,
    }

    if pypdf is None:

        result["status"] = "skipped"
        result["reason"] = (
            "pypdf is not installed. "
            "Run: pip install pypdf"
        )

        return result

    try:

        file_storage.stream.seek(0)

        reader = pypdf.PdfReader(
            file_storage.stream,
            strict=False
        )

        if reader.is_encrypted:

            try:
                reader.decrypt("")
            except Exception:
                pass

            if reader.is_encrypted:

                result["status"] = "skipped"
                result["reason"] = (
                    "PDF is password-protected. "
                    "Upload an unlocked PDF."
                )

                return result

        pages = len(reader.pages)
        sampled = stream_size_bytes(file_storage.stream) >= FAST_DATASET_THRESHOLD_BYTES
        pages_to_read = min(pages, FAST_PDF_PAGES) if sampled else pages
        word_count = 0

        for page_index in range(pages_to_read):

            try:
                text = reader.pages[page_index].extract_text() or ""
                word_count += len(text.split())
            except Exception:
                continue

        result.update({
            "rows": pages,
            "pdf_pages": pages,
            "pdf_words": word_count,
            "column_names": [
                f"PDF pages: {pages}",
                f"Extracted words: {word_count}"
            ],
        })

        if sampled:
            result["profile_sampled"] = True
            result["profile_sample_rows"] = pages_to_read

        # Scanned/image-only PDF.
        if word_count == 0:

            result["reason"] = (
                "PDF contains no extractable text. "
                "It may be scanned/image-based."
            )

        return result

    except Exception as e:

        error_text = str(e).lower()

        if (
            "password" in error_text
            or "encrypted" in error_text
        ):

            result["status"] = "skipped"
            result["reason"] = (
                "PDF is password-protected."
            )

        else:

            result["status"] = "skipped"
            result["reason"] = (
                "PDF is damaged, truncated, invalid, "
                "or not a real PDF."
            )

        print(
            f"PDF skipped: {file_storage.filename} -> {e}"
        )

        return result


# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_image(file_storage):

    result = {
        "rows": 1,
        "columns": 0,
        "missing_values": 0,
        "duplicate_rows": 0,
        "outliers": 0,
        "column_names": [],
        "status": "success",
        "image_count": 1,
        "classes": [],
    }

    if Image is None:

        result["status"] = "skipped"
        result["reason"] = (
            "Pillow is not installed. "
            "Run: pip install pillow"
        )

        return result

    try:

        file_storage.stream.seek(0)

        with Image.open(
            file_storage.stream
        ) as image:

            width, height = image.size

            result["columns"] = 3

            result["column_names"] = [
                f"Image width: {width}px",
                f"Image height: {height}px",
                f"Image mode: {image.mode}",
            ]

        return result

    except Exception as e:

        result["status"] = "skipped"
        result["reason"] = (
            "Invalid or damaged image."
        )

        print(
            f"Image error: {file_storage.filename}: {e}"
        )

        return result


# ---------------------------------------------------------
# TEXT / NOTEBOOK
# ---------------------------------------------------------

def parse_text_or_code(file_storage, filename):

    try:

        file_storage.stream.seek(0)

        raw = file_storage.stream.read(
            5 * 1024 * 1024
        )

        text = raw.decode(
            "utf-8",
            errors="ignore"
        )

        if filename.lower().endswith(".ipynb"):

            if nbformat is None:

                return {
                    "rows": 0,
                    "columns": 0,
                    "missing_values": 0,
                    "duplicate_rows": 0,
                    "outliers": 0,
                    "column_names": [],
                    "status": "skipped",
                    "reason": "nbformat is not installed."
                }

            try:

                nb = nbformat.reads(
                    text,
                    as_version=4
                )

                code_cells = sum(
                    1
                    for c in nb.cells
                    if c.cell_type == "code"
                )

                markdown_cells = sum(
                    1
                    for c in nb.cells
                    if c.cell_type == "markdown"
                )

                return {
                    "rows": len(nb.cells),
                    "columns": 2,
                    "missing_values": 0,
                    "duplicate_rows": 0,
                    "outliers": 0,
                    "column_names": [
                        f"Code cells: {code_cells}",
                        f"Markdown cells: {markdown_cells}",
                    ],
                    "status": "success",
                }

            except Exception:

                return {
                    "rows": len(text.splitlines()),
                    "columns": 1,
                    "missing_values": 0,
                    "duplicate_rows": 0,
                    "outliers": 0,
                    "column_names": [
                        "Notebook text"
                    ],
                    "status": "success",
                }

        lines = text.splitlines()

        return {
            "rows": len(lines),
            "columns": 1,
            "missing_values": 0,
            "duplicate_rows": 0,
            "outliers": 0,
            "column_names": [
                f"Text lines: {len(lines)}",
                f"Words: {len(text.split())}",
            ],
            "status": "success",
        }

    except Exception as e:

        return {
            "rows": 0,
            "columns": 0,
            "missing_values": 0,
            "duplicate_rows": 0,
            "outliers": 0,
            "column_names": [],
            "status": "skipped",
            "reason": str(e),
        }


# ---------------------------------------------------------
# ZIP
# ---------------------------------------------------------

def parse_zip(file_storage):

    result = {
        "rows": 0,
        "columns": 0,
        "missing_values": 0,
        "duplicate_rows": 0,
        "outliers": 0,
        "column_names": [],
        "status": "success",
        "zip_files": 0,
        "image_files": 0,
        "tabular_files": 0,
    }

    try:

        file_storage.stream.seek(0)

        with zipfile.ZipFile(
            file_storage.stream
        ) as z:

            names = z.namelist()

            result["zip_files"] = len(names)

            for name in names:

                lower = name.lower()

                if lower.endswith(
                    (
                        ".csv",
                        ".xlsx",
                        ".xls",
                        ".json",
                        ".parquet",
                    )
                ):
                    result["tabular_files"] += 1

                elif any(
                    lower.endswith(ext)
                    for ext in IMAGE_EXTENSIONS
                ):
                    result["image_files"] += 1

            result["column_names"] = [
                f"ZIP files: {len(names)}",
                f"Tabular files: {result['tabular_files']}",
                f"Image files: {result['image_files']}",
            ]

        return result

    except Exception as e:

        return {
            **result,
            "status": "skipped",
            "reason": f"Invalid ZIP file: {e}",
        }


# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------

def clean_dataframe(df):

    if df is None or df.empty:

        return df, {
            "rows_before": 0,
            "rows_after": 0,
            "missing_before": 0,
            "missing_after": 0,
            "duplicates_before": 0,
            "duplicates_after": 0,
            "outliers_before": 0,
            "outliers_after": 0,
        }

    df = clean_column_names(df.copy())

    rows_before = len(df)

    missing_before = int(
        df.isna().sum().sum()
    )

    duplicate_mask_before = duplicate_row_mask(df)
    duplicates_before = int(duplicate_mask_before.sum())

    outliers_before = count_iqr_outliers(df)

    # -----------------------------------------------------
    # REMOVE DUPLICATES
    # -----------------------------------------------------

    df = df.loc[~duplicate_mask_before].copy()

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=[np.number]
    ).columns

    for column in numeric_columns:

        if df[column].isna().any():

            median = df[column].median()

            if pd.isna(median):
                median = 0

            df[column] = df[column].fillna(
                median
            )

    for column in categorical_columns:

        if df[column].isna().any():

            try:
                mode = df[column].mode()
                value = mode.iloc[0] if not mode.empty else "Unknown"
            except (TypeError, ValueError):
                # Pandas mode requires hashable cells. JSON and Parquet
                # columns can contain lists, dictionaries, or ndarrays.
                series = df[column]

                def mode_key(item):
                    if isinstance(item, np.ndarray):
                        item = item.tolist()
                    if isinstance(item, np.generic):
                        item = item.item()
                    return json.dumps(
                        {"type": type(item).__name__, "value": item},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )

                comparable = series.map(mode_key)
                valid = series.notna()
                counts = comparable[valid].value_counts()

                if counts.empty:
                    value = "Unknown"
                else:
                    most_common_key = counts.index[0]
                    matching = np.flatnonzero(
                        (comparable.eq(most_common_key) & valid).to_numpy()
                    )
                    value = series.iloc[matching[0]] if len(matching) else "Unknown"

            if isinstance(value, (list, tuple, dict, set, np.ndarray)):
                missing_positions = np.flatnonzero(df[column].isna().to_numpy())
                filled = df[column].astype(object).copy()
                for position in missing_positions:
                    filled.iloc[position] = value
                df[column] = filled
            else:
                df[column] = df[column].fillna(value)

    # -----------------------------------------------------
    # OUTLIER CLIPPING
    # -----------------------------------------------------

    for column in numeric_columns:

        try:

            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df[column] = df[column].clip(
                lower,
                upper
            )

        except Exception:
            continue

    missing_after = int(
        df.isna().sum().sum()
    )

    duplicates_after = int(
        duplicate_row_mask(df).sum()
    )

    outliers_after = count_iqr_outliers(df)

    return df, {
        "rows_before": rows_before,
        "rows_after": len(df),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "outliers_before": outliers_before,
        "outliers_after": outliers_after,
    }


# ---------------------------------------------------------
# TARGET DETECTION
# ---------------------------------------------------------

def is_nested_data_value(value):
    return isinstance(value, (list, tuple, dict, set, np.ndarray))


def has_nested_data_values(series):
    return any(is_nested_data_value(value) for value in series)

def infer_target_column(df):

    if df is None or df.empty:
        return None

    columns = list(df.columns)
    scalar_columns = [
        column for column in columns
        if not has_nested_data_values(df[column])
    ]

    # Prefer common target names.
    target_words = [
        "target",
        "label",
        "class",
        "output",
        "result",
        "y",
        "category",
        "diagnosis",
        "churn",
        "fraud",
    ]

    for word in target_words:

        for column in scalar_columns:

            if str(column).lower().strip() == word:

                return column

    # Prefer low-cardinality categorical column.
    candidates = []

    for column in scalar_columns:

        unique_count = df[column].nunique(
            dropna=True
        )

        if (
            unique_count >= 2
            and unique_count <= MAX_CLASSES
        ):

            candidates.append(
                (column, unique_count)
            )

    if candidates:

        return candidates[-1][0]

    # Last column fallback.
    if scalar_columns:
        return scalar_columns[-1]

    return None


# ---------------------------------------------------------
# FEATURE SELECTION + MODEL EVALUATION
# ---------------------------------------------------------

def build_pipeline(
    pipeline_name,
    numeric_columns,
    categorical_columns,
    selector
):

    transformers = []

    if numeric_columns:

        transformers.append(
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    ),
                ]),
                numeric_columns,
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    if pipeline_name == "Pipeline A":

        model = LogisticRegression(
            max_iter=500,
            n_jobs=None
        )

    elif pipeline_name == "Pipeline B":

        model = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            max_depth=14
        )

    else:

        model = ExtraTreesClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            max_depth=14
        )

    steps = [
        ("preprocessing", preprocessor)
    ]

    if selector is not None:

        steps.append(
            ("feature_selection", selector)
        )

    steps.append(
        ("model", model)
    )

    return Pipeline(steps)


def build_model_ready_dataframe(df, pipeline_name):
    """Transform cleaned tabular data with the recommended ML pipeline."""
    if not SKLEARN_AVAILABLE:
        raise ValueError("scikit-learn is required to prepare model-ready data.")

    if df is None or df.empty:
        raise ValueError("The dataset is empty.")

    df = df.copy()
    if df.shape[1] > MAX_FEATURE_COLUMNS:
        numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
        other_columns = [column for column in df.columns if column not in numeric_columns]
        retained_columns = list(dict.fromkeys(numeric_columns[:MAX_FEATURE_COLUMNS] + other_columns[:MAX_FEATURE_COLUMNS]))[:MAX_FEATURE_COLUMNS]
        df = df[retained_columns]

    target = infer_target_column(df)
    if target is None or target not in df.columns:
        raise ValueError("Could not identify a target column for model-ready data.")

    working = df.dropna(subset=[target]).copy()
    nested_columns = [column for column in working.columns if has_nested_data_values(working[column])]
    if nested_columns:
        working = working.drop(columns=nested_columns)

    if target not in working.columns or len(working) < 30:
        raise ValueError("At least 30 labeled rows are required for model-ready data.")

    labels = working[target].astype(str)
    if labels.nunique() < 2 or labels.nunique() > MAX_CLASSES:
        raise ValueError("The target must contain between 2 and the supported maximum number of classes.")

    features = working.drop(columns=[target])
    constant_columns = [column for column in features.columns if features[column].nunique(dropna=False) <= 1]
    if constant_columns:
        features = features.drop(columns=constant_columns)
    if features.empty:
        raise ValueError("No usable feature columns remain after preprocessing.")

    numeric_columns = list(features.select_dtypes(include=[np.number]).columns)
    categorical_columns = list(features.select_dtypes(exclude=[np.number]).columns)
    encoded_labels = LabelEncoder().fit_transform(labels)
    k = min(15, max(1, features.shape[1]))
    selector = SelectKBest(score_func=mutual_info_classif, k=k) if pipeline_name == "Pipeline A" else None

    if pipeline_name not in {"Pipeline A", "Pipeline B", "Pipeline C"}:
        pipeline_name = "Pipeline B"

    pipeline = build_pipeline(pipeline_name, numeric_columns, categorical_columns, selector)
    pipeline.fit(features, encoded_labels)

    preprocessor = pipeline.named_steps["preprocessing"]
    transformed = preprocessor.transform(features)
    transformed_names = preprocessor.get_feature_names_out()
    if "feature_selection" in pipeline.named_steps:
        selector_step = pipeline.named_steps["feature_selection"]
        transformed = selector_step.transform(transformed)
        transformed_names = transformed_names[selector_step.get_support()]

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    prepared = pd.DataFrame(transformed, columns=[str(name) for name in transformed_names], index=working.index)
    prepared["target"] = working[target].astype(str)
    return prepared


def evaluate_tabular_dataset(df):

    empty_result = {
        "status": "unavailable",
        "reason": "",
        "target_column": None,
        "selected_features": [],
        "pipelines": [],
        "best_pipeline": None,
        "model": None,
        "training_seconds": 0,
    }

    if not SKLEARN_AVAILABLE:

        empty_result["reason"] = (
            "scikit-learn is not installed. "
            "Run: pip install scikit-learn"
        )

        return empty_result

    if df is None or df.empty:

        empty_result["reason"] = (
            "Dataset is empty."
        )

        return empty_result

    # -----------------------------------------------------
    # LIMIT NUMBER OF COLUMNS
    # -----------------------------------------------------

    if df.shape[1] > MAX_FEATURE_COLUMNS:

        # Keep columns with useful variance.
        numeric_cols = list(
            df.select_dtypes(
                include=[np.number]
            ).columns
        )

        other_cols = [
            c for c in df.columns
            if c not in numeric_cols
        ]

        selected_columns = (
            numeric_cols[:MAX_FEATURE_COLUMNS]
            + other_cols[:MAX_FEATURE_COLUMNS]
        )

        selected_columns = list(
            dict.fromkeys(selected_columns)
        )[:MAX_FEATURE_COLUMNS]

        df = df[selected_columns]

    target = infer_target_column(df)

    if target is None:

        empty_result["reason"] = (
            "Could not automatically detect target column."
        )

        return empty_result

    if target not in df.columns:

        empty_result["reason"] = (
            "Detected target column is unavailable."
        )

        return empty_result

    # Remove rows without target.
    df = df.dropna(
        subset=[target]
    )

    if len(df) < 30:

        empty_result["reason"] = (
            "At least 30 labelled rows are required "
            "for model comparison."
        )

        return empty_result

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    y = df[target].astype(str)

    if y.nunique() < 2:

        empty_result["reason"] = (
            "Target contains only one class."
        )

        return empty_result

    if y.nunique() > MAX_CLASSES:

        empty_result["reason"] = (
            "Target has too many classes for "
            "the automatic classification comparison."
        )

        return empty_result

    # -----------------------------------------------------
    # REPRESENTATIVE SAMPLE
    # -----------------------------------------------------

    if len(df) > MAX_ML_ROWS:

        # Stratified sample if possible.
        try:

            _, df = train_test_split(
                df,
                test_size=MAX_ML_ROWS,
                random_state=42,
                stratify=y
            )

        except Exception:

            df = df.sample(
                n=MAX_ML_ROWS,
                random_state=42
            )

        y = df[target].astype(str)

    X = df.drop(
        columns=[target]
    )

    # Nested JSON/Parquet values are retained in profiling and exports,
    # but sklearn's tabular encoders cannot use lists or dictionaries as
    # scalar features. Exclude those columns from model evaluation.
    nested_feature_columns = [
        column for column in X.columns
        if has_nested_data_values(X[column])
    ]
    if nested_feature_columns:
        X = X.drop(columns=nested_feature_columns)

    # Remove columns with no information.
    constant_columns = [
        col
        for col in X.columns
        if X[col].nunique(dropna=False) <= 1
    ]

    if constant_columns:

        X = X.drop(
            columns=constant_columns
        )

    if X.shape[1] == 0:

        empty_result["reason"] = (
            "No usable feature columns remain."
        )

        return empty_result

    numeric_columns = list(
        X.select_dtypes(
            include=[np.number]
        ).columns
    )

    categorical_columns = list(
        X.select_dtypes(
            exclude=[np.number]
        ).columns
    )

    # Encode target.
    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    # Train/test split.
    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded
        )

    except Exception:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42
        )

    # -----------------------------------------------------
    # FEATURE SELECTION
    # -----------------------------------------------------

    k = min(
        15,
        max(1, X.shape[1])
    )

    pipelines = []

    definitions = [
        (
            "Pipeline A",
            "Imputation + Scaling + Mutual Information",
            SelectKBest(
                score_func=mutual_info_classif,
                k=k
            ),
        ),
        (
            "Pipeline B",
            "Imputation + Scaling + Tree Feature Selection",
            None,
        ),
        (
            "Pipeline C",
            "Imputation + Scaling + Extra Trees",
            None,
        ),
    ]

    for pipeline_name, description, selector in definitions:

        start = time.perf_counter()

        try:

            pipeline = build_pipeline(
                pipeline_name,
                numeric_columns,
                categorical_columns,
                selector
            )

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            balanced = balanced_accuracy_score(
                y_test,
                predictions
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            # Extract selected feature information where possible.
            selected_features = []

            try:

                preprocessing = pipeline.named_steps[
                    "preprocessing"
                ]

                transformed_names = (
                    preprocessing
                    .get_feature_names_out()
                )

                if (
                    "feature_selection"
                    in pipeline.named_steps
                ):

                    selector_object = pipeline.named_steps[
                        "feature_selection"
                    ]

                    mask = selector_object.get_support()

                    selected_features = [
                        str(name)
                        for name, keep
                        in zip(
                            transformed_names,
                            mask
                        )
                        if keep
                    ]

                else:

                    # For tree pipelines, show top original
                    # numerical/categorical transformed features.
                    model = pipeline.named_steps[
                        "model"
                    ]

                    if hasattr(
                        model,
                        "feature_importances_"
                    ):

                        importances = (
                            model.feature_importances_
                        )

                        pairs = list(
                            zip(
                                transformed_names,
                                importances
                            )
                        )

                        pairs.sort(
                            key=lambda x: x[1],
                            reverse=True
                        )

                        selected_features = [
                            str(name)
                            for name, _ in pairs[:k]
                        ]

            except Exception as feature_error:

                print(
                    "Feature extraction warning:",
                    feature_error
                )

            pipelines.append({
                "name": pipeline_name,
                "description": description,
                "accuracy": round(
                    float(accuracy),
                    5
                ),
                "f1_weighted": round(
                    float(f1),
                    5
                ),
                "balanced_accuracy": round(
                    float(balanced),
                    5
                ),
                "training_seconds": round(
                    float(elapsed),
                    3
                ),
                "selected_feature_count": len(
                    selected_features
                ),
                "selected_features": selected_features[
                    :30
                ],
            })

        except Exception as pipeline_error:

            print(
                f"{pipeline_name} failed:",
                pipeline_error
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            pipelines.append({
                "name": pipeline_name,
                "description": description,
                "accuracy": None,
                "f1_weighted": None,
                "balanced_accuracy": None,
                "training_seconds": round(
                    float(elapsed),
                    3
                ),
                "selected_feature_count": 0,
                "selected_features": [],
                "error": str(pipeline_error),
            })

    valid = [
        p
        for p in pipelines
        if p.get("accuracy") is not None
    ]

    if not valid:

        empty_result["reason"] = (
            "All automatic model pipelines failed."
        )

        empty_result["pipelines"] = pipelines

        return empty_result

    best = max(
        valid,
        key=lambda item: (
            item["balanced_accuracy"],
            item["accuracy"],
            item["f1_weighted"],
        )
    )

    return {
        "status": "complete",
        "target_column": str(target),
        "sample_rows": len(df),
        "original_rows": int(len(df)),
        "feature_count_before": int(X.shape[1]),
        "feature_count_after": int(
            best["selected_feature_count"]
        ),
        "selected_features": best[
            "selected_features"
        ],
        "pipelines": pipelines,
        "best_pipeline": best["name"],
        "best_score": best["accuracy"],
        "model": best["name"],
        "training_seconds": best[
            "training_seconds"
        ],
        "metric": "accuracy",
    }


# ---------------------------------------------------------
# BEFORE / AFTER MODEL COMPARISON
# ---------------------------------------------------------

def run_before_after(df, cleaned_data=None, cleaning_summary=None, cleaned_evaluation=None):

    if df is None or df.empty:

        return {
            "status": "unavailable",
            "before": None,
            "after": None,
            "change": None,
        }

    # Use a representative sample for expensive comparison.
    working = df.copy()

    if len(working) > MAX_DEMO_ROWS:

        working = working.sample(
            n=MAX_DEMO_ROWS,
            random_state=42
        )

    target = infer_target_column(
        working
    )

    if target is None:

        return {
            "status": "unavailable",
            "reason": "Target column could not be detected.",
        }

    try:

        before = evaluate_tabular_dataset(
            working
        )

        if cleaned_data is None or cleaning_summary is None:
            cleaned_data, cleaning_summary = clean_dataframe(working)

        after = cleaned_evaluation or evaluate_tabular_dataset(cleaned_data)

        before_score = before.get(
            "best_score"
        )

        after_score = after.get(
            "best_score"
        )

        change = None

        if (
            before_score is not None
            and after_score is not None
        ):

            change = round(
                after_score - before_score,
                5
            )

        return {
            "status": "complete",
            "target_column": target,
            "cleaning": cleaning_summary,
            "before": before,
            "after": after,
            "change": change,
        }

    except Exception as e:

        print(
            "Before/after evaluation error:",
            e
        )

        return {
            "status": "unavailable",
            "reason": str(e),
        }


# ---------------------------------------------------------
# MODEL RECOMMENDATION
# ---------------------------------------------------------

def recommend_model(
    dataset_type,
    profile,
    evaluation
):

    if dataset_type == "tabular":

        best = evaluation.get(
            "best_pipeline"
        )

        if best:

            model_names = {
                "Pipeline A": "Logistic Regression",
                "Pipeline B": "Random Forest",
                "Pipeline C": "Extra Trees Classifier",
            }

            return {
                "model": model_names.get(best, best),
                "reason": (
                    "This model achieved the strongest "
                    "validation performance among the "
                    "automatically tested models."
                ),
            }

        return {
            "model": "Random Forest",
            "reason": (
                "Suitable general-purpose model for "
                "mixed tabular data."
            ),
        }

    if dataset_type == "image":

        return {
            "model": "MobileNetV3 / Transfer Learning",
            "reason": (
                "Recommended for labelled image datasets "
                "because transfer learning can provide "
                "good accuracy with lower training cost "
                "than training a CNN from scratch."
            ),
        }

    if dataset_type == "pdf":

        return {
            "model": "Text classification after OCR/extraction",
            "reason": (
                "PDF files must first be converted into "
                "structured text/features."
            ),
        }

    if dataset_type == "text":

        return {
            "model": "TF-IDF + Logistic Regression",
            "reason": (
                "Fast baseline for text classification."
            ),
        }

    if dataset_type == "notebook":

        return {
            "model": "Dataset-dependent",
            "reason": (
                "Notebook files are source documents; "
                "the underlying dataset must be extracted "
                "before ML training."
            ),
        }

    if dataset_type == "zip":

        return {
            "model": "Depends on extracted dataset",
            "reason": (
                "ZIP is an archive. DataOpti first identifies "
                "the contents and then chooses a model."
            ),
        }

    return {
        "model": "Dataset-dependent",
        "reason": (
            "DataOpti needs structured data before "
            "automatic model comparison."
        ),
    }


# ---------------------------------------------------------
# DATASET TYPE
# ---------------------------------------------------------

def detect_type(filename):

    ext = os.path.splitext(
        filename.lower()
    )[1]

    if ext in [
        ".csv",
        ".xlsx",
        ".xls",
        ".json",
        ".parquet",
    ]:
        return "tabular"

    if ext == ".pdf":
        return "pdf"

    if ext in IMAGE_EXTENSIONS:
        return "image"

    if ext == ".zip":
        return "zip"

    if ext == ".ipynb":
        return "notebook"

    if ext in [
        ".txt",
        ".py",
        ".md",
        ".log",
    ]:
        return "text"

    return "unknown"


# ---------------------------------------------------------
# UPLOAD API
# ---------------------------------------------------------

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_files():

    started = time.perf_counter()

    try:

        if "files" not in request.files:

            return jsonify({
                "status": "error",
                "message": (
                    "No file field found in request."
                ),
            }), 400

        files = request.files.getlist(
            "files"
        )

        if (
            not files
            or files[0].filename == ""
        ):

            return jsonify({
                "status": "error",
                "message": "No files selected.",
            }), 400

        total_rows = 0
        total_cols = 0
        total_missing = 0
        total_duplicates = 0
        total_outliers = 0

        all_columns = []

        processed_files = []
        skipped_files = []

        primary_evaluation = None
        primary_cleaning = None
        primary_profile = None
        primary_type = None
        primary_ml_sample_limit = MAX_ML_ROWS

        # -------------------------------------------------
        # PROCESS EACH UPLOAD
        # -------------------------------------------------

        for file in files:

            filename = (
                file.filename or "unknown"
            )

            lower_name = filename.lower()

            if not filename:
                continue

            dataset_type = detect_type(
                filename
            )

            profile = None
            df_for_ml = None

            try:

                # -----------------------------------------
                # TABULAR
                # -----------------------------------------

                if dataset_type == "tabular":

                    profile = profile_tabular_file(
                        file,
                        filename
                    )

                    # Load only a representative sample
                    # for cleaning + ML.
                    df_for_ml = load_tabular_sample(
                        file,
                        filename,
                        MAX_ML_ROWS
                    )

                    if profile:

                        total_rows += safe_int(
                            profile.get("rows")
                        )

                        total_cols += safe_int(
                            profile.get("columns")
                        )

                        total_missing += safe_int(
                            profile.get(
                                "missing_values"
                            )
                        )

                        total_duplicates += safe_int(
                            profile.get(
                                "duplicate_rows"
                            )
                        )

                        total_outliers += safe_int(
                            profile.get(
                                "outliers"
                            )
                        )

                        all_columns.extend(
                            profile.get(
                                "column_names",
                                []
                            )
                        )

                    # Run full workflow only for the
                    # primary tabular dataset.
                    if (
                        df_for_ml is not None
                        and primary_evaluation is None
                    ):

                        primary_type = "tabular"
                        primary_profile = profile
                        primary_ml_sample_limit = safe_int(
                            df_for_ml.attrs.get("ml_sample_limit"),
                            MAX_ML_ROWS,
                        )

                        cleaned_df, cleaning = (
                            clean_dataframe(
                                df_for_ml
                            )
                        )

                        primary_cleaning = cleaning

                        evaluation = (
                            evaluate_tabular_dataset(
                                cleaned_df
                            )
                        )

                        before_after = (
                            run_before_after(
                                df_for_ml,
                                cleaned_data=cleaned_df,
                                cleaning_summary=cleaning,
                                cleaned_evaluation=evaluation,
                            )
                        )

                        primary_evaluation = {
                            "cleaned": evaluation,
                            "before_after": before_after,
                        }

                # -----------------------------------------
                # PDF
                # -----------------------------------------

                elif dataset_type == "pdf":

                    profile = parse_pdf(
                        file
                    )

                    if profile.get(
                        "status"
                    ) == "skipped":

                        skipped_files.append({
                            "filename": filename,
                            "type": "pdf",
                            "reason": profile.get(
                                "reason",
                                "PDF could not be processed."
                            ),
                        })

                    else:

                        total_rows += safe_int(
                            profile.get("rows")
                        )

                        total_cols += safe_int(
                            profile.get("columns")
                        )

                        all_columns.extend(
                            profile.get(
                                "column_names",
                                []
                            )
                        )

                        if primary_type is None:

                            primary_type = "pdf"
                            primary_profile = profile

                # -----------------------------------------
                # IMAGE
                # -----------------------------------------

                elif dataset_type == "image":

                    profile = parse_image(
                        file
                    )

                    if profile.get(
                        "status"
                    ) == "skipped":

                        skipped_files.append({
                            "filename": filename,
                            "type": "image",
                            "reason": profile.get(
                                "reason",
                                "Image could not be processed."
                            ),
                        })

                    else:

                        total_rows += 1
                        total_cols += 3

                        all_columns.extend(
                            profile.get(
                                "column_names",
                                []
                            )
                        )

                        if primary_type is None:

                            primary_type = "image"
                            primary_profile = profile

                # -----------------------------------------
                # ZIP
                # -----------------------------------------

                elif dataset_type == "zip":

                    profile = parse_zip(
                        file
                    )

                    if profile.get(
                        "status"
                    ) == "skipped":

                        skipped_files.append({
                            "filename": filename,
                            "type": "zip",
                            "reason": profile.get(
                                "reason",
                                "ZIP could not be processed."
                            ),
                        })

                    else:

                        all_columns.extend(
                            profile.get(
                                "column_names",
                                []
                            )
                        )

                        if primary_type is None:

                            primary_type = "zip"
                            primary_profile = profile

                # -----------------------------------------
                # TEXT / NOTEBOOK
                # -----------------------------------------

                elif dataset_type in [
                    "text",
                    "notebook",
                ]:

                    profile = parse_text_or_code(
                        file,
                        filename
                    )

                    if profile.get(
                        "status"
                    ) == "skipped":

                        skipped_files.append({
                            "filename": filename,
                            "type": dataset_type,
                            "reason": profile.get(
                                "reason",
                                "File could not be processed."
                            ),
                        })

                    else:

                        total_rows += safe_int(
                            profile.get("rows")
                        )

                        total_cols += safe_int(
                            profile.get("columns")
                        )

                        all_columns.extend(
                            profile.get(
                                "column_names",
                                []
                            )
                        )

                        if primary_type is None:

                            primary_type = dataset_type
                            primary_profile = profile

                # -----------------------------------------
                # UNKNOWN
                # -----------------------------------------

                else:

                    skipped_files.append({
                        "filename": filename,
                        "type": "unknown",
                        "reason": (
                            "Unsupported file extension."
                        ),
                    })

            except Exception as file_error:

                print(
                    f"Could not process {filename}: "
                    f"{file_error}"
                )

                skipped_files.append({
                    "filename": filename,
                    "type": dataset_type,
                    "reason": str(file_error),
                })

        # -------------------------------------------------
        # NOTHING PROCESSED
        # -------------------------------------------------

        if (
            primary_type is None
            and not processed_files
            and not primary_profile
            and not skipped_files
        ):

            return jsonify({
                "status": "error",
                "message": (
                    "Could not process the uploaded files."
                ),
            }), 400

        # -------------------------------------------------
        # MODEL RESULTS
        # -------------------------------------------------

        model_results = None

        if (
            primary_type == "tabular"
            and primary_evaluation
        ):

            cleaned_eval = (
                primary_evaluation.get(
                    "cleaned",
                    {}
                )
            )

            before_after = (
                primary_evaluation.get(
                    "before_after",
                    {}
                )
            )

            best_pipeline = (
                cleaned_eval.get(
                    "best_pipeline"
                )
            )

            selected_features = (
                cleaned_eval.get(
                    "selected_features",
                    []
                )
            )

            model_results = {
                "status": cleaned_eval.get(
                    "status",
                    "unavailable"
                ),
                "target_column": cleaned_eval.get(
                    "target_column"
                ),
                "pipelines": cleaned_eval.get(
                    "pipelines",
                    []
                ),
                "best_pipeline": best_pipeline,
                "best_score": cleaned_eval.get(
                    "best_score"
                ),
                "selected_features": selected_features,
                "feature_count_before": cleaned_eval.get(
                    "feature_count_before"
                ),
                "feature_count_after": cleaned_eval.get(
                    "feature_count_after"
                ),
                "sample_rows": cleaned_eval.get(
                    "sample_rows"
                ),
                "training_seconds": cleaned_eval.get(
                    "training_seconds"
                ),
                "accuracy_comparison": {
                    "metric": "accuracy",
                    "before": (
                        before_after
                        .get("before", {})
                        .get("best_score")
                    ),
                    "after": (
                        before_after
                        .get("after", {})
                        .get("best_score")
                    ),
                    "change": before_after.get(
                        "change"
                    ),
                },
            }

        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        recommendation = recommend_model(
            primary_type,
            primary_profile or {},
            (
                primary_evaluation
                .get("cleaned", {})
                if primary_evaluation
                else {}
            )
        )

        # -------------------------------------------------
        # CLEANING SUMMARY
        # -------------------------------------------------

        cleaning_result = (
            primary_cleaning
            if primary_cleaning
            else {
                "rows_before": total_rows,
                "rows_after": total_rows,
                "missing_before": total_missing,
                "missing_after": total_missing,
                "duplicates_before": total_duplicates,
                "duplicates_after": total_duplicates,
                "outliers_before": total_outliers,
                "outliers_after": total_outliers,
            }
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            # Upload details
            "upload": {
                "file_count": len(files),
                "processed_count": (
                    len(files)
                    - len(skipped_files)
                ),
                "skipped_count": len(
                    skipped_files
                ),
                "processing_seconds": round(
                    elapsed,
                    3
                ),
            },

            # Dataset details
            "dataset_type": primary_type,

            "rows": total_rows,

            "columns": (
                total_cols
                if total_cols > 0
                else len(all_columns)
            ),

            "missing_values": total_missing,

            "duplicate_rows": total_duplicates,

            "outliers": total_outliers,

            "column_names": list(
                dict.fromkeys(
                    all_columns
                )
            )[:300],

            # Cleaning
            "cleaning": cleaning_result,

            # Feature selection / ML
            "model_results": model_results,

            # Recommendation
            "recommendation": recommendation,

            # Files that failed independently
            "skipped_files": skipped_files,

            # Performance
            "performance": {
                "processing_seconds": round(
                    elapsed,
                    3
                ),
                "ml_sample_limit": primary_ml_sample_limit,
                "profile_sampled": bool(
                    (primary_profile or {}).get("profile_sampled", False)
                ),
                "profile_sample_rows": (primary_profile or {}).get(
                    "profile_sample_rows"
                ),
                "large_dataset_strategy": (
                    "Representative profiling + "
                    "bounded ML sampling"
                ),
            },

        }), 200

    except RequestEntityTooLarge:

        return jsonify({
            "status": "error",
            "message": (
                f"Upload exceeds {MAX_GB} GB server limit."
            ),
        }), 413

    except Exception as err:

        print(
            "\n--- UNHANDLED EXCEPTION ---"
        )

        traceback.print_exc()

        print(
            "---------------------------\n"
        )

        return jsonify({
            "status": "error",
            "message": str(err),
        }), 500


@app.route(
    "/api/download-cleaned",
    methods=["POST"]
)
def download_cleaned_data():
    """Export model-ready tabular samples or extracted data for supported files."""

    files = request.files.getlist("files")

    if not files:
        return jsonify({
            "status": "error",
            "message": "Upload a supported file to download prepared data.",
        }), 400

    exports = []
    errors = []

    def csv_bytes(rows):
        return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")

    def dataframe_csv(dataframe):
        export_frame = dataframe.copy()

        def csv_value(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple, dict)):
                return json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, np.generic):
                return value.item()
            return value

        for column in export_frame.columns:
            export_frame[column] = export_frame[column].map(csv_value)

        return export_frame.to_csv(index=False).encode("utf-8-sig")

    def safe_base_name(filename):
        base = os.path.splitext(os.path.basename(filename))[0]
        return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "dataset"

    for file_index, file in enumerate(files, start=1):
        filename = file.filename or "dataset"
        dataset_type = detect_type(filename)
        output_rows = None
        output_content = None
        output_prefix = "prepared"

        try:
            if dataset_type == "tabular":
                dataframe = load_tabular_sample(file, filename, MAX_ML_ROWS)
                if dataframe is None:
                    raise ValueError("Could not read this tabular file.")

                cleaned_dataframe, _ = clean_dataframe(dataframe)
                if cleaned_dataframe is None:
                    raise ValueError("The dataset could not be cleaned.")

                pipeline_name = request.form.get("model_pipeline", "Pipeline B")
                model_ready_dataframe = build_model_ready_dataframe(cleaned_dataframe, pipeline_name)
                output_content = dataframe_csv(model_ready_dataframe)
                output_prefix = "model_ready"

            elif dataset_type == "text":
                file.stream.seek(0)
                text = file.stream.read(5 * 1024 * 1024).decode("utf-8", errors="ignore")
                output_rows = [
                    {"line": line_number, "text": line.strip()}
                    for line_number, line in enumerate(text.splitlines(), start=1)
                    if line.strip()
                ]

            elif dataset_type == "notebook":
                notebook_parse_limit = 20 * 1024 * 1024
                notebook_preview_limit = 5 * 1024 * 1024
                file.stream.seek(0, os.SEEK_END)
                notebook_size = file.stream.tell()
                file.stream.seek(0)
                read_limit = notebook_parse_limit if notebook_size <= notebook_parse_limit else notebook_preview_limit
                raw_text = file.stream.read(read_limit).decode("utf-8", errors="replace")
                try:
                    notebook = json.loads(raw_text)
                except json.JSONDecodeError:
                    # Large or malformed notebooks can be truncated by the
                    # export preview limit. Preserve readable content instead
                    # of failing the entire download.
                    output_rows = []
                    for line_number, line in enumerate(raw_text.splitlines(), start=1):
                        line = line.strip()
                        for part_index in range(0, len(line), 32000):
                            output_rows.append({
                                "line": line_number,
                                "part": part_index // 32000 + 1,
                                "text": line[part_index:part_index + 32000],
                            })
                else:
                    output_rows = []
                    for index, cell in enumerate(notebook.get("cells", []), start=1):
                        source = cell.get("source", "")
                        source = "".join(source) if isinstance(source, list) else str(source)
                        if source.strip():
                            output_rows.append({
                                "cell": index,
                                "type": cell.get("cell_type", "unknown"),
                                "source": source,
                            })

            elif dataset_type == "pdf":
                if pypdf is None:
                    raise ValueError("PDF text export needs pypdf installed.")
                file.stream.seek(0)
                reader = pypdf.PdfReader(file.stream, strict=False)
                output_rows = [
                    {"page": page_index, "text": " ".join((page.extract_text() or "").split())}
                    for page_index, page in enumerate(reader.pages, start=1)
                ]

            elif dataset_type == "image":
                if Image is None:
                    raise ValueError("Image metadata export needs Pillow installed.")
                file.stream.seek(0)
                with Image.open(file.stream) as image:
                    output_rows = [{
                        "filename": os.path.basename(filename),
                        "format": image.format or "unknown",
                        "width": image.width,
                        "height": image.height,
                        "mode": image.mode,
                    }]

            elif dataset_type == "zip":
                file.stream.seek(0)
                with zipfile.ZipFile(file.stream) as archive:
                    output_rows = [
                        {
                            "path": item.filename,
                            "size_bytes": item.file_size,
                            "compressed_size_bytes": item.compress_size,
                            "extension": os.path.splitext(item.filename)[1].lower(),
                        }
                        for item in archive.infolist()
                        if not item.is_dir()
                    ]
            else:
                errors.append(f"{filename}: unsupported file type")
                continue

            output_name = f"{file_index:02d}_{output_prefix}_{safe_base_name(filename)}.csv"
            if output_content is None:
                output_content = csv_bytes(output_rows or [])
            exports.append((output_name, output_content))

        except Exception as export_error:
            errors.append(f"{filename}: {export_error}")

    if not exports:
        message = "No prepared data could be exported. " + "; ".join(errors)
        return jsonify({"status": "error", "message": message}), 400

    if len(exports) == 1:
        output_name, content = exports[0]
        output = io.BytesIO(content)
        return send_file(
            output,
            mimetype="text/csv",
            as_attachment=True,
            download_name=output_name.split("_", 1)[1],
        )

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output_name, content in exports:
            archive.writestr(output_name, content)

    archive_buffer.seek(0)
    return send_file(
        archive_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name="dataopti_prepared_exports.zip",
    )


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.route("/")
def home():

    return jsonify({
        "status": "success",
        "message": "DataOpti AI backend is running.",
    })


# ---------------------------------------------------------
# START
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "DataOpti AI backend: "
        "http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
