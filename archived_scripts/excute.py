import cv2
import json
import os
os.environ["ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING"] = "True"

import numpy as np
import datetime
from inference import InferencePipeline
import os


output_folder = "output_frames"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "frames"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "json_outputs"), exist_ok=True)

frame_counter = 0

def make_serializable(obj):
    """
    Convert custom objects (Detections, WorkflowImageData, datetime, numpy arrays) into JSON-friendly structures.
    Excludes large image data to keep JSON files manageable.
    """
    if hasattr(obj, "__dict__"):
        # Filter out image-related attributes to avoid saving large data
        filtered_dict = {}
        for k, v in obj.__dict__.items():
            # Skip image data attributes
            if k in ['_numpy_image', '_base64_image', 'numpy_image', 'base64_image']:
                filtered_dict[k] = f"<excluded_image_data_{type(v).__name__}>"
            else:
                filtered_dict[k] = make_serializable(v)
        return filtered_dict
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        # Filter out image data from dictionaries too
        filtered_dict = {}
        for k, v in obj.items():
            if k in ['_numpy_image', '_base64_image', 'numpy_image', 'base64_image']:
                filtered_dict[k] = f"<excluded_image_data_{type(v).__name__}>"
            else:
                filtered_dict[k] = make_serializable(v)
        return filtered_dict
    elif isinstance(obj, np.ndarray):
        # For numpy arrays, check if it's likely image data (large arrays)
        if obj.size > 1000:  # Skip large arrays (likely images)
            return f"<excluded_large_array_shape_{obj.shape}>"
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)  # Convert numpy integers to Python int
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)  # Convert numpy floats to Python float
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()  # convert datetime to ISO string
    else:
        return obj


def my_sink(result, video_frame):
    global frame_counter
    
    # Save the frame
    if result.get("output_image") or result.get("label_visualization"):
        cv2.imwrite(os.path.join(output_folder, "frames", f"frame_{frame_counter:05d}.png"), result["output_image"].numpy_image if result.get("output_image") else result["label_visualization"].numpy_image    )

    # Serialize the result
    serializable_result = make_serializable(result)

    # Save JSON output
    with open(os.path.join(output_folder, "json_outputs", f"frame_{frame_counter:05d}.json"), "w") as f:
        json.dump(serializable_result, f, indent=2)

    frame_counter += 1
    print(f"Saved frame {frame_counter}")


# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="MQ1Wd6PJGMPBMCvxsCS6",
    workspace_name="rcs-k9i1w",
    workflow_id="awais-detect-trash",
    video_reference="test.mov",
    max_fps=1,
    on_prediction=my_sink,
)

pipeline.start()
pipeline.join()
