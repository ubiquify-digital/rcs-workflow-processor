# Import the InferencePipeline object
from inference import InferencePipeline
import cv2
import os

# Create a folder to store frames
os.makedirs("frames", exist_ok=True)

frame_counter = 0  # global counter

def my_sink(result, video_frame):
    global frame_counter
    if result.get("output_image"):  # If workflow returned an image
        filename = f"frames/output_{frame_counter:04d}.png"
        cv2.imwrite(filename, result["output_image"].numpy_image)
        frame_counter += 1
    print(result)  # do something with predictions

# initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="MQ1Wd6PJGMPBMCvxsCS6",
    workspace_name="rcs-k9i1w",
    workflow_id="license-plate",
    video_reference="test.mov",
    max_fps=30,
    on_prediction=my_sink
)

pipeline.start()
pipeline.join()
