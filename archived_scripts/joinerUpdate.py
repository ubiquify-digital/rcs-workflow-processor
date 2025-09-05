import cv2
import os
from tqdm import tqdm

# Folder with frames
frames_dir = "output_frames/frames"
output_video = "output_video.mp4"
fps = 2  # match the fps you used in InferencePipeline

# Get sorted list of frame files
frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

# Read the first frame to get video dimensions
first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
height, width, layers = first_frame.shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # for .mp4
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write all frames
for frame_file in tqdm(frame_files, desc="Processing frames"):
    frame_path = os.path.join(frames_dir, frame_file)
    img = cv2.imread(frame_path)
    video_writer.write(img)

video_writer.release()
print(f"Video saved as {output_video}")
