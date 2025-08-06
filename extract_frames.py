import cv2
import os

def extract_frames(video_path, output_dir, frames_per_second=1):
    """
    This script extracts a specified number of frames per second from a video file
    and saves them as individual image files in a new folder.

    These extracted frames can then be used for manual annotation and model training.

    Args:
        video_path (str): The path to the input video file (e.g., 'pills.mp4').
        output_dir (str): The directory where the extracted frames will be saved.
        frames_per_second (int): The number of frames to extract per second of video.
                                  For example, 1 means one frame every second.
    """
    # 1. Check if the video file exists at the given path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # 2. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' created or already exists.")

    # 3. Open the video file using OpenCV
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video file at '{video_path}'")
        return

    # 4. Get the video's original frame rate (FPS)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not get video frame rate.")
        vidcap.release()
        return

    # 5. Calculate the interval (in frames) at which to save images
    frame_interval = int(fps / frames_per_second)
    frame_count = 0
    saved_count = 0

    print(f"Video frame rate: {fps} FPS")
    print(f"Saving 1 frame every {frame_interval} frames...")

    # 6. Loop through the video frames
    while True:
        success, image = vidcap.read()
        if not success:
            print("Finished extracting frames from the video.")
            break

        # 7. Save the frame at the specified interval
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, image)
            print(f"Saved frame: {frame_filename}")
            saved_count += 1
        
        frame_count += 1

    # 8. Release the video capture object and print the total count
    vidcap.release()
    print(f"\nTotal frames saved: {saved_count}")

if __name__ == "__main__":
    # --- USAGE EXAMPLE ---
    # 1. Place a video file (e.g., 'pills.mp4') in your pills-detection folder.
    # 2. Update the video_filename variable below.
    # 3. Choose your desired output folder name and frames per second.
    # 4. Run this script from your VS Code terminal after activating your venv.

    # IMPORTANT: Adjust this path to your video file's name
    video_filename = 'videoplayback.mp4'
    
    # IMPORTANT: Name of the folder where the extracted frames will be saved
    output_folder_name = 'video_frames_dataset'
    
    # IMPORTANT: Number of frames to extract per second (e.g., 1 frame every second)
    fps_to_extract = 1
    
    print("Starting video frame extraction...")
    extract_frames(video_filename, output_folder_name, frames_per_second=fps_to_extract)
