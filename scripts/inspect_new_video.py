import cv2
import os

video_path = r"C:\Users\VIJAY\.gemini\antigravity\scratch\dr_screening\frontend\public\videos\12024715_1920_1080_30fps (1).mp4"
print("Exists:", os.path.exists(video_path))
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frames: {total_frames}, FPS: {fps}, Dimensions: {width}x{height}")

ret, frame = cap.read()
if ret:
    out_path = r"C:\Users\VIJAY\.gemini\antigravity\brain\0798ee77-2cd7-4133-b794-f21afa8d59e0\new_vortex_frame0.jpg"
    cv2.imwrite(out_path, frame)
    print("Saved sample frame to:", out_path)
cap.release()
