import cv2
import os

IMAGE_DIR = 'generated_images/shirt_rollout'
pause_frames = 60

# Optional: set video dimensions (based on first image)
img1 = cv2.imread(os.path.join(IMAGE_DIR, "0_source_0_20250724_230127.png"))
img2 = cv2.imread(os.path.join(IMAGE_DIR, "0_source_0_20250724_230127.png"))
img3 = cv2.imread(os.path.join(IMAGE_DIR, "0_source_0_20250724_230127.png"))

# Resize to same height (optional)
height = min(img1.shape[0], img2.shape[0], img3.shape[0])
width = img1.shape[1] + img2.shape[1] + img3.shape[1]
out_size = (width, height)

image_lists=[x for x in os.listdir(IMAGE_DIR) if x.endswith('.png')]
# Video writer
out = cv2.VideoWriter('three_view_bussing.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, out_size)

for i in range(0, 48, 6):
    print(i)
    fname1 = [x for x in image_lists if x.startswith(f"{i}_source_0")][0]
    fname2 = [x for x in image_lists if x.startswith(f"{i}_source_1")][0]
    fname3 = [x for x in image_lists if x.startswith(f"{i}_source_2")][0]


    print(fname1, fname2, fname3)
    img1 = cv2.imread(os.path.join(IMAGE_DIR, fname1))
    img2 = cv2.imread(os.path.join(IMAGE_DIR, fname2))
    img3 = cv2.imread(os.path.join(IMAGE_DIR, fname3))

    # Resize if necessary
    img1 = cv2.resize(img1, (img1.shape[1], height))
    img2 = cv2.resize(img2, (img2.shape[1], height))
    img3 = cv2.resize(img3, (img3.shape[1], height))

    # Concatenate side-by-side
    combined = cv2.hconcat([img1, img2, img3])
    # Repeat frame for pause
    for _ in range(pause_frames):
        out.write(combined)

out.release()
print("Video saved as three_camera_view.mp4")