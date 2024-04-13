import openvino as ov
from openvino import get_version
import cv2
import numpy as np
import collections
import matplotlib.pyplot as plt
import time
from IPython import display
from typing import Union, Tuple, NamedTuple, Optional, List
from pathlib import Path
import threading
import os
from os import PathLike
import platform

print('''⚠️ Intel® Confidential. For Internal Use ONLY
✉️ Contact: Ryan Loney (ryan.loney@intel.com)''')

# Get OpenVINO version and print in console
installed_version = get_version()
print(f"⚙️ OpenVINO version: {installed_version}")
print(f"⚙️ Current OS: {platform.platform()}")

core = ov.Core()
ir_model_path = Path("selfie_multiclass_256x256.xml")
ov_model = core.read_model(ir_model_path)
print("✅ OpenVINO background segmentation model read from disk")

# Get available devices
devices = core.available_devices

# If NPU is available, use it, otherwise switch to CPU (for older PCs)
if 'NPU' in devices:
    device = 'NPU'
    print("✅ NPU detected - NPU set as inference device.")
else:
    device = 'CPU'
    print("🟨 NO NPU DEVICE DETECTED - Using CPU device instead.") 


compiled_model = core.compile_model(ov_model, device)
print(f"✅ OpenVINO background segmentation model loaded on {device} device.")


def load_image(path: str) -> np.ndarray:
    """
    Loads an image from `path` and returns it as BGR numpy array. `path`
    should point to an image file, either a local filename or a url. The image is
    not stored to the filesystem. Use the `download_file` function to download and
    store an image.
    :param path: Local path name or URL to image.
    :return: image as BGR numpy array
    """
    import cv2
    import requests

    if path.startswith("http"):
        # Set User-Agent to Mozilla because some websites block
        # requests with User-Agent Python
        response = requests.get(path, headers={"User-Agent": "Mozilla/5.0"})
        array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(array, -1)  # Loads the image as BGR
    else:
        image = cv2.imread(path)
    return image


# Read input image and convert it to RGB
print("✅ Loading sample PNG image")
test_image_url = "sample.png"
img = load_image(test_image_url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocessing helper function
def resize_and_pad(image:np.ndarray, height:int = 256, width:int = 256):
    """
    Input preprocessing function, takes input image in np.ndarray format, 
    resizes it to fit specified height and width with preserving aspect ratio 
    and adds padding on bottom or right side to complete target height x width rectangle.
    
    Parameters:
      image (np.ndarray): input image in np.ndarray format
      height (int, *optional*, 256): target height
      width (int, *optional*, 256): target width
    Returns:
      padded_img (np.ndarray): processed image
      padding_info (Tuple[int, int]): information about padding size, required for postprocessing
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (width, np.floor(h / (w / width)).astype(int)))
    else:
        img = cv2.resize(image, (np.floor(w / (h / height)).astype(int), height))
    
    r_h, r_w = img.shape[:2]
    right_padding = width - r_w
    bottom_padding = height - r_h
    padded_img = cv2.copyMakeBorder(img, 0, bottom_padding, 0, right_padding, cv2.BORDER_CONSTANT)
    return padded_img, (bottom_padding, right_padding)

# Apply preprocessig step - resize and pad input image
padded_img, pad_info = resize_and_pad(np.array(img))

# Convert input data from uint8 [0, 255] to float32 [0, 1] range and add batch dimension
normalized_img = np.expand_dims(padded_img.astype(np.float32) / 255, 0)


# ### Run model inference
out = compiled_model(normalized_img)[0]
print("✅ Test inference on sample image complete")


class Label(NamedTuple):
    index: int
    color: Tuple
    name: Optional[str] = None

class SegmentationMap(NamedTuple):
    labels: List

    def get_colormap(self):
        return np.array([label.color for label in self.labels])

    def get_labels(self):
        labelnames = [label.name for label in self.labels]
        if any(labelnames):
            return labelnames
        else:
            return None

def segmentation_map_to_image(
    result: np.ndarray, colormap: np.ndarray, remove_holes: bool = False
) -> np.ndarray:
    """
    Convert network result of floating point numbers to an RGB image with
    integer values from 0-255 by applying a colormap.
    :param result: A single network result after converting to pixel values in H,W or 1,H,W shape.
    :param colormap: A numpy array of shape (num_classes, 3) with an RGB value per class.
    :param remove_holes: If True, remove holes in the segmentation result.
    :return: An RGB image where each pixel is an int8 value according to colormap.
    """
    import cv2
    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )

    if len(np.unique(result)) > colormap.shape[0]:
        raise ValueError(
            f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
            "different output values. Please make sure to convert the network output to "
            "pixel values before calling this function."
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )

    return mask

# helper for visualization segmentation labels
labels = [
    Label(index=0, color=(192, 192, 192), name="background"),
    Label(index=1, color=(128, 0, 0), name="hair"),
    Label(index=2, color=(255, 229, 204), name="body skin"),
    Label(index=3, color=(255, 204, 204), name="face skin"),
    Label(index=4, color=(0, 0, 128), name="clothes"),
    Label(index=5, color=(128, 0, 128), name="others"),
]
SegmentationLabels = SegmentationMap(labels)

# helper for postprocessing output mask
def postprocess_mask(out:np.ndarray, pad_info:Tuple[int, int], orig_img_size:Tuple[int, int]):
    """
    Posptprocessing function for segmentation mask, accepts model output tensor, 
    gets labels for each pixel using argmax,
    unpads segmentation mask and resizes it to original image size.
    
    Parameters:
      out (np.ndarray): model output tensor
      pad_info (Tuple[int, int]): information about padding size from preprocessing step
      orig_img_size (Tuple[int, int]): original image height and width for resizing
    Returns:
      label_mask_resized (np.ndarray): postprocessed segmentation label mask
    """
    label_mask = np.argmax(out, -1)[0]
    pad_h, pad_w = pad_info
    unpad_h = label_mask.shape[0] - pad_h
    unpad_w = label_mask.shape[1] - pad_w
    label_mask_unpadded = label_mask[:unpad_h, :unpad_w]
    orig_h, orig_w = orig_img_size
    label_mask_resized = cv2.resize(label_mask_unpadded, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return label_mask_resized

# Get info about original image
image_data = np.array(img)
orig_img_shape = image_data.shape

# Specify background color for replacement
BG_COLOR = (192, 192, 192)

# Blur image for backgraund blurring scenario using Gaussian Blur
blurred_image = cv2.GaussianBlur(image_data, (55, 55), 0)

# Postprocess output
postprocessed_mask = postprocess_mask(out, pad_info, orig_img_shape[:2])

# Get colored segmentation map
output_mask = segmentation_map_to_image(postprocessed_mask, SegmentationLabels.get_colormap())

# Replace background on original image
# fill image with solid background color
bg_image = np.full(orig_img_shape, BG_COLOR, dtype=np.uint8)

# define condition mask for separation background and foreground
condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0
# replace background with solid color
output_image = np.where(condition, image_data, bg_image)
# replace background with blurred image copy
output_blurred_image = np.where(condition, image_data, blurred_image)

# Visualize obtained result
titles = ["Original image", "Portrait mask", "Removed background", "Blurred background"]
images = [image_data, output_mask, output_image, output_blurred_image]
figsize = (16, 16)
fig, axs = plt.subplots(2, 2, figsize=figsize, sharex='all', sharey='all')
fig.patch.set_facecolor('white')
list_axes = list(axs.flat)
for i, a in enumerate(list_axes):
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.grid(False)
    a.imshow(images[i].astype(np.uint8))
    a.set_title(titles[i])
fig.subplots_adjust(wspace=0.0, hspace=-0.8)
fig.tight_layout()

class VideoPlayer:
    """
    Custom video player to fulfill FPS requirements. You can set target FPS and output size,
    flip the video horizontally or skip first N frames.
    :param source: Video source. It could be either camera device or video file.
    :param size: Output frame size.
    :param flip: Flip source horizontally.
    :param fps: Target FPS.
    :param skip_first_frames: Skip first N frames.
    """

    def __init__(self, source, size=None, flip=False, fps=None, skip_first_frames=0):
        import cv2

        self.cv2 = cv2  # This is done to access the package in class methods
        self.__cap = cv2.VideoCapture(source)
        if not self.__cap.isOpened():
            raise RuntimeError(
                f"Cannot open {'camera' if isinstance(source, int) else ''} {source}"
            )
        # skip first N frames
        self.__cap.set(cv2.CAP_PROP_POS_FRAMES, skip_first_frames)
        # fps of input file
        self.__input_fps = self.__cap.get(cv2.CAP_PROP_FPS)
        if self.__input_fps <= 0:
            self.__input_fps = 60
        # target fps given by user
        self.__output_fps = fps if fps is not None else self.__input_fps
        self.__flip = flip
        self.__size = None
        self.__interpolation = None
        if size is not None:
            self.__size = size
            # AREA better for shrinking, LINEAR better for enlarging
            self.__interpolation = (
                cv2.INTER_AREA
                if size[0] < self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                else cv2.INTER_LINEAR
            )
        # first frame
        _, self.__frame = self.__cap.read()
        self.__lock = threading.Lock()
        self.__thread = None
        self.__stop = False

    """
    Start playing.
    """

    def start(self):
        self.__stop = False
        self.__thread = threading.Thread(target=self.__run, daemon=True)
        self.__thread.start()

    """
    Stop playing and release resources.
    """

    def stop(self):
        self.__stop = True
        if self.__thread is not None:
            self.__thread.join()
        self.__cap.release()

    def __run(self):
        prev_time = 0
        while not self.__stop:
            t1 = time.time()
            ret, frame = self.__cap.read()
            if not ret:
                break

            # fulfill target fps
            if 1 / self.__output_fps < time.time() - prev_time:
                prev_time = time.time()
                # replace by current frame
                with self.__lock:
                    self.__frame = frame

            t2 = time.time()
            # time to wait [s] to fulfill input fps
            wait_time = 1 / self.__input_fps - (t2 - t1)
            # wait until
            time.sleep(max(0, wait_time))

        self.__frame = None

    """
    Get current frame.
    """

    def next(self):
        import cv2

        with self.__lock:
            if self.__frame is None:
                return None
            # need to copy frame, because can be cached and reused if fps is low
            frame = self.__frame.copy()
        if self.__size is not None:
            frame = self.cv2.resize(frame, self.__size, interpolation=self.__interpolation)
        if self.__flip:
            frame = self.cv2.flip(frame, 1)
        return frame

# Main processing function to run background blurring
def run_background_blurring(source:Union[str, int] = 0, flip:bool = False, use_popup:bool = False, skip_first_frames:int = 0, model:ov.Model = ov_model, device:str = "CPU"):
    """
    Function for running background blurring inference on video
    Parameters:
      source (Union[str, int], *optional*, 0): input video source, it can be path or link on video file or web camera id.
      flip (bool, *optional*, False): flip output video, used for front-camera video processing
      use_popup (bool, *optional*, False): use popup window for avoid flickering
      skip_first_frames (int, *optional*, 0): specified number of frames will be skipped in video processing
      model (ov.Model): OpenVINO model for inference
      device (str): inference device
    Returns:
      None
    """
    player = None
    compiled_model = core.compile_model(model, device)
    try:
        # Create a video player to play with target fps.
        player = VideoPlayer(
            source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames
        )
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            # Get the results.
            input_image, pad_info = resize_and_pad(frame, 256, 256)
            normalized_img = np.expand_dims(input_image.astype(np.float32) / 255, 0)
           
            start_time = time.time()
            # model expects RGB image, while video capturing in BGR
            segmentation_mask = compiled_model(normalized_img[:, :, :, ::-1])[0]
            stop_time = time.time()
            blurred_image = cv2.GaussianBlur(frame, (55, 55), 0)
            postprocessed_mask = postprocess_mask(segmentation_mask, pad_info, frame.shape[:2])
            condition = np.stack((postprocessed_mask,) * 3, axis=-1) > 0
            frame = np.where(condition, frame, blurred_image)
            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )
            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname=title, mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()


# Run background segmentation via webcam (default camera is 0), try camera 1 if 0 fails
try:
    print('''✅ Launching USB webcam for live inference with OpenVINO
    ⌨️ To terminate the demo, press Ctrl+C twice''')
    run_background_blurring(source=1, use_popup=True, device=device)
finally: 
    print('''✅ Launching internal webcam for live inference with OpenVINO
    
    ⌨️ To terminate the demo, press Ctrl+C''')
    run_background_blurring(source=0, use_popup=True, device=device)
