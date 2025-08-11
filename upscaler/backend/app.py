import os
import uuid
from typing import Tuple

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np
try:
    import cv2  # Optional: used for cartoonize, object removal, etc.
except Exception:
    cv2 = None
try:
    from rembg import remove as rembg_remove  # Optional: background removal
except Exception:
    rembg_remove = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clamp_scale(value: int) -> int:
    try:
        scale = int(value)
    except Exception:
        scale = 2
    return max(1, min(scale, 4))


def clamp_enhance(value: float) -> float:
    try:
        amount = float(value)
    except Exception:
        amount = 1.0
    return max(0.5, min(amount, 2.0))


def preprocess_image(image: Image.Image) -> Image.Image:
    # Auto contrast and denoise (simple median filter)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return image

def postprocess_image(image: Image.Image, super_enhance: bool = False) -> Image.Image:
    # Sharpen, color balance, and optionally more aggressive enhancement
    if super_enhance:
        image = ImageEnhance.Sharpness(image).enhance(2.0)
        image = ImageEnhance.Color(image).enhance(1.3)
        image = image.filter(ImageFilter.DETAIL)
    else:
        image = ImageEnhance.Sharpness(image).enhance(1.2)
        image = ImageEnhance.Color(image).enhance(1.1)
    return image

# Initialize Real-ESRGAN model at app startup (optional dependency)
try:
    # Import here so that deployment works even if realesrgan/torch are not installed
    from realesrgan import RealESRGANer
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(BASE_DIR, 'RealESRGANv2-animevideo-xsx4.pth')
    realesrgan_model = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        device=device
    )
except Exception as e:
    print(f"[WARN] Real-ESRGAN model could not be loaded: {e}")
    realesrgan_model = None


def upscale_image(image: Image.Image, scale: int, enhance_amount: float, super_enhance: bool = False) -> Image.Image:
    global realesrgan_model
    # Preprocessing
    image = preprocess_image(image)
    # Try Real-ESRGAN if available and scale is 2 or 4
    if realesrgan_model is not None and scale in [2, 4]:
        try:
            if scale == 4:
                upscaled = realesrgan_model.predict(image, outscale=4)
                upscaled = postprocess_image(upscaled, super_enhance)
                return upscaled
        except Exception as e:
            print(f"[WARN] Real-ESRGAN failed, falling back to PIL: {e}")
    # Fallback to PIL upscaling
    new_size: Tuple[int, int] = (int(image.width * scale), int(image.height * scale))
    resample_algo = getattr(Image, "Resampling", Image).LANCZOS
    upscaled = image.resize(new_size, resample=resample_algo)
    if enhance_amount != 1.0:
        upscaled = ImageEnhance.Sharpness(upscaled).enhance(enhance_amount)
        contrast_factor = 0.9 + (enhance_amount - 1.0) * 0.5
        upscaled = ImageEnhance.Contrast(upscaled).enhance(contrast_factor)
        upscaled = upscaled.filter(ImageFilter.DETAIL)
    # Postprocessing
    upscaled = postprocess_image(upscaled, super_enhance)
    return upscaled


app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path="",
)


@app.get("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "app": "Photoxcel"})


@app.post("/api/upscale")
def api_upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name should be 'image')."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file."}), 400

    scale = clamp_scale(request.form.get("scale", 2))
    enhance_amount = clamp_enhance(request.form.get("enhance", 1.0))
    super_enhance = request.form.get("super_enhance", "false").lower() == "true"

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Unsupported or corrupted image file."}), 400

    upscaled = upscale_image(image, scale, enhance_amount, super_enhance)

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)
    upscaled.save(save_path, format="JPEG", quality=92, optimize=True)

    return jsonify(
        {
            "output_url": f"/outputs/{filename}",
            "width": upscaled.width,
            "height": upscaled.height,
            "scale": scale,
            "enhance": enhance_amount,
            "super_enhance": super_enhance,
        }
    )


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    arr = np.array(image)
    if arr.ndim == 2:
        return arr
    # Convert RGB to BGR for OpenCV
    return arr[:, :, ::-1].copy()


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    # Convert BGR to RGB
    return Image.fromarray(image[:, :, ::-1])


def simple_background_remover(image: Image.Image) -> Image.Image:
    """Simple but effective background removal like removebg"""
    if cv2 is None:
        raise ImportError("OpenCV not available")
    
    # Convert PIL to OpenCV format
    img = pil_to_cv2(image)
    
    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Create masks for different background types
    masks = []
    
    # 1. Light background (white, light gray, etc.)
    light_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    masks.append(light_mask)
    
    # 2. Very light backgrounds (almost white)
    very_light_mask = cv2.inRange(lab, np.array([200, 128, 128]), np.array([255, 128, 128]))
    masks.append(very_light_mask)
    
    # 3. Green screen backgrounds
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    masks.append(green_mask)
    
    # 4. Blue screen backgrounds
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
    masks.append(blue_mask)
    
    # Combine all background masks
    background_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for mask in masks:
        background_mask = cv2.bitwise_or(background_mask, mask)
    
    # Invert to get foreground mask
    foreground_mask = cv2.bitwise_not(background_mask)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((3,3), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply Gaussian blur to smooth edges
    foreground_mask = cv2.GaussianBlur(foreground_mask, (5, 5), 0)
    
    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=foreground_mask)
    
    # Convert back to PIL
    result_pil = cv2_to_pil(result)
    
    # Create alpha channel from mask
    alpha = Image.fromarray(foreground_mask)
    
    # Convert to RGBA
    if result_pil.mode != 'RGBA':
        result_pil = result_pil.convert('RGBA')
    
    # Apply alpha channel
    result_pil.putalpha(alpha)
    
    return result_pil


def smart_object_remover(image: Image.Image, keep_objects: str = "people") -> Image.Image:
    """Smart object detection to keep specific objects and remove everything else"""
    if cv2 is None:
        raise ImportError("OpenCV not available")
    
    # Convert PIL to OpenCV format
    img = pil_to_cv2(image)
    
    # Create a copy for processing
    original = img.copy()
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Initialize mask
    final_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    if keep_objects == "people":
        # Method 1: Skin detection for people
        # Skin color ranges in HSV
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Additional skin range for different lighting
        lower_skin2 = np.array([170, 20, 70])
        upper_skin2 = np.array([180, 255, 255])
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        # Combine skin masks
        skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
        
        # Clean up skin mask
        kernel = np.ones((5,5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        final_mask = cv2.bitwise_or(final_mask, skin_mask)
        
        # Method 2: Face detection using Haar cascades
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Expand face region to include body
                expanded_x = max(0, x - w//2)
                expanded_y = max(0, y - h//2)
                expanded_w = min(img.shape[1] - expanded_x, w * 2)
                expanded_h = min(img.shape[0] - expanded_y, h * 3)
                
                # Create mask for expanded region
                face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                face_mask[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w] = 255
                
                # Apply gradient to face mask for smooth edges
                face_mask = cv2.GaussianBlur(face_mask, (21, 21), 0)
                final_mask = cv2.bitwise_or(final_mask, face_mask)
                
        except Exception as e:
            print(f"Face detection failed: {e}")
    
    elif keep_objects == "faces":
        # Only keep faces
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                face_mask[y:y+h, x:x+w] = 255
                face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
                final_mask = cv2.bitwise_or(final_mask, face_mask)
                
        except Exception as e:
            print(f"Face detection failed: {e}")
    
    elif keep_objects == "center":
        # Keep center object (assumes main subject is in center)
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create circular mask around center
        center_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        radius = min(width, height) // 3
        cv2.circle(center_mask, (center_x, center_y), radius, 255, -1)
        
        # Apply gradient
        center_mask = cv2.GaussianBlur(center_mask, (51, 51), 0)
        final_mask = center_mask
    
    # Clean up final mask
    kernel = np.ones((7,7), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    result = cv2.bitwise_and(original, original, mask=final_mask)
    
    # Convert back to PIL
    result_pil = cv2_to_pil(result)
    
    # Create alpha channel from mask
    alpha = Image.fromarray(final_mask)
    
    # Convert to RGBA
    if result_pil.mode != 'RGBA':
        result_pil = result_pil.convert('RGBA')
    
    # Apply alpha channel
    result_pil.putalpha(alpha)
    
    return result_pil


@app.post("/api/remove_background")
def api_remove_background():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name should be 'image')."}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty file."}), 400

    # Get keep_objects parameter
    keep_objects = request.form.get("keep_objects", "people").lower()

    try:
        # Try rembg first if available and no specific object detection requested
        if rembg_remove is not None and keep_objects == "background":
            try:
                input_bytes = file.read()
                output_bytes = rembg_remove(input_bytes)
                output_image = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
                method_used = "rembg"
            except Exception as e:
                print(f"rembg failed: {e}")
                # Fall back to smart object detection
                file.seek(0)
                input_image = Image.open(file.stream).convert("RGB")
                output_image = smart_object_remover(input_image, keep_objects)
                method_used = "smart_detection"
        else:
            # Use smart object detection
            if cv2 is None:
                return jsonify({"error": "Background removal not available. Please install 'opencv-python' or 'rembg'."}), 500
            
            input_image = Image.open(file.stream).convert("RGB")
            output_image = smart_object_remover(input_image, keep_objects)
            method_used = "smart_detection"
            
    except Exception as e:
        return jsonify({"error": f"Failed to remove background: {e}"}), 500

    filename = f"{uuid.uuid4().hex}.png"  # keep transparency
    save_path = os.path.join(OUTPUT_DIR, filename)
    output_image.save(save_path, format="PNG")

    return jsonify({"output_url": f"/outputs/{filename}", "method": method_used, "kept_objects": keep_objects})


@app.post("/api/cartoonize")
def api_cartoonize():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name should be 'image')."}), 400
    if cv2 is None:
        return jsonify({"error": "Cartoonizer not available. Please install 'opencv-python'."}), 500

    style = (request.form.get("style") or "cartoon").lower()

    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
        img = pil_to_cv2(pil_img)
        if style == "pencil":
            gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            out = color
        elif style == "watercolor":
            out = cv2.stylization(img, sigma_s=120, sigma_r=0.25)
        elif style == "ink":
            # strong edge map + bilateral smoothing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            color = cv2.bilateralFilter(img, d=9, sigmaColor=200, sigmaSpace=200)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            out = cv2.bitwise_and(color, edges_bgr)
        elif style == "comic":
            # Posterize + edges
            color = cv2.bilateralFilter(img, 9, 150, 150)
            for _ in range(2):
                color = cv2.bilateralFilter(color, 9, 75, 75)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 150)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            inv_edges = cv2.bitwise_not(edges)
            out = cv2.bitwise_and(color, inv_edges)
        else:  # default "cartoon"
            # classic cartoon: edge mask + bilateral smoothing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            color = cv2.bilateralFilter(img, 9, 200, 200)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            out = cv2.bitwise_and(color, edges_bgr)

        pil_out = cv2_to_pil(out)
    except Exception as e:
        return jsonify({"error": f"Failed to cartoonize: {e}"}), 500

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)
    pil_out.save(save_path, format="JPEG", quality=92, optimize=True)
    return jsonify({"output_url": f"/outputs/{filename}", "style": style})





@app.post("/api/resize")
def api_resize():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name should be 'image')."}), 400

    keep_aspect = (request.form.get("keep_aspect") or "true").lower() == "true"
    width = request.form.get("width")
    height = request.form.get("height")

    try:
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
        orig_w, orig_h = pil_img.width, pil_img.height
        new_w, new_h = None, None
        if width:
            new_w = max(1, int(width))
        if height:
            new_h = max(1, int(height))
        if keep_aspect:
            if new_w and not new_h:
                ratio = new_w / orig_w
                new_h = max(1, int(orig_h * ratio))
            elif new_h and not new_w:
                ratio = new_h / orig_h
                new_w = max(1, int(orig_w * ratio))
            elif not new_w and not new_h:
                return jsonify({"error": "Provide width or height."}), 400
        else:
            if not new_w or not new_h:
                return jsonify({"error": "Provide both width and height when keep_aspect=false."}), 400

        resample_algo = getattr(Image, "Resampling", Image).LANCZOS
        out_img = pil_img.resize((new_w, new_h), resample=resample_algo)
    except Exception as e:
        return jsonify({"error": f"Failed to resize: {e}"}), 500

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)
    out_img.save(save_path, format="JPEG", quality=92, optimize=True)
    return jsonify({"output_url": f"/outputs/{filename}", "width": new_w, "height": new_h})


@app.post("/api/convert")
def api_convert():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name should be 'image')."}), 400

    fmt = (request.form.get("format") or "png").lower()
    valid = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}
    if fmt not in valid:
        return jsonify({"error": "Unsupported format. Use png, jpg, jpeg, webp."}), 400

    try:
        img = Image.open(request.files["image"].stream)
        mode = img.mode
        target_format = valid[fmt]
        # Ensure correct mode for JPEG
        if target_format == "JPEG" and mode in ("RGBA", "P"):
            img = img.convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {e}"}), 400

    filename = f"{uuid.uuid4().hex}.{fmt if fmt != 'jpeg' else 'jpg'}"
    save_path = os.path.join(OUTPUT_DIR, filename)
    try:
        params = {}
        if valid[fmt] == "JPEG":
            params = {"quality": 92, "optimize": True}
        img.save(save_path, format=valid[fmt], **params)
    except Exception as e:
        return jsonify({"error": f"Failed to convert: {e}"}), 500

    return jsonify({"output_url": f"/outputs/{filename}", "format": fmt})


if __name__ == "__main__":
    # Run local dev server
    app.run(host="127.0.0.1", port=8000, debug=True)


