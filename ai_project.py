# pip install gradio
# pip install setuptools
# pip install opencv-python mediapipe numpy
# pip install requests pillow

import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
from PIL import Image
import requests
import base64
from io import BytesIO
import json
import os

# API Configuration for AI Model
apiKey = 'Your Api Key' # IMPORTANT: Replace with your actual key or use environment variables
basicUrl = "https://genai.hkbu.edu.hk/general/rest"
modelName = "gpt-4-o"
apiVersion = "2024-10-21"

RECORDS_BASE_DIR = "attendance_records"
LOGS_DIR = os.path.join(RECORDS_BASE_DIR, "logs")
IMAGES_DIR = os.path.join(RECORDS_BASE_DIR, "images")
AI_ANALYSIS_DIR = os.path.join(RECORDS_BASE_DIR, "ai_analysis")

# Face Detection Setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)

def enhance_low_light(image, brightness_factor=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    v = cv2.convertScaleAbs(v, alpha=brightness_factor, beta=10) # Increased beta for more noticeable brightness
    hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return enhanced

def detect_faces(image):
    # Convert the image to RGB if it's BGR (OpenCV default)
    # Gradio image component with type="numpy" usually provides RGB
    # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Keep this if input might be BGR
    results = face_detection.process(image) # Assuming image is already RGB
    height, width, _ = image.shape
    face_locations = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * width)
            y = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            # Ensure bounding box coordinates are within image dimensions
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            face_locations.append((x, y, w, h))
    return face_locations

def anonymize_faces(image, face_locations, method="Pixelate", overlay_type="Smile"):
    anonymized = image.copy()
    for (x, y, w, h) in face_locations:
        face_roi = anonymized[y:y+h, x:x+w]
        if face_roi.size > 0: # Check if the ROI is valid
            if method == "Pixelate":
                # Ensure face_roi is not empty and w, h are positive
                if w > 0 and h > 0:
                    small = cv2.resize(face_roi, (max(1, w//10), max(1,h//10)), interpolation=cv2.INTER_LINEAR) # Adjusted pixelation block size
                    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                    anonymized[y:y+h, x:x+w] = pixelated
            elif method == "Blur":
                if w > 0 and h > 0:
                    # Kernel size must be odd
                    kernel_w = (w // 7) if (w // 7) % 2 != 0 else (w // 7) + 1
                    kernel_h = (h // 7) if (h // 7) % 2 != 0 else (h // 7) + 1
                    kernel_w = max(1, kernel_w) # Ensure kernel size is at least 1
                    kernel_h = max(1, kernel_h) # Ensure kernel size is at least 1
                    blurred = cv2.GaussianBlur(face_roi, (kernel_w, kernel_h), 30)
                    anonymized[y:y+h, x:x+w] = blurred
            elif method == "Black Box":
                anonymized[y:y+h, x:x+w] = (0,0,0) # Black color in RGB
            elif method == "Emoji Overlay":
                # Ensure emoji files 'smile.png' and 'cat.png' are in the same directory as the script,
                # or provide full paths.
                emoji_path = 'smile.png' if overlay_type == "Smile" else 'cat.png'
                if not os.path.exists(emoji_path):
                    print(f"Warning: Emoji file not found at {emoji_path}. Drawing a circle instead.")
                    # Fallback: Draw a yellow circle if emoji not found
                    # OpenCV uses BGR by default for drawing, but image is RGB. So use RGB color.
                    cv2.circle(anonymized, (x + w // 2, y + h // 2), min(w, h) // 2, (255, 255, 0), -1) # Yellow in RGB
                    continue

                emoji_img_bgr = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)

                if emoji_img_bgr is not None and w > 0 and h > 0:
                    emoji_resized_bgr = cv2.resize(emoji_img_bgr, (w, h), interpolation=cv2.INTER_AREA)
                    
                    # The main image 'anonymized' is in RGB format.
                    # The emoji read by cv2.imread is BGR or BGRA.
                    # We need to handle the channels carefully.

                    if emoji_resized_bgr.shape[2] == 4: # BGRA
                        alpha_channel = emoji_resized_bgr[:, :, 3] / 255.0
                        alpha_inv = 1.0 - alpha_channel
                        
                        # Convert emoji's BGR part to RGB before blending
                        emoji_rgb_resized_fg = cv2.cvtColor(emoji_resized_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
                        
                        for c in range(3): # Iterate through R, G, B channels
                            anonymized[y:y+h, x:x+w, c] = (alpha_channel * emoji_rgb_resized_fg[:, :, c] +
                                                           alpha_inv * anonymized[y:y+h, x:x+w, c])
                    else: # BGR (no alpha)
                        # Convert emoji from BGR to RGB
                        emoji_rgb_resized_fg = cv2.cvtColor(emoji_resized_bgr, cv2.COLOR_BGR2RGB)
                        anonymized[y:y+h, x:x+w] = emoji_rgb_resized_fg
                elif w > 0 and h > 0: # Fallback if emoji loading failed but ROI is valid
                    cv2.circle(anonymized, (x + w // 2, y + h // 2), min(w, h) // 2, (255, 255, 0), -1) # Yellow in RGB
            elif method == "Mosaic":
                tile_size = max(1, min(w, h) // 8) # Adjust tile size based on face ROI
                if tile_size > 0 and w > 0 and h > 0:
                    for ty_idx in range(0, h, tile_size):
                        for tx_idx in range(0, w, tile_size):
                            tile = face_roi[ty_idx:min(ty_idx+tile_size, h), tx_idx:min(tx_idx+tile_size, w)]
                            if tile.size > 0:
                                # Calculate mean color of the tile (already in RGB from face_roi)
                                avg_color_rgb = cv2.mean(tile)[:3] 
                                anonymized[y+ty_idx:y+min(ty_idx+tile_size,h), x+tx_idx:x+min(tx_idx+tile_size,w)] = avg_color_rgb
    return anonymized


def submit_image_to_ai(image_np, user_message="What are the people doing in this image?"):
    if image_np is None:
        return "Error: No image provided for AI analysis."
    if apiKey == 'YOUR_API_KEY':
        return "Error: API Key not configured. Please set your API key in the script."
    
    # Ensure image is in RGB format for PIL
    pil_image = Image.fromarray(image_np.astype(np.uint8), 'RGB')
    
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG") # Save as JPEG for smaller size
    img_str = base64.b64encode(buffered.getvalue()).decode()
    img_data_url = f"data:image/jpeg;base64,{img_str}"

    classroom_monitor_prompt = """
    You are a classroom monitor. Please provide the following information based on the image:
    **Classroom Occupancy**  
    - Total number of students present in the classroom (based on visual cues, not face count rectangles if any are visible).
    **Classroom Analysis**  
    - Brief assessment of the classroom's atmosphere (e.g., engagement level, noise level, behavior of students).
    - Any notable observations (e.g., student interactions, participation in activities).
    """
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for teachers providing classroom insights."},
            {"role": "user", "content": [{"type": "text", "text": classroom_monitor_prompt + "\n" + user_message},
                                         {"type": "image_url", "image_url": {"url": img_data_url, "detail": "auto"}}]} # detail: "auto" or "low" or "high"
        ],
        "temperature": 0.3,
        "max_tokens": 500 # Added max_tokens to control response length
    }
    url = f"{basicUrl}/deployments/{modelName}/chat/completions/?api-version={apiVersion}"
    headers = {'Content-Type': 'application/json', 'api-key': apiKey}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30) # Added timeout
        response.raise_for_status() 
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0 and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
            return data["choices"][0]["message"]["content"]
        else:
            return f"Error: Unexpected AI response format: {data}"
    except requests.exceptions.Timeout:
        return "Error: AI service request timed out."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to AI service: {str(e)}"
    except (KeyError, IndexError, json.JSONDecodeError) as e: # Added JSONDecodeError
        return f"Error parsing AI response: {str(e)}"

def save_ai_analysis(ai_response, timestamp_str, file_timestamp_str):
    try:
        os.makedirs(AI_ANALYSIS_DIR, exist_ok=True)
        analysis_filename = f"ai_analysis_{file_timestamp_str}.txt"
        analysis_filepath = os.path.join(AI_ANALYSIS_DIR, analysis_filename)
        with open(analysis_filepath, 'w', encoding='utf-8') as f:
            f.write(f"AI Analysis at {timestamp_str}\n\n{ai_response}")
        return analysis_filename
    except Exception as e:
        print(f"Error saving AI analysis: {e}")
        return f"Error saving AI analysis: {str(e)}"
    


def process_and_analyze(image_rgb, low_light, anonymize, privacy_method, overlay_type="Smile", log_history=""):
    if image_rgb is None:
        return None, log_history + "\n[Error] Please provide an image or use the webcam.", log_history, "Please provide an image first.", None

    try:
        # Create directories if they don't exist
        for dir_path in [RECORDS_BASE_DIR, LOGS_DIR, IMAGES_DIR, AI_ANALYSIS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        # Log this error and continue, as some functionality might still work
        log_history += f"\n[Error creating directories: {e}]"


    # The input image_rgb from Gradio (type="numpy") is already an RGB NumPy array.
    original_image_for_ai = image_rgb.copy() # Use a clean copy for AI
    processed_image_display = image_rgb.copy()

    if low_light:
        processed_image_display = enhance_low_light(processed_image_display)

    # image_for_detection = processed_image_display.copy()
    # current_height, current_width = image_for_detection.shape[:2]
    # new_width = int(current_width * 1.5) # Example: upscale by 50%
    # new_height = int(current_height * 1.5)
    # if new_width > 0 and new_height > 0:
    #    image_for_detection = cv2.resize(image_for_detection, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


    # Face detection works best on RGB images
    face_locations = detect_faces(processed_image_display)
    count = len(face_locations)

    # Draw rectangles on the image that will be displayed/anonymized
    img_with_boxes = processed_image_display.copy()
    for (x, y, w, h) in face_locations:
        # OpenCV uses BGR for colors, but our image is RGB. So, use RGB tuple for color.
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0,255,0), 2) # Green in RGB

    if anonymize:
        # Anonymize the image that already has boxes (or a copy of processed_image_display before boxes)
        # If anonymization covers boxes, draw boxes *after* or ensure anonymization doesn't hide them.
        # Current anonymize_faces takes the image and locations, applies anonymization.
        # Let's apply anonymization to the image with boxes, so boxes might be part of anonymized area.
        # Or, better, anonymize first, then draw boxes on top if method allows.
        # For simplicity, let's use img_with_boxes, meaning boxes might get pixelated/blurred too.
        # If you want boxes on top of anonymized faces, pass `processed_image_display` to `anonymize_faces`,
        # then draw boxes on the result.
        # However, the current `anonymize_faces` draws on a copy.
        
        # Let's anonymize the `processed_image_display` and then draw boxes on the anonymized version.
        anonymized_base = anonymize_faces(processed_image_display, face_locations, method=privacy_method, overlay_type=overlay_type)
        # Now draw boxes on the anonymized_base
        processed_image_display = anonymized_base.copy() # Start with the anonymized version
        for (x, y, w, h) in face_locations:
             # If method is Black Box or Emoji, drawing a box over it might be redundant or look odd.
             # For Pixelate/Blur, it can still be useful.
            if privacy_method not in ["Black Box", "Emoji Overlay"]: # Don't draw boxes over these
                cv2.rectangle(processed_image_display, (x, y), (x+w, y+h), (0,255,0), 2) # Green in RGB
    else:
        processed_image_display = img_with_boxes # Use image with boxes if no anonymization

    # Add student count text. OpenCV uses BGR for color, image is RGB. Use RGB tuple.
    cv2.putText(processed_image_display, f"Students (Detected Faces): {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2) # Red in RGB

    current_time = datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")

    image_filename = f"attendance_{file_timestamp_str}.jpg"
    image_filepath = os.path.join(IMAGES_DIR, image_filename)
    updated_log = log_history
    log_entry_for_file = ""

    try:
        # Convert RGB to BGR for OpenCV imwrite if needed, but PIL can save RGB directly
        img_to_save_pil = Image.fromarray(processed_image_display.astype(np.uint8))
        img_to_save_pil.save(image_filepath)
        log_entry_for_file = f"[{timestamp_str}] Students detected: {count}. Image saved: {image_filename}\n"
        updated_log += f"\n[{timestamp_str}] Students detected: {count}. Record saved as {image_filename}."
    except Exception as e:
        print(f"Error saving processed image: {e}")
        updated_log += f"\n[{timestamp_str}] Students detected: {count}. Error saving image: {e}"

    ai_response_text = "AI Analysis skipped (API key not set)." if apiKey == 'YOUR_API_KEY' else submit_image_to_ai(original_image_for_ai) # Use original for AI
    analysis_saved_path = "" # Initialize variable
    if "Error" not in ai_response_text :
        analysis_filename = save_ai_analysis(ai_response_text, timestamp_str, file_timestamp_str)
        if isinstance(analysis_filename, str) and analysis_filename.startswith("ai_analysis_"):
            updated_log += f"\n[{timestamp_str}] AI Analysis completed and saved as: {analysis_filename}."
            log_entry_for_file += f"[{timestamp_str}] AI Analysis saved: {analysis_filename}\n"
            analysis_saved_path = os.path.join(AI_ANALYSIS_DIR, analysis_filename) # For potential future use
        else: # This means save_ai_analysis returned an error message
            updated_log += f"\n[{timestamp_str}] AI Analysis completed but failed to save: {analysis_filename}"
            log_entry_for_file += f"[{timestamp_str}] AI Analysis completed but error saving: {analysis_filename}\n"
    else:
        updated_log += f"\n[{timestamp_str}] AI Analysis issue: {ai_response_text}"
        log_entry_for_file += f"[{timestamp_str}] AI Analysis issue: {ai_response_text}\n"


    try:
        daily_log_filename = f"attendance_log_{current_time.strftime('%Y-%m-%d')}.txt"
        daily_log_filepath = os.path.join(LOGS_DIR, daily_log_filename)
        with open(daily_log_filepath, 'a', encoding='utf-8') as f:
            f.write(log_entry_for_file) 
    except Exception as e:
        print(f"Error writing to daily log: {e}")
        updated_log += f"\n[Error] Failed to write to daily log: {e}"

    # The image for the AI tab (chat_img_upload) should be the one that was processed for display
    # or the original one if users prefer that for AI analysis.
    # Let's pass the processed_image_display to the AI tab's image component.
    return processed_image_display, updated_log, updated_log, ai_response_text, original_image_for_ai # Pass original to AI tab image

# Define your CSS string
custom_css = """
body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    background-color: #f8f9fa; /* Light grey background for the page */
    color: #212529; /* Darker base text color for readability */
}
h1 { /* Main title of the application */
    color: #0056b3 !important; /* A strong, professional blue (like HKBU blue) */
    text-align: center;
    margin-bottom: 30px !important;
    font-weight: 600; /* Slightly bolder than default */
    letter-spacing: -0.5px; /* Tighter letter spacing for a modern feel */
}
.gr-button {
    font-weight: 600 !important; /* Bolder button text */
    border-radius: 8px !important;
    padding: 12px 18px !important; /* Generous padding for easier clicking */
    transition: background-color 0.2s ease, transform 0.1s ease; /* Smooth transitions */
}
.gr-input, .gr-output, .gr-textbox textarea, .gr-dropdown select, .gr-checkboxgroup div { /* Styling for various input elements */
    border-radius: 8px !important;
    border: 1px solid #ced4da !important; /* Lighter border color */
    background-color: #fff !important;
    padding: 10px !important;
}
.gr-markdown h3 { /* Specific styling for H3 in Markdown for section titles */
    color: #0056b3; /* Blue for section titles */
    margin-top: 0; /* Remove default top margin if it's the first element in a group */
    margin-bottom: 15px;
    font-weight: 600;
    border-bottom: 1px solid #e0e0e0; /* Underline for section titles */
    padding-bottom: 8px;
}
#main_image_input div[data-testid="image"] > svg {
    display: none !important;
}
"""


# Gradio Interface with Tabs - UPDATED
with gr.Blocks(theme=gr.themes.Glass(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), css=custom_css) as demo:
    gr.Markdown("# 🏫 Smart Classroom: AI Attendance & Insights") # Slightly updated title

    log_history_state = gr.State(value="System Initialized. Ready for image input...")

    with gr.Tabs():
        with gr.TabItem("Attendance & Face Detection"): # Tab Title
            with gr.Row(equal_height=False):
                with gr.Column(scale=2): 
                    with gr.Group():
                        gr.Markdown("### 📷 Image Input & Options")
                        img_upload = gr.Image(label="Classroom Image (Webcam or Upload)", type="numpy", sources=["webcam", "upload"], height=400, elem_id="main_image_input")
                        with gr.Accordion("⚙️ Processing Settings", open=False): # Using an icon in title
                            low_light = gr.Checkbox(label="Enhance Low-Light Conditions", value=False)
                            anonymize = gr.Checkbox(label="Enable Face Anonymization (Privacy)", value=True)
                            # Put privacy method and overlay type in a row to save vertical space if they fit
                            with gr.Row():
                                privacy_method = gr.Dropdown(label="Anonymization Method",
                                                            choices=["Pixelate", "Blur", "Black Box", "Emoji Overlay", "Mosaic"],
                                                            value="Pixelate", scale=2) # Give more space to method
                                overlay_type = gr.Dropdown(label="Emoji Type",
                                                          choices=["Smile", "Cat"],
                                                          value="Smile",
                                                          visible=False, scale=1) # Less space for type
                        detect_btn = gr.Button("📊 Process & Analyze Classroom", variant="primary", elem_id="detect_button") # Icon in button

                with gr.Column(scale=3): 
                    with gr.Group():
                        gr.Markdown("### 🖼️ Processed Image with Detections") # Icon in title
                        out_img = gr.Image(label="Detection & Anonymization Result", interactive=False, type="numpy", height=420) # Slightly adjusted height for balance
                    with gr.Group():
                        gr.Markdown("### 📝 System & Attendance Log") # Icon in title
                        log = gr.Textbox(label="Activity Log", interactive=False, lines=8, max_lines=20) # Increased lines slightly

        with gr.TabItem("🤖 AI Classroom Insights"): # Tab Title with icon
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                     with gr.Group():
                        gr.Markdown("### 💡 AI Analysis Input") # Icon in title
                        chat_img_upload = gr.Image(label="Image for AI (from Detection or New Upload)", type="numpy", interactive=False, height=300)
                        gr.Markdown("---") 
                        gr.Markdown("#### OR Upload New Image for AI:")
                        manual_ai_img_upload = gr.Image(label="Upload Fresh Image for Analysis", type="numpy", sources=["upload"], height=300)
                        user_input = gr.Textbox(label="Ask a Specific Question (Optional)", placeholder="e.g., Are students engaged? How many are looking at the front?")
                        chat_btn = gr.Button("🧠 Get AI Insights", variant="secondary") # Icon in button

                with gr.Column(scale=3):
                    with gr.Group():
                        gr.Markdown("### 💬 AI Generated Report") # Icon in title
                        ai_response_output = gr.Textbox(label="Classroom Analysis Report", interactive=False, lines=22, max_lines=35) # Increased lines


    # --- Event Handlers ---
    def update_overlay_visibility(method):
        return gr.update(visible=(method == "Emoji Overlay"))
    privacy_method.change(update_overlay_visibility, inputs=privacy_method, outputs=overlay_type)

    detect_btn.click(
        fn=process_and_analyze,
        inputs=[img_upload, low_light, anonymize, privacy_method, overlay_type, log_history_state],
        outputs=[out_img, log_history_state, log, ai_response_output, chat_img_upload]
    )

    # Function to handle AI analysis for the AI tab
    # It decides whether to use the image from the first tab or a newly uploaded one
    def run_ai_chat_analysis(image_from_tab1, newly_uploaded_image, question):
        image_to_analyze = None
        if newly_uploaded_image is not None:
            image_to_analyze = newly_uploaded_image
            print("Using newly uploaded image for AI analysis.")
        elif image_from_tab1 is not None:
            image_to_analyze = image_from_tab1
            print("Using image from detection tab for AI analysis.")
        else:
            return "Please process an image on the 'Face Detection' tab first, or upload a new image here."

        if apiKey == 'YOUR_API_KEY':
             return "Error: API Key not configured. Please set your API key in the script."
        return submit_image_to_ai(image_to_analyze, question if question else "What are the people doing in this image?")

    chat_btn.click(
        fn=run_ai_chat_analysis,
        inputs=[chat_img_upload, manual_ai_img_upload, user_input],
        outputs=ai_response_output
    )

# Create dummy emoji files if they don't exist, for testing
# In a real scenario, these should be proper images.
if not os.path.exists('smile.png'):
    try:
        from PIL import Image, ImageDraw
        img_smile = Image.new('RGBA', (100, 100), (0,0,0,0))
        draw = ImageDraw.Draw(img_smile)
        draw.ellipse((10,10,90,90), fill=(255,255,0,255)) # Yellow circle
        draw.ellipse((25,30,45,50), fill=(0,0,0,255)) # Eye
        draw.ellipse((55,30,75,50), fill=(0,0,0,255)) # Eye
        draw.arc((25,40,75,80), 15, 165, fill=(0,0,0,255), width=5) # Smile
        img_smile.save('smile.png')
        print("Created dummy smile.png")
    except ImportError:
        print("Pillow not installed, cannot create dummy smile.png. Please create it manually.")

if not os.path.exists('cat.png'):
    try:
        from PIL import Image, ImageDraw
        img_cat = Image.new('RGBA', (100, 100), (0,0,0,0))
        draw = ImageDraw.Draw(img_cat)
        draw.polygon([(50,10),(10,40),(40,50)], fill=(200,200,200,255)) # Ear1
        draw.polygon([(50,10),(90,40),(60,50)], fill=(200,200,200,255)) # Ear2
        draw.ellipse((20,30,80,90), fill=(128,128,128,255)) # Grey face
        draw.ellipse((30,45,45,60), fill=(0,255,0,255)) # Eye
        draw.ellipse((55,45,70,60), fill=(0,255,0,255)) # Eye
        img_cat.save('cat.png')
        print("Created dummy cat.png")
    except ImportError:
        print("Pillow not installed, cannot create dummy cat.png. Please create it manually.")


demo.launch(debug=True, share=False) # share=False for local testing, set to True if you need to share