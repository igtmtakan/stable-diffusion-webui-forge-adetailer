# ADetailer Extras Installation Guide

Complete installation guide for ADetailer integration in Stable Diffusion WebUI Forge Extras tab.

## 📋 Prerequisites

- **Stable Diffusion WebUI Forge** (latest version)
- **ADetailer Extension** installed and working
- **Python 3.10+** with required dependencies
- **GPU**: NVIDIA CUDA or AMD ROCm support

## 🚀 Quick Installation

### Method 1: Complete Repository Clone (Recommended)

```bash
# Clone the complete repository
git clone https://github.com/igtmtakan/stable-diffusion-webui-forge-adetailer.git

# Copy to your WebUI Forge directory
# Replace [YOUR_WEBUI_PATH] with your actual path
cp -r stable-diffusion-webui-forge-adetailer/* [YOUR_WEBUI_PATH]/
```

### Method 2: Manual File Installation

Download and place the following files in your WebUI Forge directory:

#### Required Files for Extras Tab Integration:

1. **Core Processing Class**
   ```
   extensions/adetailer/scripts/adetailer_extras.py
   ```
   - Main ADetailer processing class
   - Face detection and inpainting logic
   - Model management

2. **WebUI Integration Modules**
   ```
   modules/extras.py
   modules/postprocessing.py  
   modules/ui_postprocessing.py
   ```
   - Extras tab integration
   - Batch processing support
   - UI components

3. **ADetailer Extension** (if not installed)
   ```
   extensions/adetailer/
   ```
   - Complete ADetailer extension
   - Required for face detection models

## 📁 File Structure

After installation, your directory should look like:

```
your-webui-forge/
├── extensions/
│   └── adetailer/
│       ├── scripts/
│       │   ├── !adetailer.py
│       │   └── adetailer_extras.py    # ← New file
│       ├── adetailer/
│       │   ├── common.py
│       │   ├── inpaint_models.py
│       │   └── ...
│       └── ...
├── modules/
│   ├── extras.py                      # ← Modified
│   ├── postprocessing.py              # ← Modified
│   ├── ui_postprocessing.py           # ← Modified
│   └── ...
└── README_ADETAILER.md                # ← Documentation
```

## ⚙️ Configuration

### 1. Install ADetailer Extension (if not already installed)

```bash
# In your WebUI Forge directory
cd extensions
git clone https://github.com/Bing-su/adetailer.git
```

### 2. Download Required Models

The integration supports these model types:

#### Inpaint Models (17 available):
- **Region-specific**: `indianBeauty_v15_inpainting`, `koreanDollLikeness_v15_inpainting`
- **Universal**: `sd-v1-5-inpainting`, `chilloutmix_inpainting`

#### Detection Models (11 available):
- **YOLOv8**: `face_yolov8n.pt`, `face_yolov8s.pt`
- **MediaPipe**: `mediapipe_face_full`, `mediapipe_face_mesh`

### 3. Verify Installation

1. **Start WebUI Forge**
   ```bash
   python webui.py
   ```

2. **Check Extras Tab**
   - Navigate to **Extras** tab
   - Look for **ADetailer** accordion section
   - Verify dropdown menus show available models

3. **Test Functionality**
   - Upload a portrait image
   - Enable ADetailer
   - Select models and click Generate

## 🔧 Troubleshooting

### Common Issues:

#### 1. "ADetailer not available" Error
```
Solution: Install ADetailer extension
cd extensions && git clone https://github.com/Bing-su/adetailer.git
```

#### 2. No Models in Dropdown
```
Solution: Download ADetailer models
- Check models/adetailer/ directory
- Download from ADetailer releases
```

#### 3. Import Errors
```
Solution: Check Python dependencies
pip install -r requirements.txt
```

#### 4. GPU Memory Issues
```
Solution: Adjust settings
- Lower detection confidence
- Use smaller models
- Enable model offloading
```

### Debug Mode:

Enable detailed logging by checking console output:
```
[ADetailer Extras] ADetailer modules imported successfully
[ADetailer Extras] Available inpaint models: [...]
[ADetailer Extras] Available detection models: [...]
```

## 🎯 Usage Guide

### Basic Usage:

1. **Go to Extras Tab**
2. **Select Mode**: Single Image, Batch Process, or Batch from Directory
3. **Upload Image(s)**
4. **Configure ADetailer**:
   - ✅ Enable ADetailer
   - 🎨 Select Inpaint Model (region-specific recommended)
   - 🔍 Select Detection Model (or leave as "None" for RetinaFace)
   - ⚙️ Adjust Detection Confidence (0.3 default)
   - 🎭 Set Mask Blur (4 default)
5. **Click Generate**

### Advanced Features:

#### Region-Specific Processing:
- **Japanese/Asian**: `yayoiMix_v25`, `chineseDollLikeness_v15_inpainting`
- **Korean**: `koreanDollLikeness_v15_inpainting`
- **Indian**: `indianBeauty_v15_inpainting`
- **Western**: `americanBeauty_v15_inpainting`, `europeanBeauty_v20_inpainting`

#### Detection Methods:
- **None**: RetinaFace (CodeFormer equivalent) - High precision
- **YOLOv8**: Ultra-fast detection (80ms)
- **MediaPipe**: Specialized face mesh detection

#### Batch Processing:
- **Batch Process**: Upload multiple files
- **Batch from Directory**: Process entire folders
- **Automatic**: Same settings applied to all images

## 📊 Performance

### Expected Processing Times:
- **Face Detection**: 80ms (YOLOv8) to 200ms (RetinaFace)
- **Inpainting**: 2-3 seconds per face
- **Total**: 3-5 seconds per image (single face)

### Memory Usage:
- **VRAM**: 4-8GB recommended
- **RAM**: 8GB minimum
- **Storage**: 2GB for models

## 🔄 Updates

To update the integration:

```bash
# Pull latest changes
git pull origin main

# Restart WebUI Forge
```

## 🆘 Support

### Getting Help:

1. **Check Console Logs**: Look for `[ADetailer Extras]` messages
2. **Verify File Placement**: Ensure all files are in correct locations
3. **Test Dependencies**: Confirm ADetailer extension works in txt2img
4. **GPU Compatibility**: Check CUDA/ROCm installation

### Reporting Issues:

Include the following information:
- WebUI Forge version
- ADetailer extension version
- GPU type and VRAM
- Console error messages
- Steps to reproduce

## ✅ Verification Checklist

- [ ] WebUI Forge installed and working
- [ ] ADetailer extension installed
- [ ] Required files copied to correct locations
- [ ] Models downloaded and available
- [ ] Extras tab shows ADetailer section
- [ ] Dropdown menus populated with models
- [ ] Test image processing successful
- [ ] Console shows no errors

## 🎉 Success!

If all steps completed successfully, you now have:
- ✅ **17 region-specific inpaint models**
- ✅ **11 face detection models**
- ✅ **RetinaFace + YOLOv8 detection**
- ✅ **Complete Extras tab integration**
- ✅ **Batch processing support**
- ✅ **High-quality face enhancement**

Enjoy your enhanced ADetailer experience in Stable Diffusion WebUI Forge!
