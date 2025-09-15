from PIL import Image, ImageDraw
import os
from typing import Optional, Tuple

# ADetailerの利用可能性をチェック
try:
    # より確実なチェック方法
    import adetailer
    try:
        from adetailer import __version__ as adetailer_version
        print(f"[ADetailer Extras] ADetailer version: {adetailer_version}")
    except ImportError:
        print("[ADetailer Extras] ADetailer version not available")
    
    # 必要なモジュールをインポート
    from adetailer.common import get_models
    from adetailer.ultralytics import ultralytics_predict
    from adetailer.mediapipe import mediapipe_predict
    from adetailer.mask import mask_preprocess
    from adetailer.inpaint_models import get_inpaint_models
    
    ADETAILER_AVAILABLE = True
    print("[ADetailer Extras] ADetailer modules imported successfully")
    
except ImportError as e:
    print(f"[ADetailer Extras] ADetailer not available: {e}")
    ADETAILER_AVAILABLE = False

# Face restoration utilsのインポート
try:
    from modules.face_restoration_utils import FaceRestoreHelper
    FACE_RESTORATION_AVAILABLE = True
    print("[ADetailer Extras] Face restoration utils available")
except ImportError as e:
    print(f"[ADetailer Extras] Face restoration utils not available: {e}")
    FACE_RESTORATION_AVAILABLE = False

class ADetailerExtras:
    """
    ADetailer機能をExtrasタブで使用するためのクラス
    """
    
    def __init__(self):
        self.face_helper = None
        
    def get_adetailer_models(self):
        """
        利用可能なADetailer inpaintモデルのリストを取得
        """
        if not ADETAILER_AVAILABLE:
            return ["None"]
        
        try:
            # inpaintモデルのみを取得
            inpaint_models = get_inpaint_models()
            
            # 地域特化モデルを優先的に表示（実在するもののみ）
            priority_models = [
                "🎨 chineseDollLikeness_v15_inpainting",
                "🎨 americanBeauty_v15_inpainting",
                "🎨 europeanBeauty_v20_inpainting",
                "🎨 yayoiMix_v25",
                "🎨 chilloutmix_inpainting",
                "🎨 sd-v1-5-inpainting"
            ]
            
            # 利用可能なモデルをフィルタリング
            available_models = ["None"]
            
            # 優先モデルを先に追加
            for model in priority_models:
                if model in inpaint_models:
                    available_models.append(model)
            
            # その他のモデルを追加
            for model in inpaint_models:
                if model not in available_models and model not in priority_models:
                    available_models.append(model)
            
            print(f"[ADetailer Extras] Available inpaint models: {available_models}")
            return available_models
            
        except Exception as e:
            print(f"[ADetailer Extras] Error getting inpaint models: {e}")
            return ["None"]
    
    def get_detection_models(self):
        """
        利用可能なADetailer顔検出モデルのリストを取得
        """
        if not ADETAILER_AVAILABLE:
            return ["None"]
        
        try:
            # 顔検出モデルのみを取得
            all_models = get_models()
            
            # 顔検出モデルをフィルタリング
            detection_models = ["None"]
            
            # 顔検出モデルの優先順位
            priority_detection = [
                "🎨 face_yolov8n.pt",
                "🎨 face_yolov8s.pt", 
                "🎨 mediapipe_face_full",
                "🎨 mediapipe_face_mesh",
                "🎨 mediapipe_face_mesh_eyes_only",
                "🎨 hand_yolov8n.pt",
                "🎨 person_yolov8n-seg.pt",
                "🎨 person_yolov8s-seg.pt"
            ]
            
            # 優先モデルを先に追加
            for model in priority_detection:
                if model in all_models:
                    detection_models.append(model)
            
            # その他の顔検出モデルを追加
            for model in all_models:
                if ("face" in model.lower() or "person" in model.lower() or "hand" in model.lower()) and model not in detection_models:
                    detection_models.append(model)
            
            print(f"[ADetailer Extras] Available detection models: {detection_models}")
            return detection_models
            
        except Exception as e:
            print(f"[ADetailer Extras] Error getting detection models: {e}")
            return ["None"]
    
    def detect_faces(self, image: Image.Image, confidence: float = 0.3, detection_model: str = "None") -> list:
        """顔検出を実行"""
        try:
            # Detection Modelが指定されている場合は実際のADetailer顔検出を試行
            if detection_model and detection_model != "None":
                try:
                    # ADetailerの顔検出を使用
                    faces = self._detect_faces_with_adetailer(image, detection_model, confidence)
                    if faces:
                        print(f"[ADetailer Extras] Detected {len(faces)} faces using {detection_model}")
                        return faces
                    else:
                        print(f"[ADetailer Extras] No faces detected with {detection_model}, trying RetinaFace")
                        
                except Exception as e:
                    print(f"[ADetailer Extras] Detection model {detection_model} failed: {e}, trying RetinaFace")
            
            # RetinaFace検出を試行（CodeFormerと同じ方式）
            try:
                faces = self._detect_faces_with_retinaface(image, confidence)
                if faces:
                    print(f"[ADetailer Extras] Detected {len(faces)} faces using RetinaFace")
                    return faces
                else:
                    print(f"[ADetailer Extras] No faces detected with RetinaFace, using fallback")
            except Exception as e:
                print(f"[ADetailer Extras] RetinaFace detection failed: {e}, using fallback")
            
            # フォールバック検出（最後の手段）
            width, height = image.size
            face_size = min(width, height) // 3
            x = (width - face_size) // 2
            y = (height - face_size) // 2

            # 顔のバウンディングボックス (x, y, width, height)
            face_bbox = (x, y, face_size, face_size)
            print(f"[ADetailer Extras] Using fallback detection (center region) at {face_bbox}")
            return [face_bbox]

        except Exception as e:
            print(f"[ADetailer Extras] Face detection error: {e}")
            return []
    
    def _detect_faces_with_adetailer(self, image: Image.Image, detection_model: str, confidence: float) -> list:
        """ADetailerを使用した顔検出"""
        if not ADETAILER_AVAILABLE:
            return []
        
        try:
            print(f"[ADetailer Extras] Attempting detection with model: {detection_model}")
            
            # YOLOv8モデルの場合
            if "yolo" in detection_model.lower():
                results = ultralytics_predict(detection_model, image, confidence)
                if results and len(results) > 0:
                    faces = []
                    for result in results:
                        # バウンディングボックスを取得 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = result[:4]
                        # (x, y, width, height) 形式に変換
                        faces.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    return faces
            
            # MediaPipeモデルの場合
            elif "mediapipe" in detection_model.lower():
                results = mediapipe_predict(detection_model, image, confidence)
                if results and len(results) > 0:
                    faces = []
                    for result in results:
                        # MediaPipeの結果をバウンディングボックスに変換
                        x1, y1, x2, y2 = result[:4]
                        faces.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
                    return faces
            
            return []
            
        except Exception as e:
            print(f"[ADetailer Extras] ADetailer detection failed: {e}")
            return []
    
    def _detect_faces_with_retinaface(self, image: Image.Image, confidence: float) -> list:
        """RetinaFaceを使用した顔検出（CodeFormerと同じ方式）"""
        if not FACE_RESTORATION_AVAILABLE:
            return []
        
        try:
            # FaceRestoreHelperを初期化（CodeFormerと同じ設定）
            if self.face_helper is None:
                self.face_helper = FaceRestoreHelper(
                    upscale_factor=1,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model='retinaface_resnet50',
                    save_ext='png',
                    use_parse=True,
                    device='cuda'
                )
            
            # 顔検出を実行
            self.face_helper.clean_all()
            self.face_helper.read_image(image)
            self.face_helper.get_face_landmarks_5(
                only_center_face=False, 
                resize=640, 
                eye_dist_threshold=5
            )
            
            # 検出された顔のバウンディングボックスを取得
            faces = []
            for face_info in self.face_helper.cropped_faces:
                # バウンディングボックス情報を取得
                bbox = face_info.get('bbox', None)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    faces.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
            
            return faces
            
        except Exception as e:
            print(f"[ADetailer Extras] RetinaFace detection failed: {e}")
            return []

    def create_face_mask(self, image: Image.Image, face_bbox: Tuple[int, int, int, int], mask_blur: int = 4) -> Image.Image:
        """顔のマスクを作成"""
        try:
            x, y, w, h = face_bbox

            # マスク画像を作成
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)

            # 楕円形のマスクを描画
            draw.ellipse([x, y, x + w, y + h], fill=255)

            # ブラーを適用
            if mask_blur > 0:
                from PIL import ImageFilter
                mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))

            return mask

        except Exception as e:
            print(f"[ADetailer Extras] Mask creation error: {e}")
            # フォールバック：全体マスク
            return Image.new('L', image.size, 255)

    def process_face_inpainting(self, image: Image.Image, mask: Image.Image, inpaint_model: str = "None") -> Optional[Image.Image]:
        """顔のinpainting処理を実行"""
        try:
            if inpaint_model == "None":
                print("[ADetailer Extras] Using current checkpoint for inpainting")
                # 現在のcheckpointを使用
                return self._inpaint_with_current_model(image, mask)
            else:
                print(f"[ADetailer Extras] Using inpaint model: {inpaint_model}")
                # 指定されたinpaintモデルを使用
                return self._inpaint_with_model(image, mask, inpaint_model)

        except Exception as e:
            print(f"[ADetailer Extras] Inpainting error: {e}")
            return None

    def _inpaint_with_current_model(self, image: Image.Image, mask: Image.Image) -> Optional[Image.Image]:
        """現在のcheckpointを使用したinpainting"""
        try:
            # WebUIのimg2imgパイプラインを使用
            from modules.processing import StableDiffusionProcessingImg2Img
            from modules import shared

            # 処理パラメータを設定
            p = StableDiffusionProcessingImg2Img(
                init_images=[image],
                mask=mask,
                mask_blur=4,
                inpainting_fill=1,  # original
                inpaint_full_res=False,
                inpaint_full_res_padding=0,
                inpainting_mask_invert=0,
                prompt="high quality face, detailed, sharp",
                negative_prompt="blurry, low quality, distorted",
                steps=20,
                sampler_name="DPM++ 2M",
                cfg_scale=7.0,
                width=image.width,
                height=image.height,
                denoising_strength=0.4
            )

            # 処理を実行
            from modules.processing import process_images
            processed = process_images(p)

            if processed and processed.images:
                return processed.images[0]

            return None

        except Exception as e:
            print(f"[ADetailer Extras] Current model inpainting failed: {e}")
            return None

    def _inpaint_with_model(self, image: Image.Image, mask: Image.Image, model_name: str) -> Optional[Image.Image]:
        """指定されたinpaintモデルを使用したinpainting"""
        try:
            # モデル切り替えとinpainting処理
            from modules.processing import StableDiffusionProcessingImg2Img
            from modules import shared, sd_models

            # 現在のモデルを保存
            current_model = shared.sd_model

            try:
                # inpaintモデルに切り替え
                model_path = self._find_model_path(model_name)
                if model_path:
                    sd_models.load_model(model_path)

                # inpainting処理を実行
                result = self._inpaint_with_current_model(image, mask)

                return result

            finally:
                # 元のモデルに戻す
                if current_model:
                    shared.sd_model = current_model

        except Exception as e:
            print(f"[ADetailer Extras] Model inpainting failed: {e}")
            return None

    def _find_model_path(self, model_name: str) -> Optional[str]:
        """モデル名からパスを検索"""
        try:
            from modules import sd_models

            # モデル名からファイル名を抽出
            clean_name = model_name.replace("🎨 ", "").strip()

            # 利用可能なモデルから検索
            for model_info in sd_models.checkpoints_list.values():
                if clean_name in model_info.filename or clean_name in model_info.title:
                    return model_info.filename

            return None

        except Exception as e:
            print(f"[ADetailer Extras] Model path search failed: {e}")
            return None

    def process_image(self, image: Image.Image, enable: bool = True, inpaint_model: str = "None",
                     detection_model: str = "None", prompt_enhancement: bool = False,
                     confidence: float = 0.3, mask_blur: int = 4) -> Tuple[Optional[Image.Image], str]:
        """
        画像のADetailer処理を実行

        Returns:
            Tuple[Optional[Image.Image], str]: (処理済み画像, メッセージ)
        """
        if not enable:
            return image, "ADetailer disabled"

        try:
            print(f"[ADetailer Extras] Processing image with inpaint_model={inpaint_model}, detection_model={detection_model}")

            # 顔検出を実行
            faces = self.detect_faces(image, confidence, detection_model)

            if not faces:
                return image, "No faces detected"

            print(f"[ADetailer Extras] Processing {len(faces)} face(s)")

            # 各顔を処理
            result_image = image.copy()
            processed_count = 0

            for i, face_bbox in enumerate(faces):
                try:
                    print(f"[ADetailer Extras] Processing face {i+1}/{len(faces)}")

                    # マスクを作成
                    mask = self.create_face_mask(result_image, face_bbox, mask_blur)

                    # inpainting処理
                    print(f"[ADetailer Extras] Running inpainting for face {i+1}...")
                    inpainted = self.process_face_inpainting(result_image, mask, inpaint_model)

                    if inpainted:
                        result_image = inpainted
                        processed_count += 1
                        print(f"[ADetailer Extras] Face {i+1} processed successfully")
                    else:
                        print(f"[ADetailer Extras] Face {i+1} processing failed")

                except Exception as e:
                    print(f"[ADetailer Extras] Error processing face {i+1}: {e}")
                    continue

            if processed_count > 0:
                message = f"Processed {processed_count} face(s)"
                print(f"[ADetailer Extras] {message}")
                return result_image, message
            else:
                return image, "No faces were successfully processed"

        except Exception as e:
            error_msg = f"ADetailer processing failed: {e}"
            print(f"[ADetailer Extras] {error_msg}")
            return image, error_msg
