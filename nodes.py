import os
import ast
import json
import time
import requests
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

import folder_paths


def parse_json(json_output: str) -> str:
    """Extract the JSON payload from a model response string."""
    if "```json" in json_output:
        json_output = json_output.split("```json", 1)[1]
        json_output = json_output.split("```", 1)[0]

    try:
        parsed = json.loads(json_output)
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                json_output = inner
    except Exception:
        pass
    return json_output


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Return bounding boxes parsed from the model's raw JSON output."""
    text = parse_json(text)
    try:
        data = json.loads(text)
    except Exception:
        try:
            data = ast.literal_eval(text)
        except Exception:
            end_idx = text.rfind('"}') + len('"}')
            truncated = text[:end_idx] + "]"
            data = ast.literal_eval(truncated)
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    items: List[DetectedBox] = []
    x_scale = img_width / input_w
    y_scale = img_height / input_h

    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        label = item.get("label", "")
        score = float(item.get("score", 1.0))
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 * y_scale)
        abs_x1 = int(x1 * x_scale)
        abs_y2 = int(y2 * y_scale)
        abs_x2 = int(x2 * x_scale)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        if score >= score_threshold:
            items.append(DetectedBox([abs_x1, abs_y1, abs_x2, abs_y2], score, label))
    items.sort(key=lambda x: x.score, reverse=True)
    return [
        {"score": b.score, "bbox": b.bbox, "label": b.label}
        for b in items
    ]


@dataclass
class DetectedBox:
    bbox: List[int]
    score: float
    label: str = ""


@dataclass
class QwenModel:
    model: Any
    processor: Any
    device: str
    original_params: Dict[str, Any] = None


class DownloadAndLoadQwenModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-32B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                ], ),
                "device": ([
                    "auto",
                    "cuda:0",
                    "cuda:1",
                    "cpu",
                ], ),
                "precision": ([
                    "INT4",
                    "INT8",
                    "BF16",
                    "FP16",
                    "FP32",
                ], ),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], ),
                "offline_mode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "Qwen2.5-VL"

    def load(self, model_name: str, device: str, precision: str, attention: str, offline_mode: bool = False):
        # 保存原始参数，用于后续可能的重新加载
        original_params = {
            "model_name": model_name,
            "device": device,
            "precision": precision,
            "attention": attention,
            "offline_mode": offline_mode
        }
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        
        # 检查本地目录是否已存在
        model_exists_locally = os.path.exists(model_dir) and any(os.listdir(model_dir))
        
        # 离线模式或在线模式下载逻辑
        if not offline_mode and not model_exists_locally:
            # 添加重试逻辑来处理SSL错误
            max_retries = 3
            retry_delay = 5  # 秒
            success = False
            
            for attempt in range(max_retries):
                try:
                    # Always attempt download with resume enabled so an interrupted download
                    # can be continued when the node is executed again.
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    success = True
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"SSL/连接错误，将在{retry_delay}秒后重试 ({attempt+1}/{max_retries})...")
                        print(f"错误详情: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避策略
                    else:
                        print(f"下载模型失败，已达到最大重试次数。尝试使用本地缓存。错误: {e}")
                        # 尝试直接从本地加载，如果有缓存的话
                except Exception as e:
                    print(f"下载时发生其他错误: {e}")
                    traceback.print_exc()
                    if attempt < max_retries - 1:
                        print(f"将在{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print("已达到最大重试次数，尝试使用本地缓存。")
        elif offline_mode and model_exists_locally:
            print(f"离线模式: 使用本地模型，不进行在线检查 '{model_dir}'")
        elif offline_mode and not model_exists_locally:
            raise ValueError(f"离线模式: 本地模型不存在 '{model_dir}'。请先在线模式下载模型。")
        else:
            print(f"使用现有本地模型 '{model_dir}'")
                
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        precision = precision.upper()
        dtype_map = {
            "BF16": torch.bfloat16,
            "FP16": torch.float16,
            "FP32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision, torch.bfloat16)
        quant_config = None
        if precision == "INT4":
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        elif precision == "INT8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = attention
        if precision == "FP32" and attn_impl == "flash_attention_2":
            # FlashAttention doesn't support fp32. Fall back to SDPA.
            attn_impl = "sdpa"

        # 添加重试逻辑用于模型加载
        max_model_load_retries = 3
        model_retry_delay = 3
        
        for attempt in range(max_model_load_retries):
            try:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch_dtype,
                    quantization_config=quant_config,
                    device_map=device_map,
                    attn_implementation=attn_impl,
                    trust_remote_code=True,
                    use_cache=True,
                    local_files_only=offline_mode,  # 离线模式下仅使用本地文件
                )
                break
            except OSError:
                if attempt < max_model_load_retries - 1 and not offline_mode:
                    print(f"加载模型失败，尝试重新下载 ({attempt+1}/{max_model_load_retries})...")
                    try:
                        # Force a re-download and try again.
                        snapshot_download(
                            repo_id=model_name,
                            local_dir=model_dir,
                            local_dir_use_symlinks=False,
                            resume_download=True,
                            force_download=True,
                        )
                        time.sleep(model_retry_delay)
                        model_retry_delay *= 2
                    except Exception as e:
                        print(f"重新下载时发生错误: {e}")
                else:
                    if offline_mode:
                        print("离线模式: 本地模型文件可能损坏或不完整。")
                    else:
                        print("已达到最大重试次数，无法加载模型。")
                    raise
            except Exception as e:
                if "SSL" in str(e) and attempt < max_model_load_retries - 1 and not offline_mode:
                    print(f"SSL错误，将在{model_retry_delay}秒后重试...")
                    time.sleep(model_retry_delay)
                    model_retry_delay *= 2
                else:
                    # Surface any other errors (e.g. missing flash_attn)
                    print(f"加载模型时发生错误: {str(e)}")
                    raise
        
        # 加载处理器
        try:
            processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=offline_mode)
        except Exception as e:
            print(f"加载处理器时发生错误: {str(e)}")
            raise
            
        return (QwenModel(model=model, processor=processor, device=device, original_params=original_params),)


class QwenVLDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "image": ("IMAGE",),
                "target": ("STRING", {"default": "object"}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "score_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
                "unload_after_detection": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX")
    RETURN_NAMES = ("text", "bboxes")
    FUNCTION = "detect"
    CATEGORY = "Qwen2.5-VL"

    def detect(
        self,
        qwen_model: QwenModel,
        image,
        target: str,
        bbox_selection: str = "all",
        score_threshold: float = 0.0,
        merge_boxes: bool = False,
        unload_after_detection: bool = True,
    ):
        """Generate bounding boxes for ``target`` within ``image``."""
        model = qwen_model.model
        processor = qwen_model.processor
        
        # 检查模型是否已被卸载，如果是则重新加载
        if model is None or processor is None:
            print("模型已被卸载，正在重新加载... Model has been unloaded, reloading...")
            # 获取DownloadAndLoadQwenModel类的实例
            loader = DownloadAndLoadQwenModel()
            
            # 如果QwenModel对象中保存了原始参数，则使用这些参数重新加载
            if hasattr(qwen_model, 'original_params') and qwen_model.original_params:
                params = qwen_model.original_params
                model_name = params.get('model_name', "Qwen/Qwen2.5-VL-7B-Instruct")
                device = params.get('device', "auto")
                precision = params.get('precision', "BF16")
                attention = params.get('attention', "flash_attention_2")
                offline_mode = params.get('offline_mode', False)
            else:
                # 如果没有原始参数，则使用默认值
                model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
                device = qwen_model.device  # 使用原来的device设置
                precision = "BF16"  # 默认精度
                attention = "flash_attention_2"  # 默认注意力机制
                offline_mode = False  # 默认在线模式
            
            # 重新加载模型，添加重试逻辑
            max_retries = 3
            retry_delay = 5
            loaded = False
            
            for attempt in range(max_retries):
                try:
                    # 重新加载模型
                    reloaded_model = loader.load(model_name, device, precision, attention, offline_mode)[0]
                    
                    # 更新QwenModel对象
                    qwen_model.model = reloaded_model.model
                    qwen_model.processor = reloaded_model.processor
                    
                    # 更新局部变量
                    model = qwen_model.model
                    processor = qwen_model.processor
                    print("模型重新加载完成 Model reloaded successfully")
                    loaded = True
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"重新加载时发生SSL/连接错误，将在{retry_delay}秒后重试 ({attempt+1}/{max_retries})...")
                        print(f"错误详情: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise ValueError(f"重新加载模型失败，无法继续。错误: {e}")
                except Exception as e:
                    print(f"重新加载时发生错误: {e}")
                    traceback.print_exc()
                    if attempt < max_retries - 1:
                        print(f"将在{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise ValueError(f"重新加载模型失败，无法继续。错误: {e}")
            
            if not loaded:
                raise ValueError("模型重新加载失败，请尝试手动重新运行节点。")
        
        device = qwen_model.device
        if device == "auto":
            device = str(next(model.parameters()).device)
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(device.split(":")[1]))
            except Exception:
                pass

        prompt = f"Locate the {target} and output bbox in JSON"

        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14
        items = parse_boxes(
            output_text,
            image.width,
            image.height,
            input_w,
            input_h,
            score_threshold,
        )

        selection = bbox_selection.strip().lower()
        boxes = items
        if selection != "all" and selection:
            idxs = []
            for part in selection.replace(",", " ").split():
                try:
                    idxs.append(int(part))
                except Exception:
                    continue
            boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

        if merge_boxes and boxes:
            x1 = min(b["bbox"][0] for b in boxes)
            y1 = min(b["bbox"][1] for b in boxes)
            x2 = max(b["bbox"][2] for b in boxes)
            y2 = max(b["bbox"][3] for b in boxes)
            score = max(b["score"] for b in boxes)
            label = boxes[0].get("label", target)
            boxes = [{"bbox": [x1, y1, x2, y2], "score": score, "label": label}]

        json_boxes = [
            {"bbox_2d": b["bbox"], "label": b.get("label", target)} for b in boxes
        ]
        json_output = json.dumps(json_boxes, ensure_ascii=False)
        bboxes_only = [b["bbox"] for b in boxes]
        
        # 卸载模型，释放内存
        if unload_after_detection and device.startswith("cuda"):
            import gc
            # 确保在卸载前保留原始参数信息
            original_params = qwen_model.original_params
            del qwen_model.model
            del qwen_model.processor
            # 将 QwenModel 对象中的引用设为 None
            qwen_model.model = None
            qwen_model.processor = None
            # 确保原始参数信息保留
            qwen_model.original_params = original_params
            # 触发垃圾回收
            gc.collect()
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            print("模型已卸载，显存已释放 Model unloaded, GPU memory released")
            
        return (json_output, bboxes_only)


class BBoxesToSAM2:
    """Convert a list of bounding boxes to the format expected by SAM2 nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"bboxes": ("BBOX",)}}

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("sam2_bboxes",)
    FUNCTION = "convert"
    CATEGORY = "Qwen2.5-VL"

    def convert(self, bboxes):
        if not isinstance(bboxes, list):
            raise ValueError("bboxes must be a list")

        # If already batched, return as-is
        if bboxes and isinstance(bboxes[0], (list, tuple)) and bboxes[0] and isinstance(bboxes[0][0], (list, tuple)):
            return (bboxes,)

        return ([bboxes],)


class SortBBoxes:
    """Sort bounding boxes from left to right or from top to bottom."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "sort_method": (["left_to_right", "top_to_bottom", "right_to_left", "bottom_to_top"], {"default": "left_to_right"}),
            }
        }

    RETURN_TYPES = ("BBOX", "JSON")
    RETURN_NAMES = ("sorted_bboxes", "sorted_json")
    FUNCTION = "sort"
    CATEGORY = "Qwen2.5-VL"

    def sort(self, bboxes, sort_method):
        """Sort bounding boxes by the specified method."""
        if not isinstance(bboxes, list) or not bboxes:
            return (bboxes, json.dumps([]))
        
        # 如果是单个bbox，直接返回
        if not isinstance(bboxes[0], list):
            return (bboxes, json.dumps([{"bbox": bboxes}]))
        
        # 确定排序键
        sort_key = None
        reverse = False
        
        if sort_method == "left_to_right":
            # 按x1（左边界）从左到右排序
            sort_key = lambda bbox: bbox[0]
        elif sort_method == "right_to_left":
            # 按x1（左边界）从右到左排序
            sort_key = lambda bbox: bbox[0]
            reverse = True
        elif sort_method == "top_to_bottom":
            # 按y1（上边界）从上到下排序
            sort_key = lambda bbox: bbox[1]
        elif sort_method == "bottom_to_top":
            # 按y1（上边界）从下到上排序
            sort_key = lambda bbox: bbox[1]
            reverse = True
        else:
            # 默认左到右
            sort_key = lambda bbox: bbox[0]
            
        # 复制边界框列表并排序
        sorted_bboxes = sorted(bboxes, key=sort_key, reverse=reverse)
        
        # 创建包含索引信息的JSON，便于查看排序结果
        json_result = []
        for i, bbox in enumerate(sorted_bboxes):
            json_result.append({
                "index": i,
                "bbox": bbox,
                "position": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "center_x": (bbox[0] + bbox[2]) / 2,
                    "center_y": (bbox[1] + bbox[3]) / 2
                }
            })
            
        return (sorted_bboxes, json.dumps(json_result, ensure_ascii=False))


class LoadLocalQwenModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-32B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                ], ),
                "device": ([
                    "auto",
                    "cuda:0",
                    "cuda:1",
                    "cpu",
                ], ),
                "precision": ([
                    "INT4",
                    "INT8",
                    "BF16",
                    "FP16",
                    "FP32",
                ], ),
                "attention": ([
                    "flash_attention_2",
                    "sdpa",
                ], ),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "Qwen2.5-VL"

    def load(self, model_name: str, device: str, precision: str, attention: str):
        # 保存原始参数，用于后续可能的重新加载
        original_params = {
            "model_name": model_name,
            "device": device,
            "precision": precision,
            "attention": attention,
            "offline_mode": True  # 固定为离线模式
        }
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        
        # 检查本地目录是否已存在
        model_exists_locally = os.path.exists(model_dir) and any(os.listdir(model_dir))
        
        # 如果本地不存在模型，则自动下载
        if not model_exists_locally:
            print(f"本地模型不存在 '{model_dir}'，将自动下载...")
            max_retries = 3
            retry_delay = 5  # 秒
            success = False
            
            for attempt in range(max_retries):
                try:
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=model_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                    )
                    success = True
                    print(f"成功下载模型: {model_name}")
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    if attempt < max_retries - 1:
                        print(f"SSL/连接错误，将在{retry_delay}秒后重试 ({attempt+1}/{max_retries})...")
                        print(f"错误详情: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避策略
                    else:
                        raise ValueError(f"下载模型失败，已达到最大重试次数。错误: {e}")
                except Exception as e:
                    print(f"下载时发生其他错误: {e}")
                    traceback.print_exc()
                    if attempt < max_retries - 1:
                        print(f"将在{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise ValueError(f"下载模型失败: {e}")
            
            if not success:
                raise ValueError(f"无法下载模型: {model_name}")
        else:
            print(f"使用现有本地模型 '{model_dir}'")
            
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        precision = precision.upper()
        dtype_map = {
            "BF16": torch.bfloat16,
            "FP16": torch.float16,
            "FP32": torch.float32,
        }
        torch_dtype = dtype_map.get(precision, torch.bfloat16)
        quant_config = None
        if precision == "INT4":
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        elif precision == "INT8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = attention
        if precision == "FP32" and attn_impl == "flash_attention_2":
            # FlashAttention doesn't support fp32. Fall back to SDPA.
            attn_impl = "sdpa"

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                quantization_config=quant_config,
                device_map=device_map,
                attn_implementation=attn_impl,
                trust_remote_code=True,
                use_cache=True,
                local_files_only=model_exists_locally,  # 如果是新下载的，允许在线检查
            )
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
        
        try:
            processor = AutoProcessor.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=model_exists_locally,  # 如果是新下载的，允许在线检查
            )
        except Exception as e:
            del model
            torch.cuda.empty_cache()
            raise RuntimeError(f"加载处理器失败: {e}")

        print(f"成功加载模型: {model_name}")
        return (QwenModel(model, processor, device, original_params),)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQwenModel": DownloadAndLoadQwenModel,
    "QwenVLDetection": QwenVLDetection,
    "BBoxesToSAM2": BBoxesToSAM2,
    "SortBBoxes": SortBBoxes,
    "LoadLocalQwenModel": LoadLocalQwenModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQwenModel": "Download and Load Qwen2.5-VL Model",
    "QwenVLDetection": "Qwen2.5-VL Object Detection",
    "BBoxesToSAM2": "Prepare BBoxes for SAM2",
    "SortBBoxes": "Sort Bounding Boxes",
    "LoadLocalQwenModel": "Load Local Qwen2.5-VL Model",
}
