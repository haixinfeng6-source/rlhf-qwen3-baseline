"""
模型评估脚本
用于评估训练后的 RLHF 模型
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from typing import List
import json


def load_model(model_path: str, base_model: str = None):
    """
    加载训练后的模型
    如果是 LoRA 模型，需要提供 base_model
    """
    print(f"Loading model from {model_path}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    # 尝试直接加载完整模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("Loaded full model")
    except:
        # 如果失败，尝试加载 LoRA 模型
        if base_model is None:
            raise ValueError("Need base_model for LoRA adapter")
        
        print(f"Loading LoRA adapter with base model {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()  # 合并 LoRA 权重
        print("Loaded and merged LoRA model")
    
    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """生成模型响应"""
    
    # 使用 chat template 格式化输入
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除输入部分，只保留生成的响应
    response = response[len(input_text):].strip()
    
    return response


def evaluate_on_samples(model, tokenizer, test_prompts: List[str]):
    """在测试样本上评估模型"""
    
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50 + "\n")
    
    results = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[Sample {i}/{len(test_prompts)}]")
        print(f"Prompt: {prompt}")
        print("-" * 50)
        
        response = generate_response(model, tokenizer, prompt)
        
        print(f"Response: {response}")
        print("="*50 + "\n")
        
        results.append({
            "prompt": prompt,
            "response": response,
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLHF trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model for LoRA (if applicable)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # 加载模型
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # 测试样本
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
    ]
    
    # 评估
    results = evaluate_on_samples(model, tokenizer, test_prompts)
    
    # 保存结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
