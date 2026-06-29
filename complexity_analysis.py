import torch
import torch.nn as nn
from torchsummary import summary
import thop
import ptflops
from fvcore.nn import FlopCountAnalysis, parameter_count
import numpy as np
import time
from typing import Dict, Any
from models.segmodels_multitasking import CONVNEXTMODEL, SEGFORMER
# from segmentation_models_pytorch import DeepLabV3Plus
# from models.DPT.dpt import DPT


# dpt_model = DPT(encoder_size='base', nclass=4, features=128, out_channels=[96, 192, 384, 768], use_bn=False)
# model_dlv3p = DeepLabV3Plus(encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=4)
model_mit = SEGFORMER('MiT-B0', num_classes=4)
model_conv = CONVNEXTMODEL("ConvNeXt-T", num_classes=4)

model_mit.init_pretrained("./models/weights/mit_b0.pth")
model_conv.init_pretrained("./models/weights/convnext_tiny_1k_224_ema.pth")

def analyze_model_complexity(model: nn.Module, input_size: tuple = (3, 224, 224), 
                           device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Comprehensive complexity analysis for segmentation models
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (channels, height, width)
        device: Device to run analysis on
    """
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, *input_size).to(device)
    
    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    
    # 1. Number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Total Parameters (MB): {total_params * 4 / (1024 ** 2):.2f} MB")  # 4 bytes per float32
    
    # 2. Model summary using torchsummary
    print("\n" + "=" * 30)
    print("MODEL SUMMARY")
    print("=" * 30)
    try:
        summary(model, input_size, device=device)
    except:
        print("Torchsummary not available for this model architecture")
    
    # 3. FLOPS analysis using thop
    print("\n" + "=" * 30)
    print("FLOPS ANALYSIS (thop)")
    print("=" * 30)
    try:
        flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
        print(f"FLOPS: {flops:,}")
        print(f"GFLOPs: {gflops:.2f}")
        print(f"Parameters: {params:,}")
    except Exception as e:
        print(f"thop analysis failed: {e}")
    
    # 4. FLOPS analysis using ptflops
    print("\n" + "=" * 30)
    print("FLOPS ANALYSIS (ptflops)")
    print("=" * 30)
    try:
        macs, params = ptflops.get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False
        )
        flops = 2 * macs  # MACs to FLOPS conversion
        print(f"MACs: {macs:,}")
        print(f"FLOPS: {flops:,}")
        print(f"GFLOPs: {flops / 1e9:.2f}")
    except Exception as e:
        print(f"ptflops analysis failed: {e}")
    
    # 5. FLOPS analysis using fvcore
    print("\n" + "=" * 30)
    print("FLOPS ANALYSIS (fvcore)")
    print("=" * 30)
    try:
        flops_analyzer = FlopCountAnalysis(model, dummy_input)
        flops_fvcore = flops_analyzer.total()
        params_fvcore = parameter_count(model)['']
        
        print(f"FLOPS: {flops_fvcore:,}")
        print(f"GFLOPs: {flops_fvcore / 1e9:.2f}")
        print(f"Parameters: {params_fvcore:,}")
        
        # Print per-layer breakdown
        print("\nPer-layer FLOPs breakdown:")
        for module_name, module_flops in flops_analyzer.by_module().items():
            if module_flops > 0:
                print(f"  {module_name}: {module_flops:,}")
                
    except Exception as e:
        print(f"fvcore analysis failed: {e}")
    
    # 6. Memory usage estimation
    print("\n" + "=" * 30)
    print("MEMORY USAGE ESTIMATION")
    print("=" * 30)
    try:
        # Forward pass memory
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            _, _, output = model(dummy_input)
            
            peak_memory = torch.cuda.max_memory_allocated() if device == 'cuda' else 0
            current_memory = torch.cuda.memory_allocated() if device == 'cuda' else 0
            
        if device == 'cuda':
            print(f"Peak GPU Memory: {peak_memory / (1024 ** 2):.2f} MB")
            print(f"Current GPU Memory: {current_memory / (1024 ** 2):.2f} MB")
        
        # Model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024 ** 2)
        print(f"Model Size: {total_size:.2f} MB")
        
    except Exception as e:
        print(f"Memory analysis failed: {e}")
    
    # 7. Inference speed (FPS)
    print("\n" + "=" * 30)
    print("INFERENCE SPEED (FPS)")
    print("=" * 30)
    try:
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _, _, _ = model(dummy_input)
        
        # Benchmark
        num_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _, _, _ = model(dummy_input)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_runs
        fps = 1 / avg_time_per_batch
        
        print(f"Average inference time: {avg_time_per_batch * 1000:.2f} ms")
        print(f"FPS: {fps:.2f}")
        print(f"Total time for {num_runs} runs: {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Inference speed test failed: {e}")
    
    # 8. Output size information (important for segmentation)
    print("\n" + "=" * 30)
    print("OUTPUT ANALYSIS")
    print("=" * 30)
    try:
        with torch.no_grad():
            _, _, output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Output size: {output.element_size() * output.nelement() / (1024 ** 2):.2f} MB")
            
            # For segmentation models, typically output has shape [batch, classes, H, W]
            if len(output.shape) == 4:
                print(f"Segmentation map size: {output.shape[2]}x{output.shape[3]}")
                print(f"Number of classes: {output.shape[1]}")
                
    except Exception as e:
        print(f"Output analysis failed: {e}")
    
    return {
        'total_params': total_params,
        'flops': flops_fvcore if 'flops_fvcore' in locals() else 0,
        'gflops': flops_fvcore / 1e9 if 'flops_fvcore' in locals() else 0,
        'model_size_mb': total_size if 'total_size' in locals() else 0,
        'fps': fps if 'fps' in locals() else 0,
        'peak_memory_mb': peak_memory / (1024 ** 2) if 'peak_memory' in locals() else 0
    }

# Example usage function
def comp_analysis():
      
    # Run analysis
    results = analyze_model_complexity(model_mit, input_size=(3, 224, 224))
    
    return results

if __name__ == "__main__":
    # Run the example
    results = comp_analysis()
    
    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    for key, value in results.items():
        print(f"{key}: {value}")