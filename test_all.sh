for core in 1 2 32
do
  for bs in 1 2 4 8 16 32 64
  do
    echo "core $core, bs $bs"
    core_end=$((63 + core))
    echo "numactl -C 64-${core_end} python main_ov.py -i ../1.jpg -s 224 -m convnext_base -c ../ConvNeXt-models/convnext_base_1k_224_ema.pth -b $bs -t $core "
    numactl -C 64-${core_end} python main_ov.py -i ../1.jpg -s 224 -m convnext_base -c ../ConvNeXt-models/convnext_base_1k_224_ema.pth -b $bs -t $core | grep FPS > /tmp/${core_end}_${bs}_${core}.log
    torch_f32=$(grep "Torch_FP32" /tmp/${core_end}_${bs}_${core}.log | awk '{print $12}')
    echo "Torch_F32 FPS: $torch_f32"

    torch_bf16=$(grep "Torch_BF16" /tmp/${core_end}_${bs}_${core}.log | awk '{print $12}')
    ratio=$(awk "BEGIN {printf \"%.4f\", $torch_bf16 / $torch_f32}")
    echo "Torch_BF16 FPS: $torch_bf16, ${ratio}"

    ov_f32_max=$(grep "OV.*F32" /tmp/${core_end}_${bs}_${core}.log | awk '{print $11}' | sort -nr | head -1)
    ov_f32_type=$(grep "OV.*F32" /tmp/${core_end}_${bs}_${core}.log | awk '{print $1}' | sort -nr | head -1)
    ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f32_max / $torch_f32}")
    echo "OV F32 Max FPS: $ov_f32_max (Type: $ov_f32_type), ${ratio}"

    ov_bf16_max=$(grep "OV.*BF16" /tmp/${core_end}_${bs}_${core}.log | awk '{print $11}' | sort -nr | head -1)
    ov_bf16_type=$(grep "OV.*BF16" /tmp/${core_end}_${bs}_${core}.log | awk '{print $1}' | sort -nr | head -1)
    ratio=$(awk "BEGIN {printf \"%.4f\", $ov_bf16_max / $torch_f32}")
    echo "OV BF16 Max FPS: $ov_bf16_max (Type: $ov_bf16_type), ${ratio}"

    ov_f16_max=$(grep "OV.*_F16" /tmp/${core_end}_${bs}_${core}.log | awk '{print $11}' | sort -nr | head -1)
    ov_f16_type=$(grep "OV.*_F16" /tmp/${core_end}_${bs}_${core}.log | awk '{print $1}' | sort -nr | head -1)
    ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f16_max / $torch_f32}")
    echo "OV F16 Max FPS: $ov_f16_max (Type: $ov_f16_type), ${ratio}"
  done
done
