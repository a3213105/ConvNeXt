start_core=96
for model_name in convnext_tiny convnext_small convnext_base convnext_large convnext_xlarge
do
  for core in 2 #4 8
  do
    for bs in 1 2 4 8 16 32 64 128
    do
      echo "core $core, bs $bs"
      core_end=$((start_core + core - 1))
      #nstreams=$core
      nstreams=$((core / 2))
      logfile=/tmp/${model_name}_${core_end}_${bs}_${core}.log
      echo "numactl -C ${start_core}-${core_end} python main_ov.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_22k_224.pth -n 21841 -b $bs -t $nstreams "
      numactl -C ${start_core}-${core_end} python main_ov.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_22k_224.pth -n 21841 -b $bs -t $nstreams | grep FPS > ${logfile}
      torch_f32=$(grep "Torch_FP32" ${logfile} | awk '{print $12}')
      echo "Torch_F32 FPS: $torch_f32"

      torch_bf16=$(grep "Torch_BF16" ${logfile} | awk '{print $12}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $torch_bf16 / $torch_f32}")
      echo "Torch_BF16 FPS: $torch_bf16, ${ratio}"

      value=$(grep "OV.*F32" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
      ov_f32_max=$(echo ${value} | awk '{print $1}')
      ov_f32_type=$(echo ${value} | awk '{print $2}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f32_max / $torch_f32}")
      echo "OV_F32 Max_FPS: $ov_f32_max ${ratio} (Type: $ov_f32_type)"

      value=$(grep "OV.*BF16" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
      ov_bf16_max=$(echo ${value} | awk '{print $1}')
      ov_bf16_type=$(echo ${value} | awk '{print $2}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $ov_bf16_max / $torch_f32}")
      echo "OV_BF16 Max_FPS: $ov_bf16_max ${ratio} (Type: $ov_bf16_type)"

      value=$(grep "OV.*_F16" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
      ov_f16_max=$(echo ${value} | awk '{print $1}')
      ov_f16_type=$(echo ${value} | awk '{print $2}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f16_max / $torch_f32}")
      echo "OV_F16 Max_FPS: $ov_f16_max ${ratio} (Type: $ov_f16_type)"
    done
  done
done

for model_name in convnext_tiny convnext_small convnext_base convnext_large
do
  for core in 1 #2 4 8
  do
    for bs in 8 16 32 64 128 256
    do
      echo "core $core, bs $bs"
      core_end=$((start_core + core - 1))
      nstreams=$((core / 2))
      nstreams=$core
      logfile=/tmp/${model_name}_${core_end}_${bs}_${core}.log
      echo "numactl -C ${start_core}-${core_end} python main_ov.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_1k_224_ema.pth -b $bs -t $nstreams "
      numactl -C ${start_core}-${core_end} python main_ov.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_1k_224_ema.pth -b $bs -t $nstreams | grep FPS > ${logfile}
      torch_f32=$(grep "Torch_FP32" ${logfile} | awk '{print $12}')
      echo "Torch_F32 FPS: $torch_f32"

      torch_bf16=$(grep "Torch_BF16" ${logfile} | awk '{print $12}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $torch_bf16 / $torch_f32}")
      echo "Torch_BF16 FPS: $torch_bf16, ${ratio}"

      value=$(grep "OV.*F32" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
      ov_f32_max=$(echo ${value} | awk '{print $1}')
      ov_f32_type=$(echo ${value} | awk '{print $2}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f32_max / $torch_f32}")
      echo "OV_F32 Max_FPS: $ov_f32_max ${ratio} (Type: $ov_f32_type)"

      value=$(grep "OV.*BF16" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
      ov_bf16_max=$(echo ${value} | awk '{print $1}')
      ov_bf16_type=$(echo ${value} | awk '{print $2}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $ov_bf16_max / $torch_f32}")
      echo "OV_BF16 Max_FPS: $ov_bf16_max ${ratio} (Type: $ov_bf16_type)"

      value=$(grep "OV.*_F16" ${logfile} | awk '{print $11, $1}' | sort -nr | head -1)
      ov_f16_max=$(echo ${value} | awk '{print $1}')
      ov_f16_type=$(echo ${value} | awk '{print $2}')
      ratio=$(awk "BEGIN {printf \"%.4f\", $ov_f16_max / $torch_f32}")
      echo "OV_F16 Max_FPS: $ov_f16_max ${ratio} (Type: $ov_f16_type)"
    done
  done
done
