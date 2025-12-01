#!/bin/bash
max_rss=0
max_rss_case=""
max_vms=0
max_vms_case=""

start_core=16
for model_name in convnext_tiny convnext_small convnext_base convnext_large convnext_xlarge
do
  for core in 1
  do
    for bs in 1 2 4 8 16 32 64 128
    do
      echo "core $core, bs $bs"
      core_end=$((start_core + core - 1))
      nstreams=$core
      # nstreams=$((core / 2))
      logfile=/tmp/${model_name}_${core_end}_${bs}_${core}.log
      echo "numactl -C ${start_core}-${core_end} python main_ov1.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_22k_224.pth -n 21841 -b $bs -t $nstreams "
      numactl -C ${start_core}-${core_end} python main_ov1.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_22k_224.pth -n 21841 -b $bs -t $nstreams | grep GB > ${logfile}
      rss_val=$(grep "RSS" ${logfile} | awk '{print $2}')
      vms_val=$(grep "VMS" ${logfile} | awk '{print $2}')
      echo "RSS: $rss_val , VMS: $vms_val"

      # 更新最大值
      if (( $(echo "$rss_val > $max_rss" | bc -l) )); then
        max_rss=$rss_val
        max_rss_case="Model=$model, Core=$core, BS=$bs, RSS=$RSS"
      fi

      if (( $(echo "$vms_val > $max_vms" | bc -l) )); then
        max_vms=$vms_val
        max_vms_case="Model=$model, Core=$core, BS=$bs, VMS=$VMS"
      fi

    done
  done
done

for model_name in convnext_tiny convnext_small convnext_base convnext_large
do
  for core in 1
  do
    for bs in 1 2 4 8 16 32 64 128
    do
      echo "core $core, bs $bs"
      core_end=$((start_core + core - 1))
      nstreams=$core
      # nstreams=$((core / 2))
      logfile=/tmp/${model_name}_${core_end}_${bs}_${core}.log
      echo "numactl -C ${start_core}-${core_end} python main_ov1.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_1k_224_ema.pth -b $bs -t $nstreams "
      numactl -C ${start_core}-${core_end} python main_ov1.py -i ../1.jpg -s 224 -m ${model_name} -c ../ConvNeXt-models/${model_name}_1k_224_ema.pth -b $bs -t $nstreams | grep GB > ${logfile}
      rss_val=$(grep "RSS" ${logfile} | awk '{print $2}')
      vms_val=$(grep "VMS" ${logfile} | awk '{print $2}')
      echo "RSS: $rss_val , VMS: $vms_val"

      # 更新最大值
      if (( $(echo "$rss_val > $max_rss" | bc -l) )); then
        max_rss=$rss_val
        max_rss_case="Model=$model, Core=$core, BS=$bs, RSS=$RSS"
      fi

      if (( $(echo "$vms_val > $max_vms" | bc -l) )); then
        max_vms=$vms_val
        max_vms_case="Model=$model, Core=$core, BS=$bs, VMS=$VMS"
      fi
    done
  done
done


# 输出最大值结果
echo "======================================"
echo "最大 RSS: $max_rss_case"
echo "最大 VMS: $max_vms_case"
echo "======================================"

