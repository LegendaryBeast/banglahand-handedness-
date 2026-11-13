#!/bin/sh
# Detached runner (survives Claude Code session restarts). Chain 1: CSAFE features -> evaluation.
# Chain 2: CSAFE network embeddings -> line-free fine-tuned ResNet-18 -> its Table 1/2 summary.
cd "$(dirname "$0")/../.."
R=results/csafe_ijdar_2026-11-12; L=results/ijdar_dev_2026-11-12
( OMP_NUM_THREADS=1 python3 scripts/extract_csafe_features.py --stage hand --workers 6 > $R/extract_hand.log 2>&1 && echo hand_done > $R/STATUS_hand
  while [ ! -f $R/STATUS_deep ]; do sleep 60; done
  OMP_NUM_THREADS=1 python3 scripts/run_csafe_evaluation.py --jobs 7 > $R/csafe_evaluation.log 2>&1 && echo eval_done > $R/STATUS_eval ) &
( python3 scripts/extract_csafe_features.py --stage deep > $R/extract_deep.log 2>&1 && echo deep_done > $R/STATUS_deep
  python3 scripts/run_linefree_finetune.py > $L/linefree_finetune.log 2>&1 && echo ft_done > $L/linefree_finetune/STATUS_ft
  python3 scripts/summarise_finetuned_grey_cnn.py --run $L/linefree_finetune > $L/linefree_finetune_summary.log 2>&1 && echo sum_done > $L/linefree_finetune/STATUS_sum ) &
wait
