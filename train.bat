@echo off

SET ROOT_DIR=C:/Users/DSGIANg/Documents/NYP-SDAAI/PDC-2/ITI107-DL-Networks/L4-02Dec2021/practical-session-4/balloon_project
SET MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
SET EXPERIMENT=run1
SET CUDA_VISIBLE_DEVICES=0
SET PIPELINE_CONFIG_PATH=%ROOT_DIR%/models/%MODEL%/%EXPERIMENT%/pipeline.config
SET MODEL_DIR=%ROOT_DIR%/models/%MODEL%/%EXPERIMENT%/

python model_main_tf2.py ^
       --pipeline_config_path="%PIPELINE_CONFIG_PATH%" ^
	   --model_dir="%MODEL_DIR%" ^
	   --checkpoint_every_n=100 ^
	   --alsologtostderr