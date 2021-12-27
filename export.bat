@echo off

SET ROOT_DIR=C:/Users/DSGIANg/Documents/NYP-SDAAI/PDC-2/ITI107-DL-Networks/L4-02Dec2021/practical-session-4/balloon_project

SET MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
SET EXPERIMENT=run1
SET PIPELINE_CONFIG_PATH=%ROOT_DIR%/models/%MODEL%/%EXPERIMENT%/pipeline.config
SET MODEL_DIR=%ROOT_DIR%/models/%MODEL%/%EXPERIMENT%/
SET TRAIN_CHECKPOINT_DIR=%MODEL_DIR%
SET EXPORT_DIR=%ROOT_DIR%/exported_models/%MODEL%/%EXPERIMENT%/

python exporter_main_v2.py ^
       --input_type image_tensor ^
	   --pipeline_config_path %PIPELINE_CONFIG_PATH% ^
	   --trained_checkpoint_dir %TRAIN_CHECKPOINT_DIR% ^
	   --output_directory %EXPORT_DIR%