@echo off

SET ROOT_DIR=C:/Users/DSGIANg/Documents/NYP-SDAAI/PDC-2/ITI107-DL-Networks/L4-02Dec2021/practical-session-4/balloon_project

SET MODEL=ssd_mobilenet_v2_320x320_coco17_tpu-8
SET EXPERIMENT=run1
SET LOG_DIR=%ROOT_DIR%/models/%MODEL%/%EXPERIMENT%
tensorboard --bind_all --logdir=%LOG_DIR%