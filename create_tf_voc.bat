@echo off

SET ROOT_DIR=C:/Users/DSGIANg/Documents/NYP-SDAAI/PDC-2/ITI107-DL-Networks/L4-02Dec2021/practical-session-4/balloon_project
SET DATA_DIR=%ROOT_DIR%/data
SET LABELMAP=%DATA_DIR%/label_map.pbtxt
SET OUTPUT_DIR=%ROOT_DIR%/data
SET TEST_RATIO=0.2

REM echo %DATA_DIR%
REM echo %LABELMAP%
REM echo %TEST_RATIO%
REM echo %OUTPUT_DIR%
REM cd "%OUTPUT_DIR%"

python create_tf_records_voc.py ^
       --data_dir="%DATA_DIR%" ^
	   --label_map="%LABELMAP%" ^
	   --test_ratio="%TEST_RATIO%" ^
	   --output_dir="%OUTPUT_DIR%"