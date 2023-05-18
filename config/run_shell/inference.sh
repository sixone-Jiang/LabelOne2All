
cdAlice=$"cd /data/DiffusionDet/ && conda activate test "
runAlice=$"CUDA_VISIBLE_DEVICES=0,1,2,3 python inference.py --config-file configs/alicetest.diffdet.yaml --confidence-threshold 0.4 --input datasets/myVocData/JPEGImages/*.jpg --opts MODEL.WEIGHTS /data/DiffusionDet/output/model_final.pth MODEL.DiffusionDet.NUM_PROPOSALS 2000"
screen -dmS alice_test;
screen -x -S alice_test -p 0 -X stuff "$cdAlice^M";
screen -x -S alice_test -p 0 -X stuff "$runAlice^M";
