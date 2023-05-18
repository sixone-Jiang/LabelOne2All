
cdAlice=$"cd /data/DiffusionDet/ && conda activate test "
runAlice=$"CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 --config-file configs/alicetest.diffdet.yaml MODEL.WEIGHTS /data/DiffusionDet/models/test_voc_split1_pre_model.pth"
screen -dmS alice_test;
screen -x -S alice_test -p 0 -X stuff "$cdAlice^M";
screen -x -S alice_test -p 0 -X stuff "$runAlice^M";
