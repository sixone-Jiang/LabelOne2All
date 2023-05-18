# based on transport
import paramiko
import yaml
import os
import stat
def ssh_connect(host, port, user, password):
    try:
        # 建立连接
        trans = paramiko.Transport((host, port))
        trans.connect(username=user, password=password)

        # 将sshclient的对象的transport指定为以上的trans
        ssh = paramiko.SSHClient()
        ssh._transport = trans

        # 剩下的就和上面一样了
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return ssh

    except Exception as e:
        print("Connect Error: ", e)
        return None

def ssh_connect_key(host, port, user, key_path):

    # 指定本地的RSA私钥文件
    # 如果建立密钥对时设置的有密码，password为设定的密码，如无不用指定password参数
    pkey = paramiko.RSAKey.from_private_key_file(key_path)

    # 建立连接
    trans = paramiko.Transport((host, port))
    trans.connect(username=user, pkey=pkey)

    # 将sshclient的对象的transport指定为以上的trans
    ssh = paramiko.SSHClient()
    ssh._transport = trans
    return ssh

def ssh_close(ssh):
    if ssh:
        ssh.close()


def ssh_exec_cmd(ssh, cmd):
    if ssh:
        stdin, stdout, stderr = ssh.exec_command(cmd) # 12 h screen -R Alice; cd /data/DiffusionDet ; python main.py; nohup
        # n-teminal
        out = stdout.read().decode()
        return out
    else:
        return None

def sftp_get(ssh, remotefile, localfile):

    sftp = paramiko.SFTPClient.from_transport(ssh._transport)
    sftp.get(remotefile, localfile)

# 从本地上传文件到远程服务器,如果文件已经存在，则覆盖
def sftp_put(ssh, localfile, remotefile):

    sftp = paramiko.SFTPClient.from_transport(ssh._transport)
    sftp.put(localfile, remotefile)

# 从本地上传文件夹到远程服务器,如果文件夹已经存在，则覆盖
def sftp_put_dir(ssh, localdir, remotedir):
    sftp = paramiko.SFTPClient.from_transport(ssh._transport)
    for root, dirs, files in os.walk(localdir):
        for file in files:
            local_file = os.path.join(root, file)
            a = local_file.replace(localdir, '').replace('\\', '/').lstrip('/')
            remote_file = os.path.join(remotedir, a)
            sftp.put(local_file, remote_file)

def sftp_get_dir(ssh, remotedir, localdir):
    sftp = paramiko.SFTPClient.from_transport(ssh._transport)
    if not os.path.exists(localdir):
        os.makedirs(localdir)
    for f in sftp.listdir_attr(remotedir):
        if f.filename not in ['.', '..']:
            if stat.S_ISDIR(f.st_mode):
                sftp_get_dir(ssh, os.path.join(remotedir, f.filename), os.path.join(localdir, f.filename))
            else:
                sftp.get(os.path.join(remotedir, f.filename), os.path.join(localdir, f.filename))

txt = '''
cdAlice=$"cd /data/DiffusionDet"
runAlice=$"CUDA_VISIBLE_DEVICES=0,1,2,3 python train_net.py --num-gpus 4 --config-file configs/alice.diffudet.coco.res101.yaml MODEL.WEIGHTS /data/Alice/DiffusionDet/models/model_0029999.pth"
screen -dmS alice3;
screen -x -S alice3 -p 0 -X stuff "$cdAlice^M";
screen -x -S alice3 -p 0 -X stuff "$runAlice^M";
'''

def make_yaml(iter_base, num_classes=1, path='config/alicetest.diffdet.yaml'):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    data['SOLVER']['MAX_ITER'] = 10*iter_base
    data['MODEL']['DiffusionDet']['NUM_CLASSES'] = num_classes

    with open(path, 'w') as f:
        yaml.dump(data, f)

def make_inference_sh(ssh, work_path, gpu_id, config_file, screen_name, threhold=0.3, input_folder='datasets/myVocData/JPEGImages/*.jpg'):
    my_txt = '''
cdAlice=$\"cd {0} && conda activate alice \"
runAlice=$\"CUDA_VISIBLE_DEVICES={1} python inference.py --config-file configs/{3} --confidence-threshold {2} --input {5} --opts MODEL.WEIGHTS {0}output/model_final.pth MODEL.DiffusionDet.NUM_PROPOSALS 2000\"
screen -dmS {4};
screen -x -S {4} -p 0 -X stuff \"$cdAlice^M\";
screen -x -S {4} -p 0 -X stuff \"$runAlice^M\";
'''.format(work_path, gpu_id, threhold, config_file, screen_name, input_folder)

    with open('config/run_shell/' + 'inference.sh', 'w') as f:
        f.write(my_txt)

    #yaml_file_name = 'alicetest.diffudet.coco.res101.yaml'
    sftp_put(ssh, 'config/run_shell/' + 'inference.sh', work_path + '/inference.sh')
    #sftp_put(ssh, 'config/' + yaml_file_name, work_path + '/configs/' + yaml_file_name)
    cmd1 = 'bash ' + work_path + '/inference.sh'
    ssh_exec_cmd(ssh, cmd1)

def make_run_sh(ssh, work_path, gpu_id, num_gpus, config_file, screen_name, pretrain_model='model_test.pth'):
    my_txt = '''
cdAlice=$\"cd {0} && conda activate alice \"
runAlice=$\"CUDA_VISIBLE_DEVICES={1} python train_net.py --num-gpus {2} --config-file configs/{3} MODEL.WEIGHTS {0}models/{5}\"
screen -dmS {4};
screen -x -S {4} -p 0 -X stuff \"$cdAlice^M\";
screen -x -S {4} -p 0 -X stuff \"$runAlice^M\";
'''.format(work_path, gpu_id, num_gpus, config_file, screen_name, pretrain_model)
    #cmd = 'echo -e ' +' \" ' + my_txt + ' \" ' + '> ' +work_path + '/ft.sh'
    #ssh_exec_cmd(ssh, cmd)
    with open('config/run_shell/' + 'ft.sh', 'w') as f:
        f.write(my_txt)

    yaml_file_name = config_file
    sftp_put(ssh, 'config/run_shell/' + 'ft.sh', work_path + '/ft.sh')
    sftp_put(ssh, 'config/' + yaml_file_name, work_path + '/configs/' + yaml_file_name)
    cmd1 = 'bash ' + work_path + '/ft.sh'
    ssh_exec_cmd(ssh, cmd1)

def update_annotations_and_ImageSets_Main(ssh, annotations_path, ImageSets_Main_path, remote_work_path='', remote_annotations_path='/datasets/myVocData/Annotations/', remote_ImageSets_Main_path='/datasets/myVocData/ImageSets/Main/'):
    sftp_put_dir(ssh, annotations_path, remote_work_path + remote_annotations_path)
    sftp_put_dir(ssh, ImageSets_Main_path, remote_work_path + remote_ImageSets_Main_path)

def get_inference_xml(ssh, remote_work_path, local_path, remote_inference_annotations='/datasets/myVocData/new_annotations_xml/',local_inference_annotations='/new_annotations_xml/'):
    sftp_get_dir(ssh, remote_work_path + remote_inference_annotations, local_path+local_inference_annotations)

def get_log_info(ssh, work_path, base_path='/output/log.txt', tail_num=10):
    cmd = 'tail -n {} '.format(tail_num) + work_path + base_path
    #print(cmd)
    out = ssh_exec_cmd(ssh, cmd)
    return out

def clear_output_inference_files(ssh, work_path, base_path='inference/*'):
    cmd = 'rm -rf ' + work_path + '/output/' +base_path
    print(cmd)
    ssh_exec_cmd(ssh, cmd)

def kill_screen(ssh, screen_id):
    cmd = 'screen -S ' + str(screen_id) + ' -X quit'
    ssh_exec_cmd(ssh, cmd)

def find_screen(ssh, screen_name):
    cmd = 'screen -ls | grep ' + screen_name
    out = ssh_exec_cmd(ssh, cmd)
    return out