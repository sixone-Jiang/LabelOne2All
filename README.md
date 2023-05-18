# Alice’s LabelOne2All

本项目实现了一个辅助标注系统，再也不担心老师交给我的标注数据集的任务啦！

您仅需将一堆无标签数据中每个需要标注的类别各标注1~10个物体框，剩下的就都交给程序吧！

一次效果较差？那就基于上一次的结果筛选一下，作为新的训练集呀！



## 运行环境

* pyqt5
* opencv
* Pillow
* tqdm



## 已知未修复bug

* 部分按钮在缺少前置按钮操作时，会导致程序直接崩溃
* 首次按下”run inference“可能导致信息不显示，不过不影响程序运行，可以在服务器里screen去直接查看运行结果



## 运行标注程序

注意，在完全构建好本系统之前（后续章节还有需要准备的环境和配置），仅可使用“StartLabel”页面功能，

用于对VOC格式图像进行标注。

```sheel
python test.py
```

<img src="https://raw.githubusercontent.com/sixone-Jiang/Picgo/main/image-20230518141503571.png" style="zoom:50%;" />



## coco数据集转换为VOC格式

这个程序在**tools/coco2voc.py**

处理时，建议将该文件拷贝到与数据集同级的目录下运行，

按照你的数据集定义，请按照文件注释修改变量

```python
savepath = ''
datasets_list = []
img_dir = ''
anno_dir = ''
classes_names = []

```



## 数据格式问题

**注意**，数据集是VOC格式读入的，即一个图像对应一个xml文件。

保存图像的上级文件夹必须要命名为”JPEGImages“

默认读入xml文件的文件夹要命名为”Annotations“，即使为空，这个文件夹也必须要存在

考虑到图像数据可能过大，请自行上传您数据集中**JPEGImages/**下所有图像到服务器端**datasets/myVocData/JPEGImages/**中



## 标注系统使用说明

**使用视频**详细见[作者Blibili]()，暂未上传

“StartLabel”页面，

本系统还设计了一系列快捷键，包括：“Ctrl+D”删除上一张标注，“Ctrl+U”使用“Use Self”模式，“Ctrl+A“使用“Auto Save”模式，“Ctrl+O”使用“Only Click”模式，“Ctrl+Save”保存当前图像和标注，“Ctrl+J”选择图像文件并打开，“Ctrl+X”选择另一个标注文件夹，“Ctrl+F”打开类别过滤功能，“左方向键”上翻一张图片，“右方向键”下翻一张图片。

* 补充

  **类别过滤功能**输入形式如常规python列表形式

    ```python
    ['dog','cat']
    ```
  
  使用该功能“StartLabel”页面将仅展示列表中的类别，为空会展示所有类别，且添加到训练集的图像也接受此过滤条件
  
  **画标注框**，该操作仅需要按住鼠标左键即开始，释放鼠标左键即停止，输入类别后，按下enter键即可
  
  **在展示框区域外按下鼠标右键**，该操作会唤起图像保存模式切换
  
  **标注框内鼠标右键**，该操作会自动匹配对应的标注框，并删除该标注框
  
  **所有被保存修改后的图像都会被添加到用于微调的数据集中**

"Inference"界面，

**每次按完run按钮一定要记得在运行好结果后按clean 啊，不然出说明错我就不知道了**



## 在使用辅助标注功能之前...

**！！！重要**

请[阅读小样本目标识别服务端构建](https://github.com/sixone-Jiang/fsodbydiffusiondet)以构建服务端程序，注意，该服务端程序必须在Linux系统上构建



## 核心配置文件

以下文件皆需要根据实际配置修改

1. **config/my_config.yaml**

```yaml
work_path: '/data/DiffusionDet/' # 服务程序的工作路径
gpu_id: '0,1,2,3' # 微调时使用n个GPU的编号
config_file: 'alicetest.diffdet.yaml' # 默认的模型配置文件路径
screen_name: 'alice_test' # 临时占用的screen窗口名
pretrained_model: 'test_voc_split1_pre_model.pth' # 调用models/下的模型
n_times_iter: 4 # 设置默认迭代次数的倍数
threshold: 0.4 # 设置默认阈值
```

2. **config/alice.diffdet.yaml**

该文件无需修改，如果运行时显示GPU内存不足，请调整IMS_PER_BATCH的数量

**注意**：IMS_PER_BATCH的数量要能被在**my_config.yaml**中设置的gpu数量整除

经过测算，单卡每批次为1进行微调时将占用**4G显存**

3. **config/default/info.yaml**

```yaml
CONNCECTOR:
  name: "Connector"
  description: "Connector"
  version: "1.0.0"
  data: '59.99.99.99:23@root#C:\\Users\\huip\\.ssh\\id_rsa' # 修改这里设置默认值
  # 密钥格式：ip@user#path_of_id_rsa
  # 密码格式：ip@user/password 如：59.99.99.99:23@root/password

# 以下内容并不会被识别，仅仅是保存在这里
SAVE:
  name: "Save"
  description: "Save"
  version: "1.0.0"
  data: ['59.99.99.99:22@root#C:\\Users\\huip\\.ssh\\id_rsa']
```

实际使用过程中，该文件中data下的数据可以不填，在操作页面手动输入即可



## 作者的悄悄话

这小破系统虽然好用，但是限于作者开发时间和精力的原因，搭建本系统是真的比较费力且不方便的....

