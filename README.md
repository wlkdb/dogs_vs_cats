# 猫狗大战  
数据集下载：https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data  
下载后在根目录解压即可  
各程序运行用时及成功后截图见success screenshot文件  
  
## 运行环境  
Python 3.5.4，Tensorflow-gpu 1.1.0  
论文中使用的是配置了2个Tesla M60的美团云GPU主机  
由于截图时美团云GPU主机售罄，故success screenshot文件中改为使用配置了1个Tesla P100的Google云GPU主机  
  
  
## 数据分析  
切换至data_analysis目录下  
  
### 分析训练集图片尺寸    
python data_analysis.py    
  
### 分析测试集图片尺寸    
python data_analysis.py --path ../test    
  
### 创建训练集与测试集样本以用于特征观察（已存在）    
python create_sample.py    
    

## 从头训练  
切换至train_from_beginning目录下  
  
### 创建小数据集  
python create_train_small_set.py  
  
### 创建验证集  
python create_verification_set.py  
  
### 训练  
python train.py  
参数设置  
&emsp;&emsp; * train_dir: 文件夹地址   
    训练集的位置，默认为 train_s  
&emsp;&emsp; * checkpoint_path: 文件夹地址  
    checkpoint保存位置，默认为 checkpoint  
&emsp;&emsp; * data_preprocess: 0..8   
    数据预处理方式，默认为 8。0为初始处理；1为图像标准化处理；2为长宽分别裁剪4/5；3为长宽分别裁剪3/4。  
    4-8在长宽分别裁剪3/4的基础上：4增加了随机水平翻转；5增加了随机覆盖；6增加了随机亮度与对比度；  
    7增加了随机饱和度与色差；8增加了随机水平翻转，随机亮度与对比度  
&emsp;&emsp; * net_mode: 0..2  
    网络结构，默认为 0。0为Resnet_v2_18；1为Resnet_v2_34；2为Resnet_v2_50  
&emsp;&emsp; * opt_mode: 0..2  
    优化器类型，默认为 1。0为GradientDescentOptimizer；1为MomentumOptimizer；2为AdamOptimizer  
&emsp;&emsp; * lr_mode: 0..5  
    学习速率，默认为 1。0为固定0.01；1为固定0.001；2为固定0.0001；3为固定比率衰减；4为自动衰减；5为周期性余弦衰减  
&emsp;&emsp; * batch_size: 正整数  
    每批次数据大小，默认为64  
          
### 验证  
python verify.py  
参数设置  
&emsp;&emsp; * checkpoint_path: 文件夹地址  
        checkpoint保存位置  
&emsp;&emsp; * net_mode: 0..2  
       网络结构，默认为 0。0为Resnet_v2_18；1为Resnet_v2_34；2为Resnet_v2_50  
          
### 测试，结果输出到本目录下 output.csv  
python test.py  
    参数设置  
&emsp;&emsp; * checkpoint_path: 文件夹地址  
        checkpoint保存位置   
&emsp;&emsp; * net_mode: 0..2  
       网络结构，默认为 0。0为Resnet_v2_18；1为Resnet_v2_34；2为Resnet_v2_50  
  
  
## 迁移学习  
切换至transfer_learning目录下，其中的大部分文件都是以slim中的原文件为基础，  
运行文件时需要再手动修改配置的地方以 \*\*\* 标明   
  
### 创建小数据集    
python create_train_small_set.py  
python download_and_convert_data.py \   --dataset_name=flowers \   --dataset_dir=train_s  
  
### 创建全数据集  
python create_train_whole_set.py  
python download_and_convert_data.py \   --dataset_name=flowers \   --dataset_dir=train  
  
### 创建验证集  
将datasets/download_and_convert_flowers.py中的_NUM_VALIDATION改为10000  
python create_verification_set.py  
python download_and_convert_data.py \   --dataset_name=flowers \   --dataset_dir=verify  
  
### 下载预训练模型  
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz  
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz  
tar zxvf vgg_16_2016_08_28.tar.gz   
tar zxvg resnet_v1_50_2016_08_28.tar.gz   
  
### 微调基准模型Vgg16  
python train.py   --train_dir=checkpoint_vgg16   --dataset_name=flowers   --dataset_dir=train_s   --model_name=vgg_16   --checkpoint_path=vgg_16.ckpt  --checkpoint_exclude_scopes=vgg_16/fc8    --trainable_scopes=vgg_16/fc8   --max_number_of_steps=3000   --batch_size=16   --learning_rate=0.01   --learning_rate_decay_type=fixed   --save_interval_secs=600   --save_summaries_secs=600   --log_every_n_steps=100   --optimizer=rmsprop  --weight_decay=0.00004  
  
### 微调Resnet_v1_50  
python train.py   --train_dir=checkpoint_resnet_v1_50   --dataset_name=flowers   --dataset_dir=train_s   --model_name=resnet_v1_50   --checkpoint_path=resnet_v1_50.ckpt   --max_number_of_steps=3000   --batch_size=16   --learning_rate=0.001   --save_interval_secs=300   --save_summaries_secs=300   --log_every_n_steps=100   --optimizer=adam   --weight_decay=0.00004   --checkpoint_exclude_scopes=resnet_v1_50/logits   --trainable_scopes=resnet_v1_50/logits_new, resnet_v1_50/block4     
使用全训练集时需修改datasets/flowers中的SPLITS_TO_SIZES；然后将训练参数dataset_dir设为train；将max_number_of_steps适当调高（如12000）  
关于模型微调方式的设置：  
&emsp;&emsp; * 用新输出层替换旧输出层：在nets/resnet_v1.py中将旧输出层注释掉；训练参数trainable_scopes设为resnet_v1_50/logits_new  
&emsp;&emsp; * 用带隐层的新输出层替换旧输出层：在nets/resnet_v1.py中保留旧输出层；训练参数checkpoint_exclude_scopes设为resnet_v1_50/logits；trainable_scopes设为resnet_v1_50/logits_new， resnet_v1_50/logits  
&emsp;&emsp; * 加入新输出层，保留旧输出层并对其进行训练：在nets/resnet_v1.py中保留旧输出层；训练参数checkpoint_exclude_scopes设为空；trainable_scopes设为resnet_v1_50/logits_new， resnet_v1_50/logits  
&emsp;&emsp; * 加入新输出层，保留旧输出层而不对其进行训练：在nets/resnet_v1.py中保留旧输出层；训练参数checkpoint_exclude_scopes设为空；trainable_scopes设为resnet_v1_50/logits_new  
&emsp;&emsp; * 用新输出层替换旧输出层，并训练最后一个block：在nets/resnet_v1.py中将旧输出层注释掉；训练参数trainable_scopes设为resnet_v1_50/logits_new，resnet_v1_50/block4   
  
### 验证Resnet_v1_50  
python verify.py   --checkpoint_path=checkpoint_resnet_v1_50   --eval_dir=verify/data   --dataset_name=flowers   --dataset_split_name=validation   --dataset_dir=verify   --model_name=resnet_v1_50    
  
### 测试Resnet_v1_50  
python test.py   --checkpoint_path checkpoint_resnet_v1_50   --model_name resnet_v1_50   --infile ../test   
  
  
## 结果统计  
切换至results_show目录下  
python draw_results.py  
  
## Android  
切换至android目录下  
前往 https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android 下载Tensorflow Android Camare demo，  
将android文件夹中的文件替换demo中assets文件夹的原始文件，并在 ClassifierActivity.java 文件中进行相应修改:  
&emsp;&emsp; * INPUT_SIZE 设为 208  
&emsp;&emsp; * IMAGE_MEAN 设为 0  
&emsp;&emsp; * IMAGE_STD 设为 1  
&emsp;&emsp; * MODEL_FILE 设为 "file:///android_asset/cats_and_dogs.pb"  
&emsp;&emsp; * LABEL_FILE 设为 "file:///android_asset/cats_and_dogs.txt"  
  
  
## 微信小程序  
切换至weixin目录下  
将相同迁移学习模型下的checkpoint相关文件移至weixin/checkpoint目录下  
运行 python get_img_category.py 将识别结果输出至控制台  
搭建服务端时，服务端将收到的图片保存，并调用  python get_img_category.py --infile *图片文件位置*  来获取识别结果  
越接近0越可能为猫，越接近1越可能为狗  
  

