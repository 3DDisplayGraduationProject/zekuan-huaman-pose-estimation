# wget https://github.com/wolverinn/human-pose-estimation-GAN/raw/master/auto.sh && bash auto.sh
# tensorflow==1.14
# 确认src/main.py中使用的device是选gpu还是cpu，以及config.py中的batch_size和数据集(+use_3d_lable)和logs目录下是否有新文件夹

mkdir hmr
cd hmr/
mkdir models
mkdir logs

cd ./models/
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/neutral_smpl_mean_params.h5
wget http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz
mkdir resnet_v2_101
tar -xf resnet_v2_101_2017_04_14.tar.gz -C ./resnet_v2_101
tar -xf models.tar.gz
mv ./models/neutral_smpl_with_cocoplus_reg.pkl ./neutral_smpl_with_cocoplus_reg.pkl
rm resnet_v2_101_2017_04_14.tar.gz
rm models.tar.gz
cd ..

pip3 install deepdish --user
pip3 install chumpy --user
# pip3 install gast==0.2.2 --user

# get train code from github
wget https://github.com/wolverinn/human-pose-estimation-GAN/archive/master.zip
unzip master.zip
rm master.zip
mv ./human-pose-estimation-GAN-master/src_modify ./src
mv ./human-pose-estimation-GAN-master/start.sh ./start.sh
rm -rf human-pose-estimation-GAN-master

# get processed data tf_record
wget https://github.com/wolverinn/human-pose-estimation-GAN/releases/download/v1.0/tf1.tar.gz
tar -xf tf1.tar.gz
mv ./tf1 ./tf_records
rm tf1.tar.gz

wget https://github.com/wolverinn/human-pose-estimation-GAN/releases/download/v1.1/tf2.tar.gz
tar -xf tf2.tar.gz
mv ./mocap_neutrMosh ./tf_records/mocap_neutrMosh
rm tf2.tar.gz

wget https://github.com/wolverinn/human-pose-estimation-GAN/releases/download/v1.2/tf3.tar.gz
tar -xf tf3.tar.gz
mv ./coco_pre ./tf_records/coco
rm tf3.tar.gz

wget https://github.com/wolverinn/human-pose-estimation-GAN/releases/download/v1.3/tf4.tar.gz
tar -xf tf4.tar.gz
mv ./coco/* ./tf_records/coco/
rm tf4.tar.gz

# 3d mpi_inf_3dhp
wget https://github.com/wolverinn/human-pose-estimation-GAN/releases/download/v1.5/tf6.tar.gz
tar -xf tf6.tar.gz
mv ./mpi_inf_3dhp ./tf_records/mpi_inf_3dhp
rm tf6.tar.gz

wget https://github.com/wolverinn/human-pose-estimation-GAN/releases/download/v1.4/tf5.tar.gz
tar -xf tf5.tar.gz
mv ./mpi_inf_3dhp/train_pre/* ./tf_records/mpi_inf_3dhp/train/

# 最后运行 python3 -m src.main 即可，或者使用：
# bash start.sh