# mobilenet_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf


# 导入本地数据集
train_dir = "F:.../train"  # 训练集路径
val_dir = "F:.../val"  # 验证集路径

# 设置图片的高和宽，一次训练所选取的样本数，迭代次数
im_height = xx
im_width = xx
batch_size = 64
epochs = 13

# 对数据进行增强
train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                           rotation_range=40,  # 旋转范围
                                           width_shift_range=0.2,  # 水平平移范围
                                           height_shift_range=0.2,  # 垂直平移范围
                                           shear_range=0.2,  # 剪切变换的程度
                                           zoom_range=0.2,  # 缩放范围
                                           horizontal_flip=True,  # 水平翻转
                                           fill_mode='nearest')

# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,  # 打乱数据
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
# 训练集样本数
total_train = train_data_gen.n

# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # 归一化

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                              batch_size=batch_size,
                                                              shuffle=False,  # 不打乱数据
                                                              target_size=(im_height, im_width),
                                                              class_mode='categorical')
# 验证集样本数
total_val = val_data_gen.n

# 使用tf.keras.applications中的MobileNetV2网络，并且使用官方的预训练模型
covn_base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
covn_base.trainable = True

# 将原网络前面的层冻结，只训练最后x层
for layers in covn_base.layers[:-x]:
    layers.trainable = False

# 构建模型
model = tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalAveragePooling2D())  # 加入全局平均池化层
model.add(tf.keras.layers.Dense(5, activation='softmax'))  # 加入输出层(5分类)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # 使用adam优化器，学习率为0.0001
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 交叉熵损失函数
              metrics=["accuracy"])  # 评价函数
# 每层参数信息
model.summary()

# 开始训练
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size)
                    
                    

# 记录训练集和验证集的准确率和损失值
history_dict = history.history
train_loss = history_dict["loss"]
train_accuracy = history_dict["accuracy"]
val_loss = history_dict["val_loss"]
val_accuracy = history_dict["val_accuracy"]

# 绘制损失值
plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# 绘制准确率
plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
