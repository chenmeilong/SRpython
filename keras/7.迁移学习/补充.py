# 场景：你要用自己的训练完的模型，作为下一个模型初始化的权重，譬如inceptionv3中的no_top版本。
# 如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型：
# model.load_weights(‘my_model_weights.h5’, by_name=True)
# 例如：
#
# 假如原模型为：
# model = Sequential()
# model.add(Dense(2, input_dim=3, name="dense_1"))
# model.add(Dense(3, name="dense_2"))
# ...
# model.save_weights(fname)
#
#
# # new model
# model = Sequential()
# model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
# model.add(Dense(10, name="new_dense"))  # will not be loaded
#
# # load weights from first model; will only affect the first layer, dense_1.
# model.load_weights(fname, by_name=True)

conda config --set show_channel_urls yes