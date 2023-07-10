import torch

# # 仅保存和加载模型参数(推荐使用)
# torch.save(model_object.state_dict(), 'params.pkl')
# model_object.load_state_dict(torch.load('params.pkl'))
#
# # 保存和加载整个模型
# torch.save(model_object, 'model.pkl')
# model = torch.load('model.pkl')

# 第一种方式需要自己定义网络，并且其中的参数名称与结构要与保存的模型中的一致（可以是部分网络，比如只使用VGG的前几层），相对灵活，
# 便于对网络进行修改。第二种方式则无需自定义网络，保存时已把网络结构保存，比较死板，不能调整网络结构。