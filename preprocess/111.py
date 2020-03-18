#
# examples_num = 15300000
# epoch = 10
#
# examples_num_per_update = 4 * 4 * 2 * 16
#
# update_num = examples_num * epoch * 1.0 / examples_num_per_update
# print(update_num)
#
# need_time = update_num * 2.7 / 3600 /24
# print(need_time)
# print(need_time / 10.0 * 24)
# import torch
# x = torch.randn((1, 1), requires_grad=True)
# print(x.device)
# with torch.autograd.profiler.profile() as prof:
#     y = x ** 2
#     y.backward()
#     # NOTE: some columns were removed for brevity
#     print(prof)
#     print(type(prof))
print(sorted([4,2,1]))