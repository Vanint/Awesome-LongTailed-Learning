import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable


class RSG(nn.Module):
    def __init__(self, n_center = 3, feature_maps_shape = [32, 16, 16], num_classes=10, contrastive_module_dim = 128, head_class_lists = [], transfer_strength = 1.0, epoch_thresh = 100):
        super(RSG, self).__init__()
        
        self.num_classes = num_classes
        self.C, self.H, self.W = feature_maps_shape

        self.n_center = n_center
        self.pooling = nn.AvgPool2d(self.H)
 
        self.linear = nn.Parameter(torch.randn(num_classes, self.C, n_center).cuda())
        self.bias = nn.Parameter(torch.ones(num_classes, n_center).cuda()) 

        self.centers = nn.Parameter(torch.zeros(num_classes, n_center, self.C).cuda())
        self.softmax = nn.Softmax(dim=1)
        self.strength =  transfer_strength
        self.epoch_thresh = epoch_thresh

        self.contrastive_module_dim = contrastive_module_dim

        self.vec_transformation_module = nn.Sequential(
                  nn.Conv2d(self.C, self.C,  kernel_size=3, stride=1, padding=1),
                )
          
        self.contrastive_module = nn.Sequential(
                  nn.Conv2d(self.C * 2, contrastive_module_dim,  kernel_size=1, stride=1, padding=0),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(contrastive_module_dim, contrastive_module_dim,  kernel_size=3, stride=1, padding=1),
                  nn.AvgPool2d(self.H)
                )

        self.contrastive_fc = nn.Linear(self.contrastive_module_dim, 2)

        for m in self.modules():
         if isinstance(m, nn.Conv2d):
          init.kaiming_normal_(m.weight)
          if m.bias is not None:
           init.zeros_(m.bias)
         elif isinstance(m, nn.Linear):
          init.normal_(m.weight)
          if m.bias is not None: 
           init.zeros_(m.bias)

    def compute_cesc_loss(self, feature_maps, c, gamma, target, epoch):
        num, C, H, W = feature_maps.size()
        gamma = gamma.unsqueeze(1)
   
        if epoch <= self.epoch_thresh:
         feature1 = feature_maps[0:num//2, :, :, :]
         feature2 = feature_maps[num//2:num, :, :, :]

         target1 = target[0 : num//2]
         target2 = target[num//2 : num]

         feature_cat = torch.cat([feature1, feature2], dim=1)
         target_cat = torch.eq(target1, target2).long()
        
         pair_fea = self.contrastive_module(feature_cat).contiguous().view(-1, self.contrastive_module_dim)
         pair_pred = self.contrastive_fc(pair_fea)
         loss = torch.sum(torch.bmm(gamma, torch.pow((feature_maps.unsqueeze(1).expand(-1, c.size()[1], -1, -1, -1) - c), 2).view(num, self.n_center, -1)))/(num) + F.cross_entropy(pair_pred, target_cat)
        else:
         loss = torch.sum(torch.bmm(gamma, torch.pow((feature_maps.unsqueeze(1).expand(-1, c.size()[1], -1, -1, -1) - c), 2).view(num, self.n_center, -1)))/(num)
        return loss

    def to_one_hot_vector(self, num_class, label):
        label = label.cpu().numpy()
        b = np.zeros((label.shape[0], num_class))
        b[np.arange(label.shape[0]), label] = 1
        b = torch.from_numpy(b)
        return b

    def compute_mv_loss(self, origin_feature, origin_center, target_center, target_features, gamma_head, target, gamma_tail):
        c = origin_center.detach()
        num, C, H, W = target_features.size()
        gamma_h = gamma_head.detach()
        gamma_t = gamma_tail.detach()
        ori_f = origin_feature.detach()
        c_ = target_center.detach()

        for p in self.contrastive_module.parameters():
            p.requires_grad = False
        for p in self.contrastive_fc.parameters():
            p.requires_grad = False

        index = gamma_h.argmax(dim=1)
        index_ = gamma_t.argmax(dim=1)
        index =  self.to_one_hot_vector(self.n_center, index).unsqueeze(1).cuda()
        index_ = self.to_one_hot_vector(self.n_center, index_).unsqueeze(1).cuda()

        c_o = torch.bmm(index, c.view(-1, self.n_center, self.H*self.W*self.C).double()).view(ori_f.size()).cuda()
        c_t = torch.bmm(index_, c_.view(-1, self.n_center, self.H*self.W*self.C).double()).view(ori_f.size()).cuda()

        var_map = (ori_f - c_o.float()).cuda()
        var_map_t = self.vec_transformation_module(var_map)

        target_features = target_features.cuda()

        target_features_vector = target_features - c_t.float()
        
        target_features_f = target_features + var_map_t
        target_features_norm = F.normalize(target_features_vector.view(-1, self.C), dim=1)
        var_map_norm = F.normalize(var_map_t.view(-1, self.C), dim=1)

        paired = torch.cat([ori_f, var_map_t], dim=1)

        pair_fea = self.contrastive_module(paired).contiguous().view(-1, self.contrastive_module_dim)
        pair_pred = self.contrastive_fc(pair_fea)

        loss = F.cross_entropy(pair_pred, torch.zeros(num).long().cuda()) + \
        (torch.sum(torch.abs(torch.norm(var_map_t.view(-1,self.C), dim=1) - torch.norm(var_map.view(-1, self.C), dim=1)))  + torch.sum(torch.abs(target_features_norm * var_map_norm - torch.ones(target_features_norm.size()).cuda())))/(num)

        return loss, target_features_f

    def forward(self, feature_maps, head_class_lists, target, epoch):
        maps_detach = feature_maps.detach()
        total = target.size()[0]
        num_head_list = len(head_class_lists)

        index_head = []
        index_tail = []
        
        head_class_lists_tensor = torch.Tensor(head_class_lists).cuda()
        head_class_lists_tensor = head_class_lists_tensor.unsqueeze(0).repeat(total, 1)
        target_expand = target.unsqueeze(1).repeat(1, num_head_list)
        
        index_head = torch.sum((target_expand == head_class_lists_tensor).long(), dim = 1).cuda()
        index_tail = 1 - index_head
        index_head_ = torch.eq(index_head, 1).cuda()
        index_tail_ = torch.eq(index_tail, 1).cuda()

        maps_detach_p = self.pooling(maps_detach).view(-1, self.C)
        target_select = target.unsqueeze(1)
        linear = self.linear[target_select,:,:].view(-1, self.C, self.n_center)
        bias = self.bias[target_select,:].view(-1, self.n_center)

        maps_detach_fc = torch.bmm(maps_detach_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
        gamma = self.softmax(maps_detach_fc)

        centers_ = self.centers[target_select,:,:].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])

        loss_cesc = self.compute_cesc_loss(maps_detach, centers_, gamma, target, epoch)
        loss_mv_total = torch.zeros(loss_cesc.size()).cuda()

        maps_tail = maps_detach[index_tail_,:,:,:]
        maps_head = maps_detach[index_head_,:,:,:]
        target_tail = target[index_tail_]
        target_head = target[index_head_]
        segment = 1

        num_tail = maps_tail.size()[0]
        num_head = maps_head.size()[0]
        
        if num_tail != 0 and num_head !=0 and epoch > self.epoch_thresh:
         if num_head >= num_tail:
          segment = int(num_head * self.strength / num_tail)
          if segment == 0:
             segment = 1

          for j in range(0, segment):
           latent_2 = maps_tail
           feature_origin =  maps_head[j * num_tail : (j + 1)*num_tail,:,:,:]

           maps_head_p = self.pooling(feature_origin).view(-1, self.C)
           target_head_select = target_head[j * num_tail : (j+1)* num_tail].unsqueeze(1)
           linear = self.linear[target_head_select, :, :].view(-1, self.C, self.n_center)
           bias = self.bias[target_head_select, :].view(-1, self.n_center)

           maps_head_fc = torch.bmm(maps_head_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
           gamma_head = self.softmax(maps_head_fc)
           center_origin = self.centers[target_head_select,:,:].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])

           maps_tail_p = self.pooling(latent_2).view(-1, self.C)
           target_tail_select = target_tail.unsqueeze(1)
           linear_ = self.linear[target_tail_select, :,:].view(-1, self.C, self.n_center)

           bias_ = self.bias[target_tail_select,:].view(-1, self.n_center)
           maps_tail_fc = torch.bmm(maps_tail_p.unsqueeze(1), linear_).view(-1, self.n_center) + bias_
           gamma_tail = self.softmax(maps_tail_fc)
           target_center = self.centers[target_tail_select,:,:].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
           loss_mv, feature_f = self.compute_mv_loss(feature_origin ,center_origin, target_center, latent_2, gamma_head, target_tail, gamma_tail)

           loss_mv_total += loss_mv

           feature_maps = torch.cat((feature_maps, feature_f) ,dim=0)
           target = torch.cat((target, target_tail), dim=0)
         else:
          segment = int(num_tail * self.strength / num_head)
          if segment == 0:
             segment = 1 
          for j in range(0, segment):
           latent_2 = maps_tail[j * num_head : (j + 1) * num_head,:,:,:]
           feature_origin =  maps_head

           maps_head_p = self.pooling(feature_origin).view(-1, self.C)
           target_head_select = target_head.unsqueeze(1)
           linear = self.linear[target_head_select, :, :].view(-1, self.C, self.n_center)
           bias = self.bias[target_head_select, :].view(-1, self.n_center)

           maps_head_fc = torch.bmm(maps_head_p.unsqueeze(1), linear).view(-1, self.n_center) + bias
           gamma_head = self.softmax(maps_head_fc)
           center_origin = self.centers[target_head_select,:,:].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])

           maps_tail_p = self.pooling(latent_2).view(-1, self.C)
           target_tail_select = target_tail[j * num_head : (j + 1) * num_head].unsqueeze(1)
           linear_ = self.linear[target_tail_select, :,:].view(-1, self.C, self.n_center)
           bias_ = self.bias[target_tail_select,:].view(-1, self.n_center)

           maps_tail_fc = torch.bmm(maps_tail_p.unsqueeze(1), linear_).view(-1, self.n_center) + bias_
           gamma_tail = self.softmax(maps_tail_fc)
           target_center = self.centers[target_tail_select,:,:].view(-1, self.n_center, maps_detach.size()[1]).unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, maps_detach.size()[2], maps_detach.size()[3])
           loss_mv, feature_f = self.compute_mv_loss(feature_origin ,center_origin, target_center, latent_2, gamma_head, target_tail, gamma_tail)
           feature_maps = torch.cat((feature_maps, feature_f) ,dim=0)

           loss_mv_total += loss_mv
           target = torch.cat((target, target_tail[j * num_head : (j + 1) * num_head]), dim=0)

        return feature_maps, loss_cesc, loss_mv_total/segment, target
