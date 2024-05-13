
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.arch_VHNet as arch_VHNet
import models.intraModule as intraModule
import models.fusionModule as fusionModule


class VHNet(nn.Module):
    def __init__(self, arg):
        #  nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None,
        #          predeblur=False, HR_in=False, w_TSA=True
        super(VHNet, self).__init__()
        # encoder
        self.front_RBs = arg.front_RBs
        self.nframes = arg.nframes
        self.center = int(self.nframes//2)
        self.ks = 3
        self.nf = 64
        self.groupID = [[0,3,6],[1,3,5],[2,3,4]]
        ################################################################

        self.FE_FG = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        # self.FE_FG_group2 = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        # self.FE_FG_group3 = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        self.FE_BG = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        # self.FE_BG_group2 = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        # self.FE_BG_group3 = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        self.FE_all = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        # self.FE_all_group2 = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        # self.FE_all_group3 = arch_VHNet.FeatureExtractor(self.front_RBs, self.nf, self.ks)
        ################################################################

        self.intraModuleGroup1 = intraModule.IntraModule4(3*self.nf)
        self.intraModuleGroup2 = intraModule.IntraModule4(3*self.nf)
        self.intraModuleGroup3 = intraModule.IntraModule4(3*self.nf)
        ################################################################

        self.FusionModuleAllGroups = fusionModule.FusionModule1(3*self.nf)

        ################################################################
        self.FD = arch_VHNet.FeatureDecoder(self.nf, self.ks)
        self.conv_last = nn.Conv2d(self.nf, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, frame, mask):
        Bf, Nf, Cf, Hf, Wf = frame.size()
        Bm, Nm, Cm, Hm, Wm = mask.size()

        background = frame * (1-mask)   # harmony
        foreground = frame * (mask)     # inharmony
        # torch.Size([4, 7, 3, 256, 256]) torch.Size([4, 7, 3, 256, 256])

        #####################################################################################
        # print(torch.unsqueeze(frame[:,self.groupID[0][2],:,:,:], 1).shape)
        frame_group1 = torch.cat((torch.unsqueeze(frame[:,self.groupID[0][0],:,:,:], 1),\
                                 torch.unsqueeze(frame[:,self.groupID[0][1],:,:,:], 1),\
                                 torch.unsqueeze(frame[:,self.groupID[0][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        frame_group2 = torch.cat((torch.unsqueeze(frame[:,self.groupID[1][0],:,:,:], 1),\
                                 torch.unsqueeze(frame[:,self.groupID[1][1],:,:,:], 1),\
                                 torch.unsqueeze(frame[:,self.groupID[1][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        frame_group3 = torch.cat((torch.unsqueeze(frame[:,self.groupID[2][0],:,:,:], 1),\
                                 torch.unsqueeze(frame[:,self.groupID[2][1],:,:,:], 1),\
                                 torch.unsqueeze(frame[:,self.groupID[2][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        foreground_group1 = torch.cat((torch.unsqueeze(foreground[:,self.groupID[0][0],:,:,:], 1),\
                                 torch.unsqueeze(foreground[:,self.groupID[0][1],:,:,:], 1),\
                                 torch.unsqueeze(foreground[:,self.groupID[0][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        foreground_group2 = torch.cat((torch.unsqueeze(foreground[:,self.groupID[1][0],:,:,:], 1),\
                                 torch.unsqueeze(foreground[:,self.groupID[1][1],:,:,:], 1),\
                                 torch.unsqueeze(foreground[:,self.groupID[1][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        foreground_group3 = torch.cat((torch.unsqueeze(foreground[:,self.groupID[2][0],:,:,:], 1),\
                                 torch.unsqueeze(foreground[:,self.groupID[2][1],:,:,:], 1),\
                                 torch.unsqueeze(foreground[:,self.groupID[2][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        background_group1 = torch.cat((torch.unsqueeze(background[:,self.groupID[0][0],:,:,:], 1),\
                                 torch.unsqueeze(background[:,self.groupID[0][1],:,:,:], 1),\
                                 torch.unsqueeze(background[:,self.groupID[0][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        background_group2 = torch.cat((torch.unsqueeze(background[:,self.groupID[1][0],:,:,:], 1),\
                                 torch.unsqueeze(background[:,self.groupID[1][1],:,:,:], 1),\
                                 torch.unsqueeze(background[:,self.groupID[1][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        background_group3 = torch.cat((torch.unsqueeze(background[:,self.groupID[2][0],:,:,:], 1),\
                                 torch.unsqueeze(background[:,self.groupID[2][1],:,:,:], 1),\
                                 torch.unsqueeze(background[:,self.groupID[2][2],:,:,:], 1)), 1).view(-1, Cf, Hf, Wf)
        #####################################################################################

        fea_frame_group1 = self.FE_all(frame_group1).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_frame_group2 = self.FE_all(frame_group2).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_frame_group3 = self.FE_all(frame_group3).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_foreground_group1 = (self.FE_FG(foreground_group1)).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_foreground_group2 = (self.FE_FG(foreground_group2)).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_foreground_group3 = (self.FE_FG(foreground_group3)).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_background_group1 = (self.FE_BG(background_group1)).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_background_group2 = (self.FE_BG(background_group2)).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)
        fea_background_group3 = (self.FE_BG(background_group3)).view(Bf,-1, 3*self.nf, Hf//4, Wf//4)

        outGroup1 = self.intraModuleGroup1(fea_frame_group1, fea_foreground_group1, fea_background_group1)
        outGroup2 = self.intraModuleGroup2(fea_frame_group2, fea_foreground_group2, fea_background_group2)
        outGroup3 = self.intraModuleGroup3(fea_frame_group3, fea_foreground_group3, fea_background_group3)

        AllGroups = torch.cat((torch.unsqueeze(outGroup1, 1), torch.unsqueeze(outGroup2, 1), torch.unsqueeze(outGroup3, 1)),1)
        fused = self.FusionModuleAllGroups(AllGroups)

        out = self.FD(fused)
        out = self.conv_last(out)
        
        out = frame[:,3,:,:,:] + out
        out_FG = out * mask[:,3,:,:,:]
        out_BG = frame[:,3,:,:,:] * (1-mask[:,3,:,:,:])
        out = out_BG + out_FG
        return out, out_FG


