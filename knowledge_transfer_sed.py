import numpy as np
import torch
import torch.nn as nn
from models import SupResNet, SSLResNet
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from sklearn.covariance import ShrunkCovariance
from sklearn import preprocessing
from utils import AverageMeter
from scipy.special import softmax

def get_public_dataset(args):
    transform = [transforms.Resize(args.size), transforms.ToTensor()]
    transform = transforms.Compose(transform)
    public_dataset = PublicDataset("./data/pseudo_imagenet", transform=transform)

    return public_dataset

class PublicDataset(datasets.ImageFolder):

    def __init__(self, *args, **kwargs):
        super(PublicDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):

        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class KnowledgeTransfer():

    def __init__(self, public_dataset, model):
        super(KnowledgeTransfer, self).__init__()
        self.public_dataset = public_dataset
        self.model = model

    def train(self, lr, args):
        public_loader = torch.utils.data.DataLoader(
            self.public_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            drop_last=False
        )
        optimizer = torch.optim.SGD(self.model.student.parameters(), lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        losses = AverageMeter('Loss', ':5.3f')
        for epoch in range(3):
            losses.reset()
            for i, data in enumerate(public_loader):
                images = data[0].cuda()
                with torch.cuda.amp.autocast(enabled=True):
                    logit, label = self.model(images)
                    loss = self.model.loss_function(logit, label)
                losses.update(loss.item())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            print("Distillation Epoch: [{}] \tLoss: {:.3f}".format(epoch + 1, losses.avg))
        self.model.cpu()
        torch.cuda.empty_cache()

        return self.model.student.state_dict()


class SEED(nn.Module):

    def __init__(self, model_dict_list, global_model_dict, args, dim=128, K=16384, t=0.07, temp=1e-4):

        super(SEED, self).__init__()

        self.K = K
        self.t = t
        self.temp = temp
        self.dim = dim

        # create the Teacher/Student encoders
        # num_classes is the output fc dimension
        if args.training_mode in ["SimCLR", "SupCon"]:
            self.teacher = SSLResNet(arch=args.arch, out_dim=dim).cpu()
            self.student = SSLResNet(arch=args.arch, out_dim=dim).cpu()
        elif args.training_mode == "SupCE":
            self.teacher = SupResNet(arch=args.arch, num_classes=args.num_classes).cpu()
            self.student = SupResNet(arch=args.arch, num_classes=args.num_classes).cpu()
        else:
            raise ValueError("training mode not supported")

        # create the queue
        num_clients = len(model_dict_list)
        self.queue_list = [torch.randn(dim, K) for _ in range(num_clients)]
        for i in range(num_clients):
            self.queue_list[i] = nn.functional.normalize(self.queue_list[i], dim=0)
            self.queue_list[i] = self.queue_list[i].cuda()

        self.queue_ptr_list = [torch.zeros(1, dtype=torch.long) for _ in range(num_clients)]

        self.student.load_state_dict(global_model_dict)
        self.student.train()
        self.teacher_model_list = model_dict_list
        self.freeze_teacher()

        self.student.cuda()
        self.teacher.cuda()

    # queue updation
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_index):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_list[queue_index])

        # replace the keys at ptr (de-queue and en-queue)
        if ptr + batch_size < self.K:
            self.queue_list[queue_index][:, ptr: ptr + batch_size] = keys.T
        else:
            self.queue_list[queue_index][:, ptr: self.K] = keys[: (self.K - ptr)].T
            self.queue_list[queue_index][:, 0: (batch_size - self.K + ptr)] = keys[(self.K - ptr):].T

        # move pointer
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr_list[queue_index][0] = ptr

    def freeze_teacher(self):
        # not update by gradient
        for param_k in self.teacher.parameters():
            param_k.requires_grad = False
        self.teacher.eval()


    def forward(self, image):
        num_clients = len(self.teacher_model_list)
        batch_size = image.shape[0]

        # compute query features
        s_emb = self.student(image)  # NxC
        s_emb = nn.functional.normalize(s_emb, dim=1)

        logit_stu_list = []
        logit_tea_list = []
        t_emb_list = []
        # compute key features
        for i in range(num_clients):
            with torch.no_grad():
                self.teacher.load_state_dict(self.teacher_model_list[i])
                self.freeze_teacher()

                t_emb = self.teacher(image)  # keys: NxC
                t_emb = nn.functional.normalize(t_emb, dim=1)
                t_emb_list.append(t_emb)
                logit_tea = torch.einsum('nc,ck->nk', [t_emb, self.queue_list[i].clone().detach()])
                logit_t_p = torch.einsum('nc,nc->n', [t_emb, t_emb]).unsqueeze(-1)
                logit_tea = torch.cat([logit_t_p, logit_tea], dim=1)
                logit_tea = nn.functional.softmax(logit_tea / self.temp, dim=1)
                logit_tea_list.append(logit_tea)

            logit_stu = torch.einsum('nc,ck->nk', [s_emb, self.queue_list[i].clone().detach()])
            logit_s_p = torch.einsum('nc,nc->n', [s_emb, t_emb]).unsqueeze(-1)
            logit_stu = torch.cat([logit_s_p, logit_stu], dim=1)
            logit_stu /= self.t
            logit_stu_list.append(logit_stu)

        for i in range(num_clients):
            self._dequeue_and_enqueue(t_emb_list[i], i)

        return logit_stu_list, logit_tea_list

    def loss_function(self, student_logit_list, teacher_logit_list):
        total_loss = 0
        num_clients = len(student_logit_list)
        batch_size = student_logit_list[0].shape[0]

        index = 0
        for student_logit, teacher_logit in zip(student_logit_list, teacher_logit_list):
            total_loss += -(teacher_logit * torch.nn.functional.log_softmax(student_logit, 1)).sum() / batch_size
            index += 1

        avg_loss = total_loss / num_clients
        return avg_loss

    def cpu(self):
        self.teacher.cpu()
        self.student.cpu()
        for i in range(len(self.queue_list)):
            self.queue_list[i] = self.queue_list[i].cpu()
