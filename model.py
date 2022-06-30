class CascadeNetStrong(nn.Module):
    def __init__(self, iou = (0.5, 0.6, 0.8, 0.9), conf=0.45, filters=24):
        super(CascadeNetStrong, self).__init__()
        self.conf = conf
        filters = 24
        
        
        backbone = [
                    nn.Conv2d(1, 24, kernel_size = 3, stride=2, padding=1, dilation=1),
                    nn.ReLU(),
                    nn.Conv2d(24, 24, kernel_size = 3, stride=1, padding=1, dilation=1),
                    nn.ReLU(),
                    nn.Conv2d(24, 24, kernel_size = 3, stride=2, padding=1, dilation=1),
                    nn.ReLU(),
                    nn.Conv2d(24, 24, kernel_size = 3, stride=1, padding=1, dilation=1),
                    nn.ReLU(), 
                    nn.Conv2d(24, 24, kernel_size = 3, stride=1, padding=2, dilation=2),
                    nn.ReLU(), 
                    nn.Conv2d(24, 24, kernel_size = 3, stride=1, padding=4, dilation=4),
                    nn.ReLU(),
                    nn.Conv2d(24, 24, kernel_size = 3, stride=1, padding=8, dilation=8),
                    nn.ReLU(),
                    nn.Conv2d(24, 24, kernel_size = 3, stride=1, padding=16, dilation=16),
                    nn.ReLU(),
                ]
        
        self.backbone = nn.Sequential(*backbone)
        
        self.first_downscale = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size = 3, stride=4, padding=1),
            nn.ReLU(),)
        
        self.first_classification = nn.Sequential(
            nn.Conv2d(filters, 1, kernel_size = 1, stride=1, padding=0),
            nn.Sigmoid(),)
        
        self.second_classification = nn.Sequential(
            nn.Conv2d(filters + 2, filters, kernel_size = 3, stride=1, padding=1, dilation=1),
                    nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = 3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(filters, 1, kernel_size = 1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        self.third_classification = nn.Sequential(
            nn.Conv2d(filters + 1, filters, kernel_size = 3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = 3, stride=1, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(filters, 1, kernel_size = 1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        self.last_downscale = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size = 3, stride=4, padding=1),
            nn.ReLU(),)
        
        self.forth_classification = nn.Sequential(
            nn.Conv2d(filters + 2, filters, kernel_size = 3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = 3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size = 3, stride=1, padding=8, dilation=8),
            nn.ReLU(),
            nn.Conv2d(filters, 1, kernel_size = 1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        self.iou = iou
        
    def get_iou(self, shape, countour1, countour2):
        contours = [countour1, countour2]
        blank = np.zeros(shape)
        
        image1 = cv2.drawContours(blank.copy(), contours, 0, 1, thickness=cv2.FILLED)
        image2 = cv2.drawContours(blank.copy(), contours, 1, 1, thickness=cv2.FILLED)
        intersection = np.logical_and(image1, image2).sum()
        union = np.logical_or(image1, image2).sum() + 1e-8
        return intersection / union
        
    def find_rest(self, pred, gt, iou, only_found=False, max_pr=False):
        
        pred_numpy = (pred.cpu().detach().numpy())
        gt_numpy = (gt.cpu().detach().numpy() * 255).astype(np.uint8)
        
        mask = np.zeros(pred_numpy.shape).astype(np.uint8)
        mask_gt = np.zeros(pred_numpy.shape).astype(np.uint8)
        for batch in range(pred.shape[0]):
            shape = pred_numpy[batch][0].shape
            cont_pred = cv2.findContours((pred_numpy[batch][0] >= self.conf).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            cont_gt = cv2.findContours(gt_numpy[batch][0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            for pr in cont_pred:
                if not only_found:
                    found = False
                else:
                    found = True
                gs = None
                for g in cont_gt:
                    cur_iou = self.get_iou(shape, g, pr)
                    if  cur_iou >= iou:
                        found = True
                        gs = g
                    if found:
                        break
                if found:
                    mask[batch, 0] = cv2.drawContours(mask[batch, 0], [pr], 0, 1, thickness=cv2.FILLED)
                    mask[batch, 0] = cv2.drawContours(mask_gt[batch, 0], [gs], 0, 1, thickness=cv2.FILLED)
                    
                    mask_gt[batch, 0] = cv2.drawContours(mask_gt[batch, 0], [pr], 0, 1, thickness=cv2.FILLED)
                    mask_gt[batch, 0] = cv2.drawContours(mask_gt[batch, 0], [gs], 0, 1, thickness=cv2.FILLED)
        return pred * torch.tensor(1 - mask).to(device), gt * torch.tensor(1 - mask_gt).to(device)
        
        
    def forward(self, x, gt = None):
       
        x1__ = self.first_downscale(x)
        x1_ = self.last_downscale(x)
        x = self.backbone(x)
        first_classification = self.first_classification(x)
        
        rest_data_to_classify1 = self.find_rest(first_classification, gt, self.iou[0])
        
        
        x1 = torch.cat([x, first_classification, x1__], dim=1)

        second_classification = self.second_classification(x1)
        x1 = None
        rest_data_to_classify2 = self.find_rest(second_classification, gt, self.iou[1])

        x2 = torch.cat([x, second_classification], dim=1)

        third_classification = self.third_classification(x2)
        x2 = None
        rest_data_to_classify3 = self.find_rest(third_classification, gt, self.iou[2], max_pr=True)
        
        
        
        x3 = torch.cat([x, third_classification, x1_], dim=1)
        
        forth_classification = self.forth_classification(x3)
        x3 = None
        rest_data_to_classify4 = self.find_rest(forth_classification, gt, self.iou[3], max_pr=True)
        
        # loss, metric
        return (first_classification, second_classification, third_classification, forth_classification),\
            (rest_data_to_classify1, rest_data_to_classify2, rest_data_to_classify3, rest_data_to_classify4)
    
    def loss(self, xs1):
        sum_loss = 0.
        for i in range(len(xs1)):
            sum_loss += (self.iou_loss(xs1[i][1], xs1[i][0]) + self.bce_loss(xs1[i][1], xs1[i][0]) + self.dice_loss(xs1[i][1], xs1[i][0])) / 3
        return sum_loss
    
    def iou_loss(self, true, pred):
        intersection = true * pred

        notTrue = 1 - true
        union = true + (notTrue * pred)

        return torch.mean(1 - torch.sum(intersection, dim=(2, 3)) / torch.clip(torch.sum(union, dim=(2, 3)), eps, 1 / eps))

    def bce_loss(self, true, pred):
        los = nn.BCELoss()
        return los(pred, true)

    def dice_loss(self, true, pred):
        intersection = 2 * true * pred
        f = true ** 2
        s = pred ** 2

        return  torch.mean(1 - torch.sum(intersection, dim=(2,3))/ torch.clip(torch.sum(f + s, dim=(2,3)), eps, 1 / eps))

