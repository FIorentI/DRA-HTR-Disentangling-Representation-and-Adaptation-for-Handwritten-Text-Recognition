import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_weights(task, dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1
        # READ2018 datasets
        if 'read_2018_general' in dsets and 'read_2018_30866' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general_2_2018_30866'], alpha['style']['read_2018_30866_2_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_30866'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_30866'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_30882' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general_2_2018_30882'], alpha['style']['read_2018_30882_2_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_30882'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_30882'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_30893' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general_2_2018_30893'], alpha['style']['read_2018_30893_2_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_30893'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_30893'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_35013' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general_2_2018_35013'], alpha['style']['read_2018_35013_2_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_35013'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_35013'] = 0.5, 1.0

        elif 'read_2018_general' in dsets and 'read_2018_35015' in dsets and 'U' not in dsets:
            alpha['style']['read_2018_general_2_2018_35015'], alpha['style']['read_2018_35015_2_2018_general'] = 5e4, 1e4
            alpha['dis']['read_2018_general'], alpha['dis']['read_2018_35015'] = 0.5, 0.5
            alpha['gen']['read_2018_general'], alpha['gen']['read_2018_35015'] = 0.5, 1.0

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = dict()
        self.alpha['style'], self.alpha['dis'], self.alpha['gen'] = dict(), dict(), dict()
        self.alpha['recon'], self.alpha['consis'], self.alpha['content'] = 5, 1, 1
        self.alpha['l_style'] = 1
        self.alpha['l_content'] = 1
        self.alpha['style'][args.dataset_source + '_2_' + args.dataset_target] = 5e4
        self.alpha['style'][args.dataset_target + '_2_' + args.dataset_source] = 1e4
        self.alpha['dis'][args.dataset_source] = 0.5
        self.alpha['dis'][args.dataset_target] = 0.5
        self.alpha['gen'][args.dataset_source] = 0.5
        self.alpha['gen'][args.dataset_target] =  1.0

    @staticmethod
    def _foreground_weights(target, fg_weight=0.0, mode="auto"):
        if fg_weight <= 0:
            return None
        target = target.detach().float().clamp(0.0, 1.0)
        if mode == "dark":
            foreground = 1.0 - target
        elif mode == "bright":
            foreground = target
        elif mode == "auto":
            mean_dims = tuple(range(1, target.dim()))
            mean_value = target.mean(dim=mean_dims, keepdim=True)
            foreground = torch.where(mean_value >= 0.5, 1.0 - target, target)
        else:
            raise ValueError(f"Unsupported reconstruction foreground mode: {mode}")
        return 1.0 + float(fg_weight) * foreground

    @staticmethod
    def _weighted_l1(pred, target, weights=None):
        values = (pred.float() - target.float()).abs()
        if weights is None:
            return values.mean()
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @classmethod
    def _gradient_l1(cls, pred, target, weights=None):
        losses = []
        if pred.size(-1) > 1:
            weight_x = None if weights is None else 0.5 * (weights[..., 1:] + weights[..., :-1])
            losses.append(cls._weighted_l1(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1], weight_x))
        if pred.size(-2) > 1:
            weight_y = None if weights is None else 0.5 * (weights[..., 1:, :] + weights[..., :-1, :])
            losses.append(cls._weighted_l1(pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :], weight_y))
        return sum(losses) / max(len(losses), 1)

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        fg_weight = float(getattr(self.args, "recon_foreground_weight", 0.0))
        fg_mode = str(getattr(self.args, "recon_foreground_mode", "auto"))
        edge_weight = float(getattr(self.args, "recon_edge_weight", 0.0))
        for dset in imgs.keys():
            weights = self._foreground_weights(imgs[dset], fg_weight=fg_weight, mode=fg_mode)
            recon_loss += self._weighted_l1(recon_imgs[dset], imgs[dset], weights)
            if edge_weight > 0:
                recon_loss += edge_weight * self._gradient_l1(recon_imgs[dset], imgs[dset], weights)
        return self.alpha['recon'] * recon_loss
    
    def latent_consistency(self, imgs, converted_imgs, case):
        latent_consistency_loss = F.l1_loss(imgs, converted_imgs)
        return self.alpha[case] * latent_consistency_loss
    
    def latent_cosine_consistency(self, source, converted, case):
        latent_cosine_consistency_loss = 1 - F.cosine_similarity(
                source.flatten(1),
                converted.flatten(1)
            ).mean()
        return self.alpha[case] * latent_cosine_consistency_loss

    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf' and self.args.discriminator != "PatchGAN":  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split("_2_", 1)
                # target = "read_" + target
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg' or self.args.discriminator == "PatchGAN":  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split("_2_", 1)
                # target = "read_" + target
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss


    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split("_2_", 1)
            # target = "read_" + target
            if self.args.task == 'clf' and self.args.discriminator != "PatchGAN":  # DCGAN loss
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg' or self.args.discriminator == "PatchGAN":  # Hinge loss
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    
    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split("_2_", 1)
            # target = "read_" + target
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split("_2_", 1)
            # target = "read_" + target
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr],
                                                                             style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split("_2_", 1)
            # target = "read_" + target
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '_2_' in key:
                source, target = key.split('_2_')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss

    def task_CRNN(self, pred, ctc_loss, y_enc, x_reduced_len, y_len_enc):
        # task_loss = 0

        task_loss = None
        loss_enc_main_epoch = 0
        loss_enc_shortcut_epoch = 0
        weight_loss_shortcut = 0.1

        for key in pred.keys():
            if '_2_' in key:
                source, target = key.split('_2_')
            else:
                source = key
            # task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
            y, _, after_blstm = pred[key]
            output, aux_output = y
            output = torch.nn.functional.log_softmax(output, dim=-1)
            loss = ctc_loss(output.cpu(), y_enc[source], x_reduced_len[source], y_len_enc[source])
            # loss_enc_main_epoch += loss.detach().item()

            if aux_output is not None:
                aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)
                loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc[source], x_reduced_len[source], y_len_enc[source])
                # loss_enc_shortcut_epoch += loss_shortcut.detach().item()
                loss += loss_shortcut
            task_loss = loss if task_loss is None else task_loss + loss
        return task_loss
    
    def single_task_CRNN(self, pred, source, ctc_loss, y_enc, x_reduced_len, y_len_enc):
        weight_loss_shortcut = 0.1

        y, _, after_blstm = pred
        output, aux_output = y
        output = torch.nn.functional.log_softmax(output, dim=-1)
        loss = ctc_loss(output, y_enc[source], x_reduced_len[source], y_len_enc[source])

        if aux_output is not None:
            aux_output = torch.nn.functional.log_softmax(aux_output, dim=-1)
            loss_shortcut = weight_loss_shortcut * ctc_loss(aux_output.cpu(), y_enc[source], x_reduced_len[source], y_len_enc[source])
            loss += loss_shortcut
        return loss
