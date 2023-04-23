import torch
from torch.nn import functional as F
from torch_scatter import scatter

class trainer_mfd(object):
    def __init__(self, model, optimizer, scheduler, ema, datatype_list, loss_weights, embedding_decay):
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.datatype_list = datatype_list
        self.loss_weights = loss_weights
        self.embedding_decay = embedding_decay
        self.iter = 0

    def train(self):
        self.model.train()
        if len(self.ema.backup) == 0 or self.iter == 0:
            pass
        else:
            self.ema.restore()

    def train_on_batch(self, batches, train_metrics):
        loss_list = []
        loss_mean = []
        for task_id, (batch, tasktype) in enumerate(zip(batches, self.datatype_list)):
            batch = batch.to(self.device)
            output = self.model(batch=batch, task_id=task_id)
            if tasktype == 'r':
                loss = nan_mean_loss(output, batch.y, F.l1_loss)
                loss_list.append(loss.detach().cpu().item())
                loss_mean.append(loss)
            elif tasktype == 'c':
                loss = label_smoothing_loss(output, batch.y.flatten().round().long())
                # loss = F.cross_entropy(output, batch.y.view(-1).round().long())
                loss_list.append(loss.detach().cpu().item())
                loss_mean.append(loss)
        loss = (torch.stack(loss_mean) * self.loss_weights).sum()
        loss_item = loss.detach().cpu().item()
        loss = loss + self.embedding_decay * self.model.model.task_embedding_atom.square().sum()/(2*sum([batch.num_graphs for batch in batches]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()
        self.iter += 1
        train_metrics.append({'iteration': self.iter, 'loss': loss_item, 'loss_list': loss_list})
        del(loss, loss_mean, output)
        return [loss_item]+loss_list

    def eval(self):
        self.model.eval()
        if len(self.ema.shadow) == 0 or self.iter == 0:
            pass
        else:
            self.ema.apply_shadow()

    @torch.no_grad()
    def test_on_batch(self, loader_list, val_metrics):
        loss_list = []
        result_dict = []
        for task_id, (loader, tasktype) in enumerate(zip(loader_list, self.datatype_list)):
            loss_batch = []
            nsample_batch = []
            result_target_list = []
            result_predict_list = []
            for batch in loader:
                batch = batch.to(self.device)
                output = self.model(batch=batch, task_id=task_id)
                nsample_batch.append(output.shape[0])
                if tasktype == 'r':
                    loss = nan_mean_loss(output, batch.y, F.l1_loss)
                    loss_batch.append(loss.detach().cpu().item())
                elif tasktype == 'c':
                    loss = F.cross_entropy(output, batch.y.view(-1).round().long())
                    loss_batch.append(loss.detach().cpu().item())
                result_target_list.append(batch.y.detach().cpu())
                result_predict_list.append(output.detach().cpu())
            result_dict.append({'target': torch.concat(result_target_list, dim=0).numpy(), 'predict': torch.concat(result_predict_list, dim=0).numpy()})
            loss_mean = (torch.Tensor(loss_batch)*torch.Tensor(nsample_batch)).sum().item()/sum(nsample_batch)
            loss_list.append(loss_mean)
        loss_sum = (torch.Tensor(loss_list) * self.loss_weights.cpu()).sum().detach().cpu().item()
        val_metrics.append({'iteration': self.iter, 'loss': loss_sum, 'loss_list': loss_list})
        del(loss, output)
        return [loss_sum] + loss_list, result_dict

    @torch.no_grad()
    def predict_on_batch(self, batch, task_id):
        output = self.model(batch=batch, task_id=task_id)
        return output

    def save_state_dict_step(self, step_ckpt_folder):
        self.ema.apply_shadow()
        torch.save(self.model.state_dict(), step_ckpt_folder+'/model.pt')

    def save_state_dict_best(self, best_ckpt_file):
        self.ema.apply_shadow()
        torch.save(self.model.state_dict(), best_ckpt_file)



class trainer_scratch(object):
    def __init__(self, model, optimizer, scheduler, ema, datatype):
        self.model = model
        self.device = next(model.parameters()).device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.tasktype = datatype
        self.iter = 0

    def train(self):
        self.model.train()
        if len(self.ema.backup) == 0 or self.iter == 0:
            pass
        else:
            self.ema.restore()

    def train_on_batch(self, batch, train_metrics):
        batch = batch.to(self.device)
        output = self.model(data=batch)
        if self.tasktype == 'r':
            loss = nan_mean_loss(output, batch.y, F.l1_loss)
        elif self.tasktype == 'c':
            loss = label_smoothing_loss(output, batch.y.flatten().round().long())
            # loss = F.cross_entropy(output, batch.y.view(-1).round().long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()
        self.iter += 1
        train_metrics.append({'iteration': self.iter, 'loss': loss.detach().cpu().item()})
        loss_item = loss.detach().cpu().item()
        del(loss, output)
        return loss_item


    def eval(self):
        self.model.eval()
        if len(self.ema.shadow) == 0 or self.iter == 0:
            pass
        else:
            self.ema.apply_shadow()

    @torch.no_grad()
    def test_on_batch(self, loader, val_metrics):
        nsample_batch = []
        result_target_list = []
        result_predict_list = []
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(data=batch)
            nsample_batch.append(output.shape[0])
            if self.tasktype == 'r':
                loss = nan_mean_loss(output, batch.y, F.l1_loss)
            elif self.tasktype == 'c':
                loss = F.cross_entropy(output, batch.y.view(-1).round().long())
            result_target_list.append(batch.y.detach().cpu())
            result_predict_list.append(output.detach().cpu())
        val_metrics.append({'iteration': self.iter, 'loss': loss.detach().cpu().item()})
        result_dict = {'target': torch.concat(result_target_list, dim=0).numpy(), 'predict': torch.concat(result_predict_list, dim=0).numpy()}
        loss_item = loss.detach().cpu().item()
        del(loss, output)
        return loss_item, result_dict

    @torch.no_grad()
    def predict_on_batch(self, batch):
        output = self.model(data=batch)
        return output

    def save_state_dict_step(self, step_ckpt_folder):
        self.ema.apply_shadow()
        torch.save(self.model.state_dict(), step_ckpt_folder+'/model.pt')

    def save_state_dict_best(self, best_ckpt_file):
        self.ema.apply_shadow()
        torch.save(self.model.state_dict(), best_ckpt_file)



def nan_mean_loss(predict: torch.Tensor, target: torch.Tensor, loss):
    """
    loss is a torch loss function
    """
    finite_bool = torch.isfinite(target).flatten().to(target.device)
    batch_idx = torch.arange(target.shape[0]).unsqueeze(1).repeat([1,target.shape[1]]).to(target.device)
    loss = scatter(src=loss(input=predict.flatten()[finite_bool], target=target.flatten()[finite_bool], reduction='none'), index=batch_idx.flatten()[finite_bool], dim_size=target.shape[0], reduce='mean').mean()
    return loss


def label_smoothing_loss(pred, target, smoothing=0.1):
    num_classes = pred.shape[1]
    pred = pred.log_softmax(dim=1)
    true_dist = torch.zeros_like(pred)
    negative = smoothing / num_classes
    positive = (1 - smoothing) + negative
    true_dist.fill_(negative)
    true_dist.scatter_(1, target.unsqueeze(1), positive)
    return torch.sum(-true_dist*pred, dim=1).mean()