class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMAPP():
    def __init__(self, model_lis, decay=0.999):
        self.ema_list = []
        for model in model_lis:
            self.ema_list.append(EMA(model, decay))

    def register(self):
        for ema in self.ema_list:
            ema.register()

    def update(self):
        for ema in self.ema_list:
            ema.update()

    def apply_shadow(self):
        for ema in self.ema_list:
            ema.apply_shadow()

    def restore(self):
        for ema in self.ema_list:
            ema.restore()